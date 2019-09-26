import datetime
import json
import logging
import os
from multiprocessing.pool import Pool
from pathlib import Path
import numpy as np
from tensorboardX import SummaryWriter

import rl_agents.trainer.logger
from rl_agents.agents.common.factory import load_environment, load_agent
from rl_agents.agents.common.graphics import AgentGraphics
from rl_agents.agents.common.memory import Transition
from rl_agents.utils import near_split, zip_with_singletons
from rl_agents.configuration import serialize
from rl_agents.trainer.graphics import RewardViewer
from rl_agents.trainer.monitor import MonitorV2

logger = logging.getLogger(__name__)


class Evaluation(object):
    """
        The evaluation of an agent interacting with an environment to maximize its expected reward.
    """

    OUTPUT_FOLDER = 'out'
    SAVED_MODELS_FOLDER = 'saved_models'
    RUN_FOLDER = 'run_{}_{}'
    METADATA_FILE = 'metadata.{}.json'
    LOGGING_FILE = 'logging.{}.log'

    def __init__(self,
                 env,
                 agent,
                 directory=None,
                 run_directory=None,
                 num_episodes=1000,
                 training=True,
                 sim_seed=None,
                 recover=None,
                 display_env=True,
                 display_agent=True,
                 display_rewards=True,
                 close_env=True):
        """

        :param env: The environment to be solved, possibly wrapping an AbstractEnv environment
        :param AbstractAgent agent: The agent solving the environment
        :param Path directory: Workspace directory path
        :param Path run_directory: Run directory path
        :param int num_episodes: Number of episodes run
        !param training: Whether the agent is being trained or tested
        :param sim_seed: The seed used for the environment/agent randomness source
        :param recover: Recover the agent parameters from a file.
                        - If True, it the default latest save will be used.
                        - If a string, it will be used as a path.
        :param display_env: Render the environment, and have a monitor recording its videos
        :param display_agent: Add the agent graphics to the environment viewer, if supported
        :param display_rewards: Display the performances of the agent through the episodes
        :param close_env: Should the environment be closed when the evaluation is closed

        """
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.training = training
        self.sim_seed = sim_seed
        self.close_env = close_env
        self.display_env = display_env

        self.directory = Path(directory or self.default_directory)
        self.run_directory = self.directory / (run_directory or self.default_run_directory)
        self.monitor = MonitorV2(env,
                                 self.run_directory,
                                 video_callable=(None if self.display_env else False))
        self.writer = SummaryWriter(str(self.run_directory))
        self.agent.set_writer(self.writer)
        self.write_logging()
        self.write_metadata()

        self.recover = recover
        if self.recover:
            self.load_agent_model(self.recover)

        if display_agent:
            try:
                # Render the agent within the environment viewer, if supported
                self.env.render()
                self.env.unwrapped.viewer.set_agent_display(
                    lambda agent_surface, sim_surface: AgentGraphics.display(self.agent, agent_surface, sim_surface))
            except AttributeError:
                logger.info("The environment viewer doesn't support agent rendering.")
        self.reward_viewer = None
        if display_rewards:
            self.reward_viewer = RewardViewer()
        self.observation = None

    def train(self):
        self.training = True
        if getattr(self.agent, "batched", False):
            self.run_batched_episodes()
        else:
            self.run_episodes()
        self.close()

    def test(self):
        self.training = False
        if not self.recover:
            logger.warning("No pre-trained model has been loaded.")
        if self.display_env:
            self.monitor.video_callable = MonitorV2.always_call_video
        try:
            self.agent.eval()
        except AttributeError:
            pass
        self.run_episodes()
        self.close()

    def run_episodes(self):
        for episode in range(self.num_episodes):
            # Run episode
            terminal = False
            self.seed(episode)
            self.reset()
            rewards = []
            while not terminal:
                # Step until a terminal step is reached
                reward, terminal = self.step()
                rewards.append(reward)

                # Catch interruptions
                try:
                    if self.env.unwrapped.done:
                        return
                except AttributeError:
                    pass

            # End of episode
            self.after_all_episodes(episode, rewards)
            self.after_some_episodes(episode)

    def step(self):
        """
            Plan a sequence of actions according to the agent policy, and step the environment accordingly.
        """
        # Query agent for actions sequence
        actions = self.agent.plan(self.observation)
        if not actions:
            raise Exception("The agent did not plan any action")

        # Forward the actions to the environment viewer
        try:
            self.env.unwrapped.viewer.set_agent_action_sequence(actions)
        except AttributeError:
            pass

        # Step the environment
        previous_observation, action = self.observation, actions[0]
        self.observation, reward, terminal, info = self.monitor.step(action)

        # Record the experience.
        if self.training:
            try:
                self.agent.record(previous_observation, action, reward, self.observation, terminal, info)
            except NotImplementedError:
                pass

        return reward, terminal

    def run_batched_episodes(self):
        """
            Alternatively,
            - run multiple sample-collection jobs in parallel
            - update model
        """
        episode = 0
        episode_duration = 14  # TODO: use a fixed number of samples instead
        batch_sizes = near_split(self.num_episodes * episode_duration, size_bins=self.agent.config["batch_size"])
        self.agent.reset()
        for batch, batch_size in enumerate(batch_sizes):
            logger.info("[BATCH={}/{}]---------------------------------------".format(batch+1, len(batch_sizes)))
            logger.info("[BATCH={}/{}][run_batched_episodes] #samples={}".format(batch+1, len(batch_sizes),
                                                                                 len(self.agent.memory)))
            logger.info("[BATCH={}/{}]---------------------------------------".format(batch+1, len(batch_sizes)))
            # Save current agent
            model_path = self.save_agent_model(identifier=batch)

            # Prepare workers
            env_config, agent_config = serialize(self.env), serialize(self.agent)
            cpu_processes = self.agent.config["processes"] or os.cpu_count()
            workers_sample_counts = near_split(batch_size, cpu_processes)
            workers_starts = list(np.cumsum(np.insert(workers_sample_counts[:-1], 0, 0)) + np.sum(batch_sizes[:batch]))
            base_seed = self.seed(batch * cpu_processes)[0]
            workers_seeds = [base_seed + i for i in range(cpu_processes)]
            workers_params = list(zip_with_singletons(env_config,
                                                      agent_config,
                                                      workers_sample_counts,
                                                      workers_starts,
                                                      workers_seeds,
                                                      model_path,
                                                      batch))

            # Collect trajectories
            logger.info("Collecting {} samples with {} workers...".format(batch_size, cpu_processes))
            if cpu_processes == 1:
                results = [Evaluation.collect_samples(*workers_params[0])]
            else:
                with Pool(processes=cpu_processes) as pool:
                    results = pool.starmap(Evaluation.collect_samples, workers_params)
            trajectories = [trajectory for worker in results for trajectory in worker]

            # Fill memory
            for trajectory in trajectories:
                if trajectory[-1].terminal:  # Check whether the episode was properly finished before logging
                    self.after_all_episodes(episode, [transition.reward for transition in trajectory])
                episode += 1
                [self.agent.record(*transition) for transition in trajectory]

            # Fit model
            self.agent.update()

    @staticmethod
    def collect_samples(environment_config, agent_config, count, start_time, seed, model_path, batch):
        """
            Collect interaction samples of an agent / environment pair.

            Note that the last episode may not terminate, when enough samples have been collected.

        :param dict environment_config: the environment configuration
        :param dict agent_config: the agent configuration
        :param int count: number of samples to collect
        :param start_time: the initial local time of the agent
        :param seed: the env/agent seed
        :param model_path: the path to load the agent model from
        :param batch: index of the current batch
        :return: a list of trajectories, i.e. lists of Transitions
        """
        env = load_environment(environment_config)
        env.seed(seed)

        if batch == 0:  # Force pure exploration during first batch
            agent_config["exploration"]["final_temperature"] = 1
        agent_config["device"] = "cpu"
        agent = load_agent(agent_config, env)
        agent.load(model_path)
        agent.seed(seed)
        agent.set_time(start_time)

        state = env.reset()
        episodes = []
        trajectory = []
        for _ in range(count):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            trajectory.append(Transition(state, action, reward, next_state, done, info))
            if done:
                state = env.reset()
                episodes.append(trajectory)
                trajectory = []
            else:
                state = next_state
        if trajectory:  # Unfinished episode
            episodes.append(trajectory)
        env.close()
        return episodes

    def save_agent_model(self, identifier, do_save=True):
        # Create the folder if it doesn't exist
        permanent_folder = self.directory / self.SAVED_MODELS_FOLDER
        os.makedirs(permanent_folder, exist_ok=True)

        episode_path = None
        if do_save:
            episode_path = Path(self.monitor.directory) / "checkpoint-{}.tar".format(identifier)
            try:
                self.agent.save(filename=episode_path)
                self.agent.save(filename=permanent_folder / "latest.tar")
            except NotImplementedError:
                pass
            else:
                logger.info("Saved {} model to {}".format(self.agent.__class__.__name__, episode_path))
        return episode_path

    def load_agent_model(self, model_path):
        if model_path is True:
            model_path = self.directory / self.SAVED_MODELS_FOLDER / "latest.tar"
        if isinstance(model_path, str):
            model_path = Path(model_path)
            if not model_path.exists():
                model_path = self.directory / self.SAVED_MODELS_FOLDER / model_path
        try:
            self.agent.load(filename=model_path)
            logger.info("Load {} model from {}".format(self.agent.__class__.__name__, model_path))
        except FileNotFoundError:
            logger.warning("No pre-trained model found at the desired location.")
        except NotImplementedError:
            pass

    def after_all_episodes(self, episode, rewards):
        rewards = np.array(rewards)
        gamma = self.agent.config.get("gamma", 1)
        self.writer.add_scalar('episode/length', len(rewards), episode)
        self.writer.add_scalar('episode/total_reward', sum(rewards), episode)
        self.writer.add_scalar('episode/return', sum(r*gamma**t for t, r in enumerate(rewards)), episode)
        self.writer.add_histogram('episode/rewards', rewards, episode)
        logger.info("Episode {} score: {:.1f}".format(episode, sum(rewards)))

    def after_some_episodes(self, episode):
        if self.monitor.is_episode_selected():
            # Save the model
            if self.training:
                self.save_agent_model(episode)

    @property
    def default_directory(self):
        return Path(self.OUTPUT_FOLDER) / self.env.unwrapped.__class__.__name__ / self.agent.__class__.__name__

    @property
    def default_run_directory(self):
        return self.RUN_FOLDER.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), os.getpid())

    def write_metadata(self):
        metadata = dict(env=serialize(self.env), agent=serialize(self.agent))
        file_infix = '{}.{}'.format(self.monitor.monitor_id, os.getpid())
        file = self.run_directory / self.METADATA_FILE.format(file_infix)
        with file.open('w') as f:
            json.dump(metadata, f, sort_keys=True, indent=4)

    def write_logging(self):
        file_infix = '{}.{}'.format(self.monitor.monitor_id, os.getpid())
        rl_agents.trainer.logger.configure()
        rl_agents.trainer.logger.add_file_handler(self.run_directory / self.LOGGING_FILE.format(file_infix))

    def seed(self, episode=0):
        seed = self.sim_seed + episode if self.sim_seed is not None else None
        seed = self.monitor.seed(seed)
        self.agent.seed(seed[0])  # Seed the agent with the main environment seed
        return seed

    def reset(self):
        self.observation = self.monitor.reset()
        self.agent.reset()

    def close(self):
        """
            Close the evaluation.
        """
        if self.training:
            self.save_agent_model("final")
        self.monitor.close()
        self.writer.close()
        if self.close_env:
            self.env.close()

import json
import os

from gym import logger

from rl_agents.configuration import serialize
from rl_agents.trainer.graphics import RewardViewer
from rl_agents.agents.graphics import AgentGraphics
from rl_agents.trainer.monitor import MonitorV2


class Evaluation(object):
    """
        The evaluation of an agent interacting with an environment to maximize its expected reward.
    """

    OUTPUT_FOLDER = 'out'
    SAVED_MODELS_FOLDER = 'saved_models'
    METADATA_FILE = 'metadata.{}.json'

    def __init__(self,
                 env,
                 agent,
                 directory=None,
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
        :param str directory: Output directory path
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

        self.directory = directory or self.default_directory
        self.monitor = MonitorV2(env,
                                 self.directory,
                                 add_subdirectory=(directory is None),
                                 video_callable=(None if display_env else False))
        self.write_metadata()

        if recover:
            self.load_agent_model(recover)

        self.agent_viewer = None
        if display_agent:
            try:
                # Render the agent within the environment viewer, if supported
                self.env.render()
                self.env.unwrapped.viewer.set_agent_display(
                    lambda agent_surface, sim_surface: AgentGraphics.display(self.agent, agent_surface, sim_surface))
            except AttributeError:
                # The environment viewer doesn't support agent rendering, create a separate agent viewer
                # self.agent_viewer = AgentViewer(self.agent)
                pass
        self.reward_viewer = None
        if display_rewards:
            self.reward_viewer = RewardViewer()
        self.observation = None

    def train(self):
        self.training = True
        self.run_episodes()
        self.close()

    def test(self, model_path=True):
        self.training = False
        self.load_agent_model(model_path)
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
            self.seed()
            self.reset()
            total_reward = 0
            while not terminal:
                # Step until a terminal step is reached
                reward, terminal = self.step()
                total_reward += reward

                # Catch interruptions
                try:
                    if self.env.unwrapped.done:
                        return
                except AttributeError:
                    pass

            # End of episode
            self.after_all_episodes(episode, total_reward)
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
            self.env.unwrapped.viewer.predict_trajectory(actions)
        except AttributeError:
            pass

        if self.agent_viewer and self.monitor.is_episode_selected():
            self.agent_viewer.render()

        # Step the environment
        previous_observation, action = self.observation, actions[0]
        self.observation, reward, terminal, info = self.monitor.step(action)

        # Record the experience.
        if self.training:
            try:
                self.agent.record(previous_observation, action, reward, self.observation, terminal)
            except NotImplementedError:
                pass

        return reward, terminal

    def save_agent_model(self, episode, do_save=True):
        # Create the folder if it doesn't exist
        permanent_folder = os.path.join(self.directory, self.SAVED_MODELS_FOLDER)
        os.makedirs(permanent_folder, exist_ok=True)

        if do_save:
            episode_path = os.path.join(self.monitor.directory, "checkpoint-{}.tar".format(episode+1))
            try:
                self.agent.save(filename=episode_path)
                self.agent.save(filename=os.path.join(permanent_folder, "latest.tar"))
            except NotImplementedError:
                pass
            else:
                logger.info("Saved {} model to {}".format(self.agent.__class__.__name__, episode_path))

    def load_agent_model(self, model_path):
        if model_path is True:
            model_path = os.path.join(self.directory, self.SAVED_MODELS_FOLDER, "latest.tar")
        try:
            self.agent.load(filename=model_path)
            logger.info("Load {} model from {}".format(self.agent.__class__.__name__, model_path))
        except FileNotFoundError:
            logger.warn("No pre-trained model found at the desired location.")
        except NotImplementedError:
            pass

    def after_all_episodes(self, episode, total_reward):
        if self.reward_viewer:
            self.reward_viewer.update(total_reward)
        logger.info("Episode {} score: {}".format(episode, total_reward))

    def after_some_episodes(self, episode):
        if self.monitor.is_episode_selected():
            # Save the model
            if self.training:
                self.save_agent_model(episode)

    @property
    def default_directory(self):
        return os.path.join(self.OUTPUT_FOLDER, self.env.unwrapped.__class__.__name__, self.agent.__class__.__name__)

    def write_metadata(self):
        metadata = dict(env=serialize(self.env), agent=serialize(self.agent))
        file_infix = '{}.{}'.format(self.monitor.monitor_id, os.getpid())
        file = os.path.join(self.monitor.directory, self.METADATA_FILE.format(file_infix))
        with open(file, 'w') as f:
            json.dump(metadata, f, sort_keys=True, indent=4)

    def seed(self):
        seed = self.env.seed(self.sim_seed)
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
            self.save_agent_model(self.monitor.episode_id)
        self.monitor.close()
        if self.close_env:
            self.env.close()

import os, six
from gym import logger

from rl_agents.trainer.graphics import RewardViewer, SimulationViewer
from rl_agents.trainer.monitor import MonitorV2


class Simulation:
    """
        A simulation is the coupling of an environment and an agent, running in closed loop.
    """

    OUTPUT_FOLDER = 'out'
    SAVED_MODELS_FOLDER = 'saved_models'

    def __init__(self,
                 env,
                 agent,
                 directory=None,
                 num_episodes=1000,
                 training=True,
                 sim_seed=None,
                 recover=None,
                 agent_viewer=None):
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
        :param agent_viewer: The viewer used to render the agent internal reasoning

        """
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.training = training
        self.sim_seed = sim_seed
        self.agent_viewer = agent_viewer

        self.directory = directory or os.path.join(self.OUTPUT_FOLDER, env.unwrapped.__class__.__name__)
        self.monitor = MonitorV2(env, self.directory, add_subdirectory=(directory is None))

        if recover:
            self.load_model(recover)

        self.reward_viewer = RewardViewer()
        if agent_viewer:
            # If agent rendering is requested, create or replace the environment viewer by a simulation viewer
            # self.env.unwrapped.viewer = SimulationViewer(self)
            pass

        self.observation = None

    def run(self):
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
            self.self.env.unwrapped.viewer.predict_trajectory(actions)
        except AttributeError:
            pass

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

    def after_all_episodes(self, episode, total_reward):
        self.reward_viewer.update(total_reward)
        logger.info("Episode {} score: {}".format(episode, total_reward))

    def after_some_episodes(self, episode):
        if self.monitor.is_episode_selected():
            if self.agent_viewer:
                self.agent_viewer.display()

            # Save the model
            if self.training:
                self.save_model(episode)

    def save_model(self, episode, do_save=True):
        # Create the folder if it doesn't exist
        folder = os.path.join(self.directory, self.SAVED_MODELS_FOLDER)
        os.makedirs(folder, exist_ok=True)

        if do_save:
            episode_path = os.path.join(folder, "checkpoint-{}.tar".format(episode))
            try:
                self.agent.save(filename=episode_path)
                self.agent.save(filename=os.path.join(folder, "latest.tar"))
            except NotImplementedError:
                pass
            else:
                logger.info("Saved {} model to {}".format(self.agent.__class__.__name__, episode_path))

    def load_model(self, model_path):
        if model_path is True:
            model_path = os.path.join(self.directory, self.SAVED_MODELS_FOLDER, "latest.tar")
        try:
            self.agent.load(filename=model_path)
            logger.info("Load {} model from {}".format(self.agent.__class__.__name__, model_path))
        except FileNotFoundError:
            logger.warn("No pre-trained model found at the desired location.")
        except NotImplementedError:
            pass

    def train(self):
        self.training = True
        self.run()

    def test(self, model_path=True):
        self.training = False
        self.load_model(model_path)
        self.monitor.video_callable = MonitorV2.always_call_video
        if hasattr(self.agent, 'config'):
            self.agent.config['epsilon'] = [0, 0]
        self.run()

    def seed(self):
        seed = self.env.seed(self.sim_seed)
        self.agent.seed(seed[0])  # Seed the agent with the main environment seed
        return seed

    def reset(self):
        self.observation = self.monitor.reset()
        self.agent.reset()

    def render(self, mode='human'):
        """
            Render the environment.
        :param mode: the rendering mode
        """
        self.monitor.render(mode)

    def close(self):
        """
            Close the simulation.
        """
        if self.training:
            self.save_model(self.monitor.episode_id)
        self.monitor.close()


import pickle
from gym import logger

from rl_agents.agents.dqn.pytorch import DQNAgent
from rl_agents.agents.fitted_q.abstract import AbstractFTQAgent


class FTQAgent(AbstractFTQAgent, DQNAgent):
    def __init__(self, env, config=None):
        super(FTQAgent, self).__init__(env, config)

    def initialize_model(self):
        self.policy_net.reset()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename):
        super(FTQAgent, self).save(filename)
        dataset_filename = filename + ".data"
        with open(dataset_filename, 'wb') as f:
            pickle.dump(self.memory.memory, f)
        logger.info("Saved a replay memory of length {}".format(len(self.memory)))

    def load(self, filename):
        super(FTQAgent, self).load(filename)
        dataset_filename = filename + ".data"
        with open(dataset_filename, 'rb') as f:
            self.memory.memory = pickle.load(f)
        logger.info("Loaded a replay memory of length {}".format(len(self.memory)))

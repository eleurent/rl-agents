import numpy as np
from rl_agents.agents.abstract import AbstractAgent


class LinearAgent(AbstractAgent):
    def __init__(self, env, config):
        self.env = env
        self.config = config

    def act(self, observation):
        u = np.dot(self.config['K'], -observation)
        action = 1 if u < 0 else 0
        return action

    def reset(self):
        pass

    def seed(self, seed=None):
        pass

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()

    def record(self, state, action, reward, next_state, done):
        pass



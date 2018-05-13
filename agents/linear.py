import numpy as np
from agents.abstract import AbstractAgent


class LinearAgent(AbstractAgent):
    def __init__(self, env, config):
        self.env = env
        self.config = config

    def record(self, state, action, reward, next_state, done):
        pass

    def act(self, observation):
        u = np.dot(self.config['K'], -observation)
        action = 1 if u < 0 else 0
        return action

import numpy as np
from rl_agents.agents.common.abstract import AbstractAgent


class LinearCommandAgent(AbstractAgent):
    def __init__(self, env, config=None):
        super(LinearCommandAgent, self).__init__(config)
        self.K = np.array(self.config["K"])
        self.env = env

    @classmethod
    def default_config(cls):
        return {"K": 0}

    def act(self, observation):
        u = np.dot(self.K, -observation)
        action = 1 if u < 0 else 0
        return action

    def reset(self):
        pass

    def seed(self, seed=None):
        pass

    def save(self, filename):
        return False

    def load(self, filename):
        return False

    def record(self, state, action, reward, next_state, done, info):
        pass



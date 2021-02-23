import numpy as np
from rl_agents.agents.common.abstract import AbstractAgent


class LinearFeedbackAgent(AbstractAgent):
    def __init__(self, env, config=None):
        super().__init__(config)
        self.K = np.array(self.config["K"])
        self.env = env

    @classmethod
    def default_config(cls):
        return {
            "K": [[0]],
            "discrete": False
        }

    def act(self, observation):
        if isinstance(observation, dict):
            state = observation["state"]
            reference = observation["reference_state"]
        else:
            state = observation
            reference = np.zeros(observation.shape)
        control = self.K @ (reference - state)
        if self.config["discrete"]:
            control = 1 if control < 0 else 0
        else:
            control = control.squeeze(-1)
        return control

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



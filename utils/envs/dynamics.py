import numpy as np
from gym import Env, spaces
from gym.envs.registration import register


class DynamicsEnv(Env):
    def __init__(self, dt=0.1):
        self.x = np.zeros((2, 1))
        self.A = np.array([[1, dt], [0, 1]])
        self.B = np.array([[0], [dt]])
        self.action_space = spaces.Discrete(2)

    def step(self, action: int):
        u = np.array([[2*action - 1]])
        self.x = self.A @ self.x + self.B @ u
        return self.x, self.reward(), False, {}

    def reward(self):
        return max(1 - self.x[0, 0]**2, 0)

    def reset(self):
        self.x = np.array([[-1], [0]])
        return self

    def seed(self, seed=None):
        # TODO: include action noise?
        pass

    def render(self, mode='human'):
        pass


register(
    id='dynamics-v0',
    entry_point='utils.envs:DynamicsEnv'
)

import numpy as np
from gym import Env, spaces
from gym.envs.registration import register


class GridEnv(Env):
    REWARD_CENTER = [10, 10]
    REWARD_RADIUS = 5

    def __init__(self, use_diagonals=False):
        self.x = np.zeros(2)
        num_actions = 8 if use_diagonals else 4
        self.action_space = spaces.Discrete(num_actions)

    def step(self, action):
        if action == 0:
            self.x[0] += 1
        elif action == 1:
            self.x[0] -= 1
        elif action == 2:
            self.x[1] += 1
        elif action == 3:
            self.x[1] -= 1
        elif action == 4:
            self.x[0] += 1
            self.x[1] += 1
        elif action == 5:
            self.x[0] += 1
            self.x[1] -= 1
        elif action == 6:
            self.x[0] -= 1
            self.x[1] += 1
        elif action == 7:
            self.x[0] -= 1
            self.x[1] -= 1
        return self.x, self.reward(), False, {}

    def reward(self):
        return np.clip(1 - 1/self.REWARD_RADIUS**2 * ((self.REWARD_CENTER[0] - self.x[0])**2
                                                      + (self.REWARD_CENTER[1] - self.x[1])**2),
                       0, 1)

    def reset(self):
        self.x = np.array([0, 0])
        return self.x

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass


register(
    id='gridenv-v0',
    entry_point='utils.envs:GridEnv'
)

import numpy as np
from gym import Env, spaces
from gym.envs.registration import register
from gym.utils import seeding


class GridEnv(Env):
    REWARD_CENTER = [10, 10]
    REWARD_RADIUS = 5

    config = {
        "use_diagonals": False,
        "stochasticity": 0
    }

    def __init__(self, config=None):
        self.x = np.zeros(2)
        num_actions = 8 if self.config["use_diagonals"] else 4
        self.action_space = spaces.Discrete(num_actions)
        self.np_random = None
        self.seed()

    def configure(self, config):
        self.config.update(config)

    def step(self, action):
        if self.config["stochasticity"] > 0:
            if self.np_random.uniform() < self.config["stochasticity"]:
                action = -1
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
        return self.x.copy(), self.reward(), False, {}

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
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class LineEnv(Env):
    def __init__(self):
        self.x = 0
        self.action_space = spaces.Discrete(2)
        self.np_random = None
        self.done = False
        self.seed()

    def step(self, action):
        delta = 0
        if action == 0:
            delta -= 1
        elif action == 1:
            delta += 1
        # Noise
        delta += 2*self.np_random.randint(2) - 1
        self.x += delta // 2
        self.done = self.done or self.terminal()
        return self.x, self.reward(), self.done, {}

    def reward(self):
        return 1.0 * (abs(self.x) <= 1) if not self.done else 0

    def terminal(self):
        return abs(self.x) >= 2

    def reset(self):
        self.x = 0
        self.done = False
        return self.x

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


register(
    id='gridenv-v0',
    entry_point='utils.envs:GridEnv'
)

register(
    id='line_env-v0',
    entry_point='utils.envs:LineEnv',
    max_episode_steps=10
)

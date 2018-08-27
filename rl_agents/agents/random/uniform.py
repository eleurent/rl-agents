import gym
from gym.utils import seeding

from rl_agents.agents.abstract import AbstractAgent


class RandomUniformAgent(AbstractAgent):
    def __init__(self, env, config=None):
        super(RandomUniformAgent, self).__init__(config)
        self.np_random = None
        self.seed()
        self.env = env

    def act(self, state):
        return self.env.action_space.sample()

    def record(self, state, action, reward, next_state, done):
        pass

    def reset(self):
        pass

    def seed(self, seed=None):
        """
            Seed the rollout policy randomness source
        :param seed: the seed to be used
        :return: the used seed
        """
        self.np_random, seed = seeding.np_random(seed)
        gym.spaces.Discrete.np_random = self.np_random
        return [seed]

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()
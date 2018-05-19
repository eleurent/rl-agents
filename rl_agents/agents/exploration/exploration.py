import numpy as np
from gym.utils import seeding
from gym import spaces


class ExplorationPolicy(object):
    def __init__(self, config):
        self.config = config
        self.steps_done = 0
        self.np_random = None
        self.seed()

    def epsilon_greedy(self, optimal_action, action_space):
        sample = self.np_random.rand()
        epsilon = self.config['epsilon'][1] + (self.config['epsilon'][0] - self.config['epsilon'][1]) * \
            np.exp(-2. * self.steps_done / self.config['epsilon_tau'])
        self.steps_done += 1
        if sample > epsilon:
            return optimal_action
        else:
            # Replace the number generator that will be used in action_space.sample() by this policy's
            # TODO: possible race condition when several agents are running
            spaces.np_random = self.np_random
            return action_space.sample()

    def seed(self, seed=None):
        """
            Seed the policy randomness source
        :param seed: the seed to be used
        :return: the used seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

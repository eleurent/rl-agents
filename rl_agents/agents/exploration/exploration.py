import numpy as np
from gym.utils import seeding
from gym import spaces


class DiscreteDistribution(object):
    def __init__(self):
        self.np_random = None

    def get_distribution(self):
        """
        :return: a distribution over actions {action:probability}
        """
        raise NotImplementedError()

    def sample(self):
        """
        :return: an action sampled from the distribution
        """
        distribution = self.get_distribution()
        return self.np_random.choice(list(distribution.keys()), 1, p=list(distribution.values()))[0]

    def seed(self, seed=None):
        """
            Seed the policy randomness source
        :param seed: the seed to be used
        :return: the used seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class EpsilonGreedy(DiscreteDistribution):
    def __init__(self, config, action_space):
        super(EpsilonGreedy, self).__init__()
        self.config = config
        self.action_space = action_space
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("The action space should be discrete")
        self.optimal_action = None
        self.epsilon = 0
        self.steps_done = 0
        self.seed()

    def get_distribution(self):
        """
            Uniform distribution with probability epsilon, and optimal action with probability 1-epsilon
        """
        distribution = {action: self.epsilon / self.action_space.n for action in range(self.action_space.n)}
        distribution[self.optimal_action] += 1 - self.epsilon
        return distribution

    def update(self, optimal_action):
        """
            Update the action distribution parameters
        :param optimal_action: the optimal action
        """
        self.optimal_action = optimal_action
        self.epsilon = self.config['epsilon'][1] + (self.config['epsilon'][0] - self.config['epsilon'][1]) * \
             np.exp(-2. * self.steps_done / self.config['epsilon_tau'])
        self.steps_done += 1

    def act(self, optimal_action):
        """
            Update the actions distribution and sample an action
        :param optimal_action: the optimal action
        :return: the sampled action
        """
        self.update(optimal_action)
        return self.sample()

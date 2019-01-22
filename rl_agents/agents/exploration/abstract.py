import numpy as np
from gym.utils import seeding
from abc import abstractmethod, ABC

from rl_agents.configuration import Configurable


class DiscreteDistribution(Configurable, ABC):
    def __init__(self, config=None, **kwargs):
        super(DiscreteDistribution, self).__init__(config)
        self.np_random = None

    @abstractmethod
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
        return self.np_random.choice(list(distribution.keys()), 1, p=np.array(list(distribution.values())))[0]

    def seed(self, seed=None):
        """
            Seed the policy randomness source
        :param seed: the seed to be used
        :return: the used seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
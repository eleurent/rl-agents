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

    def set_time(self, time):
        """ Set the local time, allowing to schedule the distribution temperature. """
        pass

    def step_time(self):
        """ Step the local time, allowing to schedule the distribution temperature. """
        pass


def exploration_factory(exploration_config, action_space):
    """
        Handles creation of exploration policies
    :param exploration_config: configuration dictionary of the policy, must contain a "method" key
    :param action_space: the environment action space
    :return: a new exploration policy
    """
    from rl_agents.agents.common.exploration.boltzmann import Boltzmann
    from rl_agents.agents.common.exploration.epsilon_greedy import EpsilonGreedy
    from rl_agents.agents.common.exploration.greedy import Greedy

    if exploration_config['method'] == 'Greedy':
        return Greedy(action_space, exploration_config)
    elif exploration_config['method'] == 'EpsilonGreedy':
        return EpsilonGreedy(action_space, exploration_config)
    elif exploration_config['method'] == 'Boltzmann':
        return Boltzmann(action_space, exploration_config)
    else:
        raise ValueError("Unknown exploration method")

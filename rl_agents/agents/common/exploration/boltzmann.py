import numpy as np
from gym import spaces

from rl_agents.agents.common.exploration.abstract import DiscreteDistribution


class Boltzmann(DiscreteDistribution):
    """
        Uniform distribution with probability epsilon, and optimal action with probability 1-epsilon
    """

    def __init__(self, action_space, config=None):
        super(Boltzmann, self).__init__(config)
        self.action_space = action_space
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("The action space should be discrete")
        self.values = None
        self.seed()

    @classmethod
    def default_config(cls):
        return dict(temperature=0.5)

    def get_distribution(self):
        actions = range(self.action_space.n)
        if self.config['temperature'] > 0:
            weights = np.exp(self.values / self.config['temperature'])
        else:
            weights = np.zeros((len(actions),))
            weights[np.argmax(self.values)] = 1
        return {action: weights[action] / np.sum(weights) for action in actions}

    def update(self, values):
        self.values = values

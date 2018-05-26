import numpy as np
from gym import spaces

from rl_agents.agents.exploration.abstract import DiscreteDistribution


class EpsilonGreedy(DiscreteDistribution):
    """
        Uniform distribution with probability epsilon, and optimal action with probability 1-epsilon
    """

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
        distribution = {action: self.epsilon / self.action_space.n for action in range(self.action_space.n)}
        distribution[self.optimal_action] += 1 - self.epsilon
        return distribution

    def update(self, values):
        """
            Update the action distribution parameters
        :param values: the state-action values
        """
        self.optimal_action = np.argmax(values)
        self.epsilon = self.config['epsilon'][1] + (self.config['epsilon'][0] - self.config['epsilon'][1]) * \
                       np.exp(-2. * self.steps_done / self.config['epsilon_tau'])
        self.steps_done += 1

import numpy as np
from gym import spaces

from rl_agents.agents.common.exploration.abstract import DiscreteDistribution


class Greedy(DiscreteDistribution):
    """
        Always use the optimal action
    """

    def __init__(self, action_space, config=None):
        super(Greedy, self).__init__(config)
        self.action_space = action_space
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("The action space should be discrete")
        self.values = None
        self.seed()

    def get_distribution(self):
        optimal_action = np.argmax(self.values)
        return {action: 1 if action == optimal_action else 0 for action in range(self.action_space.n)}

    def update(self, values, step_time=False):
        self.values = values

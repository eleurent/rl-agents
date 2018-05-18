import numpy as np


class ExplorationPolicy(object):
    def __init__(self, config):
        self.config = config
        self.steps_done = 0

    def epsilon_greedy(self, optimal_action, action_space):
        sample = np.random.random()
        epsilon = self.config['epsilon'][1] + (self.config['epsilon'][0] - self.config['epsilon'][1]) * \
            np.exp(-2. * self.steps_done / self.config['epsilon_tau'])
        self.steps_done += 1
        if sample > epsilon:
            return optimal_action
        else:
            return action_space.sample()
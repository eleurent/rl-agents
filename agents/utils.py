import random
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'terminal'))


class ReplayMemory(object):
    def __init__(self, config):
        self.capacity = config['memory_capacity']
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # Faster than append and pop
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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


class ValueFunctionViewer(object):
    def __init__(self, agent):
        self.agent = agent
        self.values_history = np.array([])

    def display(self):
        self.plot_values()
        self.plot_value_map()

    def plot_value_map(self):
        xx, yy, states = self.states_mesh()
        values, actions = self.agent.states_to_values(states)
        values, actions = np.reshape(values, np.shape(xx)), np.reshape(actions, np.shape(xx))

        f, (ax1, ax2) = plt.subplots(1, 2, num='Value')
        f.clf()
        ax1.imshow(values)
        # f.colorbar(f, ax=ax1)
        ax2.imshow(actions)
        # f.colorbar(f, ax=ax2)
        plt.pause(0.001)

    def plot_values(self):
        states = self.states_list()
        values, _ = self.agent.states_to_values(states)
        self.values_history = np.hstack((self.values_history, values)) if self.values_history.size else values

        plt.figure(num='Value history')
        plt.clf()
        plt.title('Predicted value')
        plt.xlabel('Episode')
        plt.ylabel('Value')
        plt.plot(self.values_history)
        plt.pause(0.001)

    predicted_values = np.array([])

    def states_mesh(self):
        # TODO env-specific instead of cartpole
        xx, yy = np.meshgrid(np.linspace(-1, 1, 15), np.linspace(-1, 1, 15))
        xf = np.reshape(xx, (np.size(xx), 1))
        yf = np.reshape(yy, (np.size(yy), 1))
        states = np.hstack((2 * xf, 2 * xf, yf * 12 * np.pi / 180, yf))
        return xx, yy, states

    def states_list(self):
        return np.array([[0, 0, 0, 0],
                         [-0.08936051, -0.37169457, 0.20398587, 1.03234316],
                         [0.10718797, 0.97770614, -0.20473761, -1.6631015]])

import numpy as np
from matplotlib import pyplot as plt, gridspec as gridspec


class ValueFunctionViewer(object):
    def __init__(self, agent, state_sampler):
        self.agent = agent
        self.state_sampler = state_sampler
        self.values_history = np.array([])
        self.figure = None
        self.axes = []

    def display(self):
        if not self.state_sampler:
            return
        if not self.figure:
            plt.ion()
            self.figure = plt.figure('Value function')
            gs = gridspec.GridSpec(2, 2)
            self.axes.append(plt.subplot(gs[0, :]))
            self.axes.append(plt.subplot(gs[1, 0]))
            self.axes.append(plt.subplot(gs[1, 1]))

            xx, _, _ = self.state_sampler.states_mesh()
            cax1 = self.axes[1].imshow(xx)
            cax2 = self.axes[2].imshow(xx)
            self.figure.colorbar(cax1, ax=self.axes[1])
            self.figure.colorbar(cax2, ax=self.axes[2])

        self.plot_values()
        self.plot_value_map()

    def plot_value_map(self):
        xx, yy, states = self.state_sampler.states_mesh()
        values, actions = self.agent.states_to_values(states)
        values, actions = np.reshape(values, np.shape(xx)), np.reshape(actions, np.shape(xx))

        self.axes[1].clear()
        self.axes[2].clear()
        self.axes[1].imshow(values)
        self.axes[2].imshow(actions)
        plt.pause(0.001)
        plt.draw()

    def plot_values(self):
        states = self.state_sampler.states_list()
        values, _ = self.agent.states_to_values(states)
        self.values_history = np.vstack((self.values_history, values)) if self.values_history.size else values

        self.axes[0].clear()
        self.axes[0].set_xlabel('Episode')
        self.axes[0].set_ylabel('Value')
        self.axes[0].plot(self.values_history)
        plt.pause(0.001)
        plt.draw()
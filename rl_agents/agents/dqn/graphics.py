import numpy as np
from matplotlib import pyplot as plt, gridspec as gridspec

import pygame
import matplotlib as mpl
import matplotlib.cm as cm
from highway_env.envs.abstract import AbstractEnv


class DQNGraphics(object):
    """
        Graphical visualization of the DQNAgent state-action values.
    """
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)
    MIN_VALUE = -10
    MAX_VALUE = 10

    @classmethod
    def display(cls, agent, surface, display_text=True):
        """
            Display the action-values for the current state

        :param agent: the DQNAgent to be displayed
        :param surface: the pygame surface on which the agent is displayed
        :param display_text: whether to display the action values as text
        """
        action_values = agent.get_state_action_values(agent.previous_state)
        action_distribution = agent.action_distribution(agent.previous_state)

        cell_size = (surface.get_width() // len(action_values), surface.get_height())
        pygame.draw.rect(surface, cls.BLACK, (0, 0, surface.get_width(), surface.get_height()), 0)

        # Display node value
        for action, value in enumerate(action_values):
            cmap = cm.jet_r
            norm = mpl.colors.Normalize(vmin=cls.MIN_VALUE, vmax=cls.MAX_VALUE)
            color = cmap(norm(value), bytes=True)
            pygame.draw.rect(surface, color, (cell_size[0]*action, 0, cell_size[0], cell_size[1]), 0)

            if display_text:
                font = pygame.font.Font(None, 15)
                text = "v={:.2f} / p={:.2f}".format(value, action_distribution[action])
                text = font.render(text,
                                   1, (10, 10, 10), (255, 255, 255))
                surface.blit(text, (cell_size[0]*action, 0))


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
        values, actions = self.agent.get_batch_state_values(states)
        values, actions = np.reshape(values, np.shape(xx)), np.reshape(actions, np.shape(xx))

        self.axes[1].clear()
        self.axes[2].clear()
        self.axes[1].imshow(values)
        self.axes[2].imshow(actions)
        plt.pause(0.001)
        plt.draw()

    def plot_values(self):
        states = self.state_sampler.states_list()
        values, _ = self.agent.get_batch_state_values(states)
        self.values_history = np.vstack((self.values_history, values)) if self.values_history.size else values

        self.axes[0].clear()
        self.axes[0].set_xlabel('Episode')
        self.axes[0].set_ylabel('Value')
        self.axes[0].plot(self.values_history)
        plt.pause(0.001)
        plt.draw()
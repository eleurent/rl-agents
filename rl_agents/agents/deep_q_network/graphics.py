import numpy as np
from matplotlib import pyplot as plt, gridspec as gridspec
import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm

from rl_agents.utils import remap, constrain


class DQNGraphics(object):
    """
        Graphical visualization of the DQNAgent state-action values.
    """
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)
    MIN_ATTENTION = 0.01

    @classmethod
    def display(cls, agent, surface, sim_surface=None, display_text=True):
        """
            Display the action-values for the current state

        :param agent: the DQNAgent to be displayed
        :param surface: the pygame surface on which the agent is displayed
        :param sim_surface: the pygame surface on which the env is rendered
        :param display_text: whether to display the action values as text
        """
        import pygame
        action_values = agent.get_state_action_values(agent.previous_state)
        action_distribution = agent.action_distribution(agent.previous_state)

        cell_size = (surface.get_width() // len(action_values), surface.get_height())
        pygame.draw.rect(surface, cls.BLACK, (0, 0, surface.get_width(), surface.get_height()), 0)

        # Display node value
        for action, value in enumerate(action_values):
            cmap = cm.jet_r
            norm = mpl.colors.Normalize(vmin=0, vmax=1/(1-agent.config["gamma"]))
            color = cmap(norm(value), bytes=True)
            pygame.draw.rect(surface, color, (cell_size[0]*action, 0, cell_size[0], cell_size[1]), 0)

            if display_text:
                font = pygame.font.Font(None, 15)
                text = "v={:.2f} / p={:.2f}".format(value, action_distribution[action])
                text = font.render(text,
                                   1, (10, 10, 10), (255, 255, 255))
                surface.blit(text, (cell_size[0]*action, 0))

        if sim_surface and hasattr(agent.value_net, "get_attention_matrix"):
            cls.display_vehicles_attention(agent, sim_surface)

    @classmethod
    def display_vehicles_attention(cls, agent, sim_surface):
        import pygame
        try:
            state = agent.previous_state
            if (not hasattr(cls, "state")) or (cls.state != state).any():
                cls.v_attention = cls.compute_vehicles_attention(agent, state)
                cls.state = state

            for head in range(list(cls.v_attention.values())[0].shape[0]):
                attention_surface = pygame.Surface(sim_surface.get_size(), pygame.SRCALPHA)
                for vehicle, attention in cls.v_attention.items():
                    if attention[head] < cls.MIN_ATTENTION:
                        continue
                    width = attention[head] * 5
                    desat = remap(attention[head], (0, 0.5), (0.7, 1), clip=True)
                    colors = sns.color_palette("dark", desat=desat)
                    color = np.array(colors[(2*head) % (len(colors) - 1)]) * 255
                    color = (*color, remap(attention[head], (0, 0.5), (100, 200), clip=True))
                    if vehicle is agent.env.vehicle:
                        pygame.draw.circle(attention_surface, color,
                                           sim_surface.vec2pix(agent.env.vehicle.position),
                                           max(sim_surface.pix(width / 2), 1))
                    else:
                        pygame.draw.line(attention_surface, color,
                                         sim_surface.vec2pix(agent.env.vehicle.position),
                                         sim_surface.vec2pix(vehicle.position),
                                         max(sim_surface.pix(width), 1))
                sim_surface.blit(attention_surface, (0, 0))
        except ValueError as e:
            print("Unable to display vehicles attention", e)

    @classmethod
    def compute_vehicles_attention(cls, agent, state):
        import torch
        state_t = torch.tensor([state], dtype=torch.float).to(agent.device)
        attention = agent.value_net.get_attention_matrix(state_t).squeeze(0).squeeze(1).detach().cpu().numpy()
        ego, others, mask = agent.value_net.split_input(state_t)
        mask = mask.squeeze()
        v_attention = {}
        obs_type = agent.env.observation_type
        if hasattr(obs_type, "agents_observation_types"):  # Handle multi-agent observation
            obs_type = obs_type.agents_observation_types[0]
        for v_index in range(state.shape[0]):
            if mask[v_index]:
                continue
            v_position = {}
            for feature in ["x", "y"]:
                v_feature = state[v_index, obs_type.features.index(feature)]
                v_feature = remap(v_feature, [-1, 1], obs_type.features_range[feature])
                v_position[feature] = v_feature
            v_position = np.array([v_position["x"], v_position["y"]])
            if not obs_type.absolute and v_index > 0:
                v_position += agent.env.unwrapped.vehicle.position
            vehicle = min(agent.env.road.vehicles, key=lambda v: np.linalg.norm(v.position - v_position))
            v_attention[vehicle] = attention[:, v_index]
        return v_attention


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

import importlib

import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np


class ValueIterationGraphics(object):
    """
        Graphical visualization of the Value Iteration value function.
    """
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    highway_module = None

    @classmethod
    def display(cls, agent, surface):
        """
            Display the computed value function of an agent.

        :param agent: the agent to be displayed
        :param surface: the surface on which the agent is displayed.
        """
        if not cls.highway_module:
            try:
                cls.highway_module = importlib.import_module("highway_env")
            except ModuleNotFoundError:
                pass
        if cls.highway_module and isinstance(agent.env, cls.highway_module.envs.common.abstract.AbstractEnv):
            cls.display_highway(agent, surface)

    @classmethod
    def display_highway(cls, agent, surface):
        """
            Particular visualization of the state space that is used for highway_env environments only.

        :param agent: the agent to be displayed
        :param surface: the surface on which the agent is displayed.
        """
        import pygame
        norm = mpl.colors.Normalize(vmin=-2, vmax=2)
        cmap = cm.jet_r
        try:
            grid_shape = agent.mdp.original_shape
        except AttributeError:
            grid_shape = cls.highway_module.finite_mdp.compute_ttc_grid(agent.env, time_quantization=1., horizon=10.).shape
        cell_size = (surface.get_width() // grid_shape[2], surface.get_height() // (grid_shape[0] * grid_shape[1]))
        speed_size = surface.get_height() // grid_shape[0]
        value = agent.get_state_value().reshape(grid_shape)
        for h in range(grid_shape[0]):
            for i in range(grid_shape[1]):
                for j in range(grid_shape[2]):
                    color = cmap(norm(value[h, i, j]), bytes=True)
                    pygame.draw.rect(surface, color, (
                        j * cell_size[0], i * cell_size[1] + h * speed_size, cell_size[0], cell_size[1]), 0)
            pygame.draw.line(surface, cls.BLACK,
                             (0, h * speed_size), (grid_shape[2] * cell_size[0], h * speed_size), 1)
        states, actions = agent.plan_trajectory(agent.mdp.state)
        for state in states:
            (h, i, j) = np.unravel_index(state, grid_shape)
            pygame.draw.rect(surface, cls.RED,
                             (j * cell_size[0], i * cell_size[1] + h * speed_size, cell_size[0], cell_size[1]), 1)

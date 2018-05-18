import pygame
import matplotlib as mpl
import matplotlib.cm as cm


class TTCVIGraphics(object):
    """
        Graphical visualization of the TTCVIAgent value function.
    """
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)

    @classmethod
    def display(cls, agent, surface):
        """
            Display the computed value function of an agent.

        :param agent: the agent to be displayed
        :param surface: the surface on which the agent is displayed.
        """
        norm = mpl.colors.Normalize(vmin=-2, vmax=2)
        cmap = cm.jet_r
        cell_size = (surface.get_width() // agent.T, surface.get_height() // (agent.L * agent.V))
        velocity_size = surface.get_height() // agent.V

        for h in range(agent.V):
            for i in range(agent.L):
                for j in range(agent.T):
                    color = cmap(norm(agent.value[h, i, j]), bytes=True)
                    pygame.draw.rect(surface, color, (
                        j * cell_size[0], i * cell_size[1] + h * velocity_size, cell_size[0], cell_size[1]), 0)
            pygame.draw.line(surface, cls.BLACK, (0, h * velocity_size), (agent.T * cell_size[0], h * velocity_size), 1)
        path, actions = agent.pick_trajectory()
        for (h, i, j) in path:
            pygame.draw.rect(surface, cls.RED,
                             (j * cell_size[0], i * cell_size[1] + h * velocity_size, cell_size[0], cell_size[1]), 1)
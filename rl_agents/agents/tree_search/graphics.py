import pygame
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np

from rl_agents.agents.common import preprocess_env


class MCTSGraphics(object):
    """
        Graphical visualization of the MCTSAgent tree.
    """
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)

    @classmethod
    def display(cls, agent, surface):
        """
            Display the whole tree of an MCTSAgent.

        :param agent: the MCTSAgent to be displayed
        :param surface: the pygame surface on which the agent is displayed
        """
        cell_size = (surface.get_width() // agent.mcts.config['max_depth'], surface.get_height())
        pygame.draw.rect(surface, cls.BLACK, (0, 0, surface.get_width(), surface.get_height()), 0)
        cls.display_node(agent.mcts.root, agent.env.action_space, surface, (0, 0), cell_size,
                         temperature=agent.mcts.config['temperature'], depth=0, selected=True)

        actions = agent.mcts.get_plan()
        font = pygame.font.Font(None, 13)
        text = font.render('-'.join(map(str, actions)), 1, (10, 10, 10), (255, 255, 255))
        surface.blit(text, (1, surface.get_height()-10))

    @classmethod
    def display_node(cls, node, action_space, surface, origin, size,
                     temperature=0,
                     depth=0,
                     selected=False,
                     display_text=True,
                     display_prior=True):
        """
            Display an MCTS node at a given position on a surface.

        :param node: the MCTS node to be displayed
        :param action_space: the environment action space
        :param surface: the pygame surface on which the node is displayed
        :param origin: the location of the node on the surface [px]
        :param size: the size of the node on the surface [px]
        :param temperature: the temperature used for exploration bonus visualization
        :param depth: the depth of the node in the tree
        :param selected: whether the node is within a selected branch of the tree
        :param display_text: display a text showing the value and visitation count of the node
        :param display_prior: show the prior probability of each action
        """
        # Display node value
        cmap = cm.jet_r
        norm = mpl.colors.Normalize(vmin=-2, vmax=2)
        color = cmap(norm(node.value), bytes=True)
        pygame.draw.rect(surface, color, (origin[0], origin[1], size[0], size[1]), 0)

        # Add selection display
        if selected:
            pygame.draw.rect(surface, cls.RED, (origin[0], origin[1], size[0], size[1]), 1)

        if display_text and depth < 2:
            font = pygame.font.Font(None, 13)
            text = "{:.2f} / {:.2f} / {}".format(node.value, node.selection_strategy(temperature), node.count)
            if display_prior:
                text += " / {:.2f}".format(node.prior)
            text = font.render(text,
                               1, (10, 10, 10), (255, 255, 255))
            surface.blit(text, (origin[0]+1, origin[1]+1))

        # Recursively display children nodes
        best_action = node.select_action(temperature=0)
        for a in range(action_space.n):
            if a in node.children:
                action_selected = (selected and (a == best_action))
                cls.display_node(node.children[a], action_space, surface,
                                 (origin[0]+size[0], origin[1]+a*size[1]/action_space.n),
                                 (size[0], size[1]/action_space.n),
                                 depth=depth+1, temperature=temperature, selected=action_selected)


class DiscreteRobustMCTSGraphics(MCTSGraphics):
    @classmethod
    def display_node(cls, node, action_space, surface, origin, size,
                     temperature=0,
                     depth=0,
                     selected=False,
                     display_text=True,
                     display_prior=True):
        # Display node value
        cmap = cm.jet_r
        norm = mpl.colors.Normalize(vmin=-2, vmax=2)
        n = np.size(node.value)
        for i in range(n):
            v = node.value[i] if n > 1 else node.value
            color = cmap(norm(v), bytes=True)
            pygame.draw.rect(surface, color, (origin[0] + i/n*size[0], origin[1], size[0]/n, size[1]), 0)

        # Add selection display
        if selected:
            pygame.draw.rect(surface, cls.RED, (origin[0], origin[1], size[0], size[1]), 1)

        if display_text and depth < 2:
            font = pygame.font.Font(None, 13)
            text = "{} / {:.2f} / {}".format('-'.join(['{:.1f}'.format(i) for i in node.value]),
                                             node.selection_strategy(temperature), node.count)
            if display_prior:
                text += " / {:.2f}".format(node.prior)
            text = font.render(text,
                               1, (10, 10, 10), (255, 255, 255))
            surface.blit(text, (origin[0] + 1, origin[1] + 1))

        # Recursively display children nodes
        best_action = node.select_action(temperature=0)
        for a in range(action_space.n):
            if a in node.children:
                action_selected = (selected and (a == best_action))
                cls.display_node(node.children[a], action_space, surface,
                                 (origin[0] + size[0], origin[1] + a * size[1] / action_space.n),
                                 (size[0], size[1] / action_space.n),
                                 depth=depth + 1, temperature=temperature, selected=action_selected)


class OneStepRobustMCTSGraphics(object):
    @classmethod
    def display(cls, agent, surface):
        cell_size = (surface.get_width() // len(agent.agents), surface.get_height())
        pygame.draw.rect(surface, MCTSGraphics.BLACK, (0, 0, surface.get_width(), surface.get_height()), 0)
        for i, sub_agent in enumerate(agent.agents):
            sub_cell_size = (cell_size[0] // sub_agent.mcts.config["max_depth"], cell_size[1])
            MCTSGraphics.display_node(sub_agent.mcts.root, sub_agent.env.action_space, surface,
                                      (i*cell_size[0], 0), sub_cell_size,
                                      temperature=sub_agent.mcts.config['temperature'], depth=0, selected=True)


class IntervalRobustMCTSGraphics(object):
    """
        Graphical visualization of the IntervalRobustMCTS interval observer.
    """
    UNCERTAINTY_TIME_COLORMAP = cm.RdYlGn_r
    MODEL_TRAJ_COLOR = (0, 0, 255)
    RED = (255, 0, 0)
    TRANSPARENCY = 128

    @classmethod
    def display(cls, agent, agent_surface, sim_surface):
        horizon = 2
        robust_env = preprocess_env(agent.env, agent.config["env_preprocessors"])
        for action in agent.sub_agent.mcts.get_plan()[:horizon]:
            robust_env.step(action)
        for vehicle in robust_env.road.vehicles:
            if not hasattr(vehicle, 'observer_trajectory'):
                continue
            min_traj = [o.position[0] for o in vehicle.observer_trajectory]
            max_traj = [o.position[1] for o in vehicle.observer_trajectory]
            uncertainty_surface = pygame.Surface(sim_surface.get_size(), pygame.SRCALPHA, 32)
            cls.display_uncertainty(min_traj, max_traj, uncertainty_surface, sim_surface, cls.UNCERTAINTY_TIME_COLORMAP)
            cls.display_trajectory(vehicle.trajectory, uncertainty_surface, sim_surface, cls.MODEL_TRAJ_COLOR)
            sim_surface.blit(uncertainty_surface, (0, 0))
        MCTSGraphics.display(agent.sub_agent, agent_surface)

    @classmethod
    def display_trajectory(cls, trajectory, surface, sim_surface, color):
        for i in range(len(trajectory)-1):
            pygame.draw.line(surface, color,
                             (sim_surface.vec2pix(trajectory[i].position)),
                             (sim_surface.vec2pix(trajectory[i+1].position)),
                             1)

    @classmethod
    def display_box(cls, min_pos, max_pos, surface, sim_surface, color):
        rect = (sim_surface.vec2pix(min_pos),
                (sim_surface.pix(max_pos[0] - min_pos[0]),
                 sim_surface.pix(max_pos[1] - min_pos[1])))
        if rect[1] != (0, 0):
            pygame.draw.rect(surface, color, rect, 0)

    @classmethod
    def display_uncertainty(cls, min_traj, max_traj, surface, sim_surface, cmap, boxes=True):
        for i in reversed(range(len(min_traj))):
            for (A, B) in [(min_traj, max_traj), (min_traj, min_traj)]:
                color = cmap(i / len(min_traj), bytes=True)
                color = (color[0], color[1], color[2], cls.TRANSPARENCY)
                if boxes:
                    cls.display_box(min_traj[i], max_traj[i], surface, sim_surface, color)
                if i < len(min_traj)-1:
                    input_points = [[(A[i][0], min_traj[i][1]), (A[i][0], max_traj[i][1])],
                                    [(B[i][0], min_traj[i][1]), (A[i][0], max_traj[i][1])],
                                    [(A[i][0], min_traj[i][1]), (B[i][0], max_traj[i][1])]]
                    output_points = [[(B[i+1][0], min_traj[i+1][1]), (B[i+1][0], max_traj[i+1][1])],
                                     [(A[i+1][0], min_traj[i+1][1]), (B[i+1][0], max_traj[i+1][1])],
                                     [(B[i+1][0], min_traj[i+1][1]), (A[i+1][0], max_traj[i+1][1])]]
                    for p1 in input_points:
                        for p2 in output_points:
                            p = list(reversed(p1)) + p2
                            p.append(p[0])
                            p = list(map(sim_surface.vec2pix, p))
                            pygame.draw.polygon(surface, color, p, 0)

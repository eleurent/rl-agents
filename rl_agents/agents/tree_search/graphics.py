import pygame
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np

from rl_agents.agents.common import preprocess_env


class TreeGraphics(object):
    """
        Graphical visualization of a tree-search based agent.
    """
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)

    @classmethod
    def display(cls, agent, surface):
        """
            Display the whole tree.

        :param agent: the agent to be displayed
        :param surface: the pygame surface on which the agent is displayed
        """
        cell_size = (surface.get_width() // (agent.planner.config['max_depth'] + 1), surface.get_height())
        pygame.draw.rect(surface, cls.BLACK, (0, 0, surface.get_width(), surface.get_height()), 0)
        cls.display_node(agent.planner.root, agent.env.action_space, surface, (0, 0), cell_size,
                         config=agent.planner.config, depth=0, selected=True)

        actions = agent.planner.get_plan()
        font = pygame.font.Font(None, 13)
        text = font.render('-'.join(map(str, actions)), 1, (10, 10, 10), (255, 255, 255))
        surface.blit(text, (1, surface.get_height()-10))

    @classmethod
    def display_node(cls, node, action_space, surface, origin, size,
                     config=0,
                     depth=0,
                     selected=False):
        """
            Display an MCTS node at a given position on a surface.

        :param node: the MCTS node to be displayed
        :param action_space: the environment action space
        :param surface: the pygame surface on which the node is displayed
        :param origin: the location of the node on the surface [px]
        :param size: the size of the node on the surface [px]
        :param config: the agent configuration
        :param depth: the depth of the node in the tree
        :param selected: whether the node is within a selected branch of the tree
        """
        # Display node value
        cls.draw_node(node, surface, origin, size, config)

        # Add selection display
        if selected:
            pygame.draw.rect(surface, cls.RED, (origin[0], origin[1], size[0], size[1]), 1)

        if depth < 3:
            cls.display_text(node, surface, origin, config)

        # Recursively display children nodes
        if depth >= config["max_depth"]:
            return
        try:
            best_action = node.selection_rule()
        except ValueError:
            best_action = None
        for a in node.children:
            action_selected = (selected and (a == best_action))
            cls.display_node(node.children[a], action_space, surface,
                             (origin[0]+size[0], origin[1]+a*size[1]/action_space.n),
                             (size[0], size[1]/action_space.n),
                             depth=depth+1, config=config, selected=action_selected)

    @classmethod
    def draw_node(cls, node, surface, origin, size, config):
        cmap = cm.jet_r
        norm = mpl.colors.Normalize(vmin=0, vmax=config["gamma"] / (1 - config["gamma"]))
        color = cmap(norm(node.get_value()), bytes=True)
        pygame.draw.rect(surface, color, (origin[0], origin[1], size[0], size[1]), 0)

    @classmethod
    def display_text(cls, node, surface, origin, config):
        font = pygame.font.Font(None, 13)
        text = "{:.2f} / {}".format(node.get_value(), node.count)
        text = font.render(text,
                           1, (10, 10, 10), (255, 255, 255))
        surface.blit(text, (origin[0] + 1, origin[1] + 1))


class MCTSGraphics(TreeGraphics):
    @classmethod
    def display_text(cls, node, surface, origin, config):
        font = pygame.font.Font(None, 13)
        text = "{:.2f} / {:.2f} / {}".format(node.value, node.selection_strategy(config['temperature']), node.count)
        text += " / {:.2f}".format(node.prior)
        text = font.render(text,
                           1, (10, 10, 10), (255, 255, 255))
        surface.blit(text, (origin[0] + 1, origin[1] + 1))


class DiscreteRobustPlannerGraphics(TreeGraphics):
    @classmethod
    def display(cls, agent, agent_surface, sim_surface):
        horizon = 2
        for v in range(3):
            for destination in ["exr", "sxr"]:
                robust_env = preprocess_env(agent.env, agent.config["env_preprocessors"])
                robust_env.road.vehicles[1 + v].plan_route_to(destination)
                for action in agent.sub_agent.planner.get_plan()[:horizon]:
                    robust_env.step(action)
                for vehicle in robust_env.road.vehicles:
                    if not hasattr(vehicle, 'observer_trajectory'):
                        continue
                    min_traj = [o.position[0] for o in vehicle.observer_trajectory]
                    max_traj = [o.position[1] for o in vehicle.observer_trajectory]
                    uncertainty_surface = pygame.Surface(sim_surface.get_size(), pygame.SRCALPHA, 32)
                    cls.display_trajectory(vehicle.trajectory, uncertainty_surface, sim_surface, cls.MODEL_TRAJ_COLOR)
                    sim_surface.blit(uncertainty_surface, (0, 0))
            TreeGraphics.display(agent.sub_agent, agent_surface)

    @classmethod
    def draw_node(cls, node, surface, origin, size, config):
        cmap = cm.jet_r
        norm = mpl.colors.Normalize(vmin=0, vmax=config["gamma"] / (1 - config["gamma"]))
        n = np.size(node.value)
        for i in range(n):
            v = node.value[i] if n > 1 else node.value
            color = cmap(norm(v), bytes=True)
            pygame.draw.rect(surface, color, (origin[0] + i / n * size[0], origin[1], size[0] / n, size[1]), 0)

    @classmethod
    def display_text(cls, node, surface, origin, config):
        font = pygame.font.Font(None, 13)
        text = "{} / {:.2f} / {}".format('-'.join(['{:.1f}'.format(i) for i in node.get_value()]),
                                         node.selection_strategy(config["temperature"]), node.count)
        text += " / {:.2f}".format(node.prior)
        text = font.render(text,
                           1, (10, 10, 10), (255, 255, 255))
        surface.blit(text, (origin[0] + 1, origin[1] + 1))


class IntervalRobustPlannerGraphics(object):
    """
        Graphical visualization of the IntervalRobustPlannerAgent interval observer.
    """
    UNCERTAINTY_TIME_COLORMAP = cm.RdYlGn_r
    MODEL_TRAJ_COLOR = (0, 0, 255)
    RED = (255, 0, 0)
    TRANSPARENCY = 128

    @classmethod
    def display(cls, agent, agent_surface, sim_surface):
        horizon = 2
        robust_env = preprocess_env(agent.env, agent.config["env_preprocessors"])
        for action in agent.sub_agent.planner.get_plan()[:horizon]:
            robust_env.step(action)
        for vehicle in robust_env.road.vehicles:
            if not hasattr(vehicle, 'observer_trajectory'):
                continue
            min_traj = [o.position[0] for o in vehicle.observer_trajectory]
            max_traj = [o.position[1] for o in vehicle.observer_trajectory]
            uncertainty_surface = pygame.Surface(sim_surface.get_size(), pygame.SRCALPHA, 32)
            cls.display_uncertainty(min_traj, max_traj, uncertainty_surface, sim_surface, cls.UNCERTAINTY_TIME_COLORMAP)
            # cls.display_trajectory(vehicle.trajectory, uncertainty_surface, sim_surface, cls.MODEL_TRAJ_COLOR)
            sim_surface.blit(uncertainty_surface, (0, 0))
            TreeGraphics.display(agent.sub_agent, agent_surface)

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

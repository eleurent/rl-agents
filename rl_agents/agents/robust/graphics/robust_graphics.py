import matplotlib as mpl
import numpy as np
from matplotlib import cm as cm

from rl_agents.agents.common.factory import preprocess_env
from rl_agents.agents.tree_search.graphics import TreeGraphics


class DiscreteRobustPlannerGraphics(TreeGraphics):
    @classmethod
    def display(cls, agent, agent_surface, sim_surface):
        plan = agent.planner.get_plan()
        for env in [preprocess_env(agent.true_env, preprocessors) for preprocessors in agent.config["models"]]:
            IntervalRobustPlannerGraphics.display_uncertainty(env, plan, sim_surface, trajectory=False)
            # for vehicle in env.road.vehicles:
            #     if hasattr(vehicle, ):
            #         IntervalRobustPlannerGraphics.display(vehicle)
            # for vehicle in env.road.vehicles:
            #     vehicle.trajectory = []
            # for action in plan[:horizon] + (horizon - len(plan)) * [1]:
            #     env.step(action)
            # for vehicle in env.road.vehicles:
            #     if vehicle is env.vehicle:
            #         continue
            #     uncertainty_surface = pygame.Surface(sim_surface.get_size(), pygame.SRCALPHA, 32)
            #     IntervalRobustPlannerGraphics.display_trajectory(vehicle.trajectory, uncertainty_surface, sim_surface,
            #                                                      IntervalRobustPlannerGraphics.MODEL_TRAJ_COLOR)
            #     sim_surface.blit(uncertainty_surface, (0, 0))
        TreeGraphics.display(agent, agent_surface)

    @classmethod
    def draw_node(cls, node, surface, origin, size, config):
        import pygame
        cmap = cm.jet_r
        norm = mpl.colors.Normalize(vmin=0, vmax=config["gamma"] / (1 - config["gamma"]))
        n = np.size(node.value)
        for i in range(n):
            v = node.value[i] if n > 1 else node.value
            color = cmap(norm(v), bytes=True)
            pygame.draw.rect(surface, color, (origin[0] + i / n * size[0], origin[1], size[0] / n, size[1]), 0)


class IntervalRobustPlannerGraphics(object):
    """
        Graphical visualization of the IntervalRobustPlannerAgent interval observer.
    """
    UNCERTAINTY_TIME_COLORMAP = cm.RdYlGn
    MODEL_TRAJ_COLOR = (0, 0, 255)
    RED = (255, 0, 0)
    TRANSPARENCY = 128

    @classmethod
    def display(cls, agent, agent_surface, sim_surface):
        robust_env = preprocess_env(agent.env, agent.config["env_preprocessors"])
        cls.display_uncertainty(robust_env, plan=agent.get_plan(), surface=sim_surface)
        if agent_surface and hasattr(agent, "sub_agent"):
            TreeGraphics.display(agent.sub_agent, agent_surface)

    @classmethod
    def display_uncertainty(cls, robust_env, plan, surface, trajectory=True):
        import pygame
        horizon = 2
        for vehicle in robust_env.road.vehicles:
            vehicle.COLLISIONS_ENABLED = False
        if plan:
            plan = plan[1:]  # First action has already been performed
        plan = plan[:horizon] + (horizon - len(plan)) * [1]
        for action in plan:
            robust_env.step(action)
        for vehicle in robust_env.road.vehicles:
            if not hasattr(vehicle, 'interval_trajectory'):
                continue
            min_traj = [o.position[0].clip(vehicle.position - 100, vehicle.position + 100) for o in vehicle.interval_trajectory]
            max_traj = [o.position[1].clip(vehicle.position - 100, vehicle.position + 100) for o in vehicle.interval_trajectory]
            uncertainty_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA, 32)
            cls.display_traj_uncertainty(min_traj, max_traj, uncertainty_surface, surface, cls.UNCERTAINTY_TIME_COLORMAP)
            if trajectory:
                cls.display_trajectory(vehicle.trajectory, uncertainty_surface, surface, cls.MODEL_TRAJ_COLOR)
            surface.blit(uncertainty_surface, (0, 0))

    @classmethod
    def display_trajectory(cls, trajectory, surface, sim_surface, color):
        import pygame
        color = (color[0], color[1], color[2], cls.TRANSPARENCY)
        pos = lambda x: getattr(x, "position", x)
        for i in range(len(trajectory)-1):
            pygame.draw.line(surface, color,
                             sim_surface.vec2pix(pos(trajectory[i])),
                             sim_surface.vec2pix(pos(trajectory[i+1])),
                             2)

    @classmethod
    def display_box(cls, min_pos, max_pos, surface, sim_surface, color):
        import pygame
        rect = (sim_surface.vec2pix(min_pos),
                (sim_surface.pix(max_pos[0] - min_pos[0]),
                 sim_surface.pix(max_pos[1] - min_pos[1])))
        try:
            if rect[1] != (0, 0):
                pygame.draw.rect(surface, color, rect, 0)
        except TypeError as e:
            print(e)

    @classmethod
    def display_traj_uncertainty(cls, min_traj, max_traj, surface, sim_surface, cmap, boxes=True):
        import pygame
        min_traj = np.clip(min_traj, -1000, 1000)
        max_traj = np.clip(max_traj, -1000, 1000)
        for i in reversed(range(len(min_traj))):
            for (A, B) in [(min_traj, max_traj), (min_traj, min_traj)]:
                color = cmap(i / len(min_traj), bytes=True)
                color = (color[0], color[1], color[2], cls.TRANSPARENCY)
                if boxes:
                    cls.display_box(min_traj[i], max_traj[i], surface, sim_surface, color)
                if i > 0:
                    input_points = [[(A[i-1][0], min_traj[i-1][1]), (A[i-1][0], max_traj[i-1][1])],
                                    [(B[i-1][0], min_traj[i-1][1]), (A[i-1][0], max_traj[i-1][1])],
                                    [(A[i-1][0], min_traj[i-1][1]), (B[i-1][0], max_traj[i-1][1])]]
                    output_points = [[(B[i][0], min_traj[i][1]), (B[i][0], max_traj[i][1])],
                                     [(A[i][0], min_traj[i][1]), (B[i][0], max_traj[i][1])],
                                     [(B[i][0], min_traj[i][1]), (A[i][0], max_traj[i][1])]]
                    for p1 in input_points:
                        for p2 in output_points:
                            try:
                                p = list(reversed(p1)) + p2
                                p.append(p[0])
                                p = list(map(sim_surface.vec2pix, p))
                                pygame.draw.polygon(surface, color, p, 0)
                            except TypeError as e:
                                print(e, p)
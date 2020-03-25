import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from rl_agents.agents.robust.graphics.robust_epc_graphics import RobustEPCGraphics, confidence_ellipse
from rl_agents.agents.robust.robust_epc import NominalEPCAgent

matplotlib.rc('text', usetex=False)
import seaborn as sns


class ConstrainedEPCGraphics(RobustEPCGraphics):
    display_prediction = True

    @classmethod
    def display(cls, agent, agent_surface, sim_surface):
        import pygame
        robust_env = agent.robustify_env()
        robust_env.unwrapped.config["state_noise"] = 0
        robust_env.unwrapped.config["derivative_noise"] = 0
        cls.display_attraction_basin(robust_env, agent, sim_surface)

        observation = agent.observation
        x_t = robust_env.unwrapped.lpv.change_coordinates(robust_env.unwrapped.lpv.x_t, back=True)
        cls.display_prediction = cls.display_prediction and not \
            np.all(np.abs(x_t) <= agent.feedback.Xf[agent.feedback.Xf.size // 2:])
        if cls.display_prediction:
            for time in range(30):
                control = agent.act(observation)
                observation, _, _, _ = robust_env.step(control)
            cls.display_uncertainty(robust_env=robust_env, plan=[], surface=sim_surface, trajectory=False)
        cls.display_agent(agent, agent_surface)

    @classmethod
    def display_attraction_basin(cls, env, agent, surface, alpha=50):
        if "highway_env" not in env.unwrapped.__module__:
            return
        try:
            dy = agent.feedback.Xf[0]
        except AttributeError:
            return

        import pygame
        from highway_env.road.graphics import LaneGraphics
        basin_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA, 32)
        LaneGraphics.draw_ground(env.unwrapped.lane, surface, color=(*surface.GREEN, alpha),
                                 width=dy, draw_surface=basin_surface)
        surface.blit(basin_surface, (0, 0))

    @classmethod
    def display_uncertainty(cls, robust_env, plan, surface, trajectory=True):
        import pygame
        horizon = 30
        robust_env.unwrapped.trajectory = []

        min_traj = [o[0] for o in robust_env.unwrapped.interval_trajectory]
        max_traj = [o[1] for o in robust_env.unwrapped.interval_trajectory]
        min_traj = np.clip(min_traj, -1000, 1000)
        max_traj = np.clip(max_traj, -1000, 1000)
        uncertainty_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA, 32)
        if trajectory:
            cls.display_trajectory(robust_env.unwrapped.trajectory, uncertainty_surface, surface, cls.MODEL_TRAJ_COLOR)
        else:
            cls.display_traj_uncertainty(min_traj, max_traj, uncertainty_surface, surface, cls.UNCERTAINTY_TIME_COLORMAP)
        surface.blit(uncertainty_surface, (0, 0))

    @classmethod
    def plot_ellipsoid(cls, ellipsoids, true_theta, config, title="", figsize=(8, 6), save_to=None):
        """
            Plot the hull of all ellipsoids.

            If a threshold beta and corresponding mixture is provided, plot them.
        """
        # Figure creation
        sns.set(font_scale=1)
        # sns.set_style("white")
        fig = plt.figure(figsize=figsize, tight_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        plt.title(title)
        for ellipsoid in ellipsoids[:20:5] + ellipsoids[20:-1:20]:
            confidence_ellipse(ellipsoid, ax, facecolor=(1, 0.3, 0.3, 0.1),
                               edgecolor="black", linewidth=0.5, label=None)
        confidence_ellipse(ellipsoids[-1], ax, facecolor=(1, 0.3, 0.3, 0.1),
                           edgecolor='red', label=r"$\hat{\Theta}$")
        plt.plot(true_theta[0], true_theta[1], '.', label=r"$\theta$")
        plt.legend(loc="upper right")
        bound = config["parameter_box"]
        ax.set_xlim(bound[0][0], bound[1][0])
        ax.set_ylim(bound[0][1], bound[1][1])
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")

        # Figure export
        if save_to is not None and len(ellipsoids) % 10 == 0:
            plt.savefig(save_to)
            plt.savefig(save_to.with_suffix(".png"))
        fig.canvas.draw()
        data_str = fig.canvas.tostring_rgb()
        plt.close()
        return data_str, fig.canvas.get_width_height()
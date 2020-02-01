import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib
# matplotlib.rc('text', usetex=True)

from rl_agents.agents.tree_search.graphics.graphics import TreeGraphics
from rl_agents.agents.tree_search.graphics.robust import IntervalRobustPlannerGraphics


class RobustEPCGraphics(IntervalRobustPlannerGraphics):
    @classmethod
    def display(cls, agent, agent_surface, sim_surface):
        import pygame
        robust_env = agent.robustify_env()
        cls.display_uncertainty(robust_env=robust_env, plan=agent.get_plan(), surface=sim_surface)
        if agent_surface and hasattr(agent, "sub_agent"):
            # TreeGraphics.display(agent.sub_agent, agent_surface)
            true = agent.env.unwrapped.dynamics.theta
            surf_size = agent_surface.get_size()
            ellipsoid = agent.theta_n_lambda, agent.g_n_lambda, agent.beta_n
            image_str, size = plot_ellipsoid(ellipsoid, true, None, figsize=(surf_size[0]/100, surf_size[1]/100))
            surf = pygame.image.fromstring(image_str, size, "RGB")
            agent_surface.blit(surf, (0, 0))

    @classmethod
    def display_uncertainty(cls, robust_env, plan, surface, trajectory=True):
        import pygame
        horizon = 3
        if plan:
            plan = plan[1:]  # First action has already been performed
        plan = plan[:horizon]
        for action in plan:
            robust_env.step(action)
        min_traj = [o[0] for o in robust_env.unwrapped.interval_trajectory]
        max_traj = [o[1] for o in robust_env.unwrapped.interval_trajectory]
        uncertainty_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA, 32)
        cls.display_traj_uncertainty(min_traj, max_traj, uncertainty_surface, surface, cls.UNCERTAINTY_TIME_COLORMAP)
        # if trajectory:
        #     cls.display_trajectory(robust_env.trajectory, uncertainty_surface, surface, cls.MODEL_TRAJ_COLOR)
        surface.blit(uncertainty_surface, (0, 0))

    # @classmethod
    # def display_trajectory(cls, trajectory, surface, sim_surface, color):
    #     import pygame
    #     color = (color[0], color[1], color[2], cls.TRANSPARENCY)
    #     for i in range(len(trajectory)-1):
    #         pygame.draw.line(surface, color,
    #                          (sim_surface.vec2pix(trajectory[i].position)),
    #                          (sim_surface.vec2pix(trajectory[i+1].position)),
    #                          2)


def plot_ellipsoid(ellipsoid, true, writer=None, epoch=0, title="", figsize=(8, 6)):
    """
        Plot the hull of all Qc, Qr points for different (action, budget).

        If a threshold beta and corresponding mixture is provided, plot them.
    :param SummaryWriter writer: will log the image to tensorboard if not None
    :param epoch: timestep for tensorboard log
    :param title: figure title
    :param figsize: figure size, inches
    :return: the string description of the image, and its size
    """
    # Figure creation
    fig = plt.figure(figsize=figsize, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    plt.title(title)
    center, cov, beta = ellipsoid
    cov = np.linalg.inv(cov / beta)
    confidence_ellipse(center, cov, ax, edgecolor='red', label=r"$\mathcal{C}_N$")
    plt.plot(true[0], true[1], '.', label=r"$\theta$")
    plt.legend()
    ax.set_xlim(-0.2, 0.7)
    ax.set_ylim(-0.2, 0.7)

    # Figure export
    fig.canvas.draw()
    data_str = fig.canvas.tostring_rgb()
    if writer:
        data = np.fromstring(data_str, dtype=np.uint8, sep='')
        data = np.rollaxis(data.reshape(fig.canvas.get_width_height()[::-1] + (3,)), 2, 0)
        writer.add_image(title, data, epoch)
    plt.close()
    return data_str, fig.canvas.get_width_height()


def confidence_ellipse(center, cov, ax, facecolor='none', **kwargs):
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    scale_x = np.sqrt(cov[0, 0])
    scale_y = np.sqrt(cov[1, 1])

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(center[0], center[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

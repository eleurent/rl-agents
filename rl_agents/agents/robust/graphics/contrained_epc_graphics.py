import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from rl_agents.agents.robust.graphics.robust_epc_graphics import RobustEPCGraphics
from rl_agents.agents.robust.robust_epc import NominalEPCAgent

matplotlib.rc('text', usetex=False)
import seaborn as sns



class ConstrainedEPCGraphics(RobustEPCGraphics):
    SAVE_IMAGES = None

    @classmethod
    def display(cls, agent, agent_surface, sim_surface):
        import pygame
        robust_env = agent.robustify_env()
        show_traj = isinstance(agent, NominalEPCAgent)
        cls.display_uncertainty(robust_env=robust_env, plan=agent.get_plan(), surface=sim_surface, trajectory=show_traj)
        if agent_surface and hasattr(agent, "sub_agent"):
            true_theta = agent.env.unwrapped.dynamics.theta
            surf_size = agent_surface.get_size()
            figsize = (surf_size[0]/100, surf_size[1]/100)
            save_to = None
            if cls.SAVE_IMAGES:
                save_to = agent.evaluation.run_directory / "ellipsoid.{}.{}.pdf".format(agent.evaluation.episode, len(agent.ellipsoids))
            image_str, size = cls.plot_ellipsoid(agent.ellipsoids, true_theta, config=agent.config, figsize=figsize, save_to=save_to)
            surf = pygame.image.fromstring(image_str, size, "RGB")
            agent_surface.blit(surf, (0, 0))

    def display_attraction_basin(self):
        pass
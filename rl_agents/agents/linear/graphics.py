import numpy as np
from matplotlib import pyplot as plt, gridspec as gridspec

import pygame
import matplotlib as mpl
import matplotlib.cm as cm


class LinearModelGraphics(object):
    """
        Graphical visualization of the LinearModelAgent interval observer.
    """
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    MIN_VALUE = -2
    MAX_VALUE = 2

    @classmethod
    def display(cls, agent, surface):
        for observer in agent.observers:
            model_traj, min_traj, max_traj = observer.compute_trajectories(30, 1/15)
            cls.display_bounding_polygons(min_traj, max_traj, surface, cls.RED)
            cls.display_trajectory(model_traj, surface, cls.BLUE)

    @classmethod
    def display_trajectory(cls, trajectory, surface, color):
        for i in range(len(trajectory)-1):
            pygame.draw.line(surface, color,
                             (surface.vec2pix(trajectory[i].position)),
                             (surface.vec2pix(trajectory[i+1].position)),
                             1)

    @classmethod
    def display_boxes(cls, min_traj, max_traj, surface, color):
        for i in range(len(min_traj)):
            pygame.draw.rect(surface, color,
                             (surface.vec2pix(min_traj[i].position),
                              (surface.pix(max_traj[i].position[0] - min_traj[i].position[0]),
                              surface.pix(max_traj[i].position[1] - min_traj[i].position[1]))), 0)

    @classmethod
    def display_bounding_polygons(cls, min_traj, max_traj, surface, color):
        for (A, B) in [(min_traj, max_traj), (min_traj, min_traj)]:
            for i in range(len(min_traj)-1):
                input_points = [[(A[i].position[0], min_traj[i].position[1]), (A[i].position[0], max_traj[i].position[1])],
                                [(B[i].position[0], min_traj[i].position[1]), (A[i].position[0], max_traj[i].position[1])],
                                [(A[i].position[0], min_traj[i].position[1]), (B[i].position[0], max_traj[i].position[1])]]
                output_points = [[(B[i+1].position[0], min_traj[i+1].position[1]), (B[i+1].position[0], max_traj[i+1].position[1])],
                                 [(A[i+1].position[0], min_traj[i+1].position[1]), (B[i+1].position[0], max_traj[i+1].position[1])],
                                 [(B[i+1].position[0], min_traj[i+1].position[1]), (A[i+1].position[0], max_traj[i+1].position[1])]]
                for p1 in input_points:
                    for p2 in output_points:
                        p = list(reversed(p1)) + p2
                        p.append(p[0])
                        p = list(map(surface.vec2pix, p))
                        pygame.draw.polygon(surface, color, p, 0)

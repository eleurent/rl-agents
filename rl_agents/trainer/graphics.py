from __future__ import division, print_function
import pygame
import numpy as np
import matplotlib.pyplot as plt

from rl_agents.agents.graphics import AgentGraphics
from highway_env.envs.graphics import EnvViewer
from highway_env.road.graphics import WorldSurface, RoadGraphics
from highway_env.vehicle.graphics import VehicleGraphics


class SimulationViewer(EnvViewer):
    """
        A simulation viewer displays at the same place the representation of an environment state and the reasoning of
        an agent.
    """

    def __init__(self, simulation):
        self.simulation = simulation
        self.SCREEN_HEIGHT *= 2
        super(SimulationViewer, self).__init__(self.simulation.highway_env)

        panel_size = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT / 2)
        self.agent_surface = pygame.Surface(panel_size)
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))

    def display(self):
        """
            Display the road, vehicles and trajectory prediction on the first panel, and agent reasoning on the second.
        """
        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)
        if self.simulation.planned_trajectory:
            VehicleGraphics.display_trajectory(self.simulation.planned_trajectory, self.sim_surface)
        RoadGraphics.display_traffic(self.env.road, self.sim_surface)
        self.screen.blit(self.sim_surface, (0, 0))

        AgentGraphics.display(self.simulation.agent, self.agent_surface)
        self.screen.blit(self.agent_surface, (0, self.SCREEN_HEIGHT / 2))

        self.clock.tick(self.env.SIMULATION_FREQUENCY)
        pygame.display.flip()


class RewardViewer(object):
    def __init__(self):
        self.rewards = []
        plt.ion()

    def update(self, reward):
        self.rewards.append(reward)
        self.display()

    def display(self):
        plt.figure(num='Rewards')
        plt.clf()
        plt.title('Total reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(self.rewards)

        # Take 100 episode averages and plot them too
        if len(self.rewards) >= 100:
            means = np.hstack((np.zeros((100,)), np.convolve(self.rewards, np.ones((100,)) / 100, mode='valid')))
            plt.plot(means)
        else:
            plt.plot(np.zeros(np.shape(self.rewards)))

        plt.pause(0.001)
        plt.draw()

from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import pygame

from rl_agents.agents.graphics import AgentGraphics


class AgentViewer(object):
    """
        A viewer to render an environment.
    """
    SCREEN_WIDTH = 400
    SCREEN_HEIGHT = 400
    FREQUENCY = 30

    def __init__(self, agent):
        self.agent = agent

        pygame.init()
        pygame.display.set_caption(agent.__class__.__name__)
        self.screen = pygame.display.set_mode([self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
        self.clock = pygame.time.Clock()

    def handle_events(self):
        """
            Handle pygame events by forwarding them to the display and environment vehicle.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def render(self):
        """
            Render the agent on a pygame window.
        """
        if self.agent:
            AgentGraphics.display(self.agent, self.screen)

        self.clock.tick(self.FREQUENCY)
        pygame.display.flip()

    def get_image(self):
        """
        :return: the rendered image as a rbg array
        """
        data = pygame.surfarray.array3d(self.screen)
        return np.moveaxis(data, 0, 1)

    @staticmethod
    def close():
        """
            Close the pygame window.
        """
        pygame.quit()


class RewardViewer(object):
    def __init__(self):
        self.rewards = []

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
        plt.plot(block=False)

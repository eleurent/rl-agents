from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import pygame

from rl_agents.agents.graphics import AgentGraphics



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

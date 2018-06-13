from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np


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

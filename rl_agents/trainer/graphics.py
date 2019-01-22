from __future__ import division, print_function
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()


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

        rewards = pd.Series(self.rewards)
        means = rewards.rolling(window=100).mean()
        plt.plot(rewards)
        plt.plot(means)
        plt.pause(0.001)
        plt.plot(block=False)

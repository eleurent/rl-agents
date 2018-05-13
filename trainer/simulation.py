import numpy as np
import matplotlib.pyplot as plt


class Simulation:
    def __init__(self, env, agent, num_episodes=1000, agent_viewer=None):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.agent_viewer = agent_viewer
        self.reward_viewer = RewardViewer()

    def train(self):
        """
            Train the model to take actions in an environment
            and maximize its rewards
        """
        state = self.env.reset()
        mins = np.copy(state)
        maxs = np.copy(state)
        for episode in range(self.num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                self.env.render()
                # Take action.
                action = self.agent.act(state)
                # Step environment.
                prev_state = state
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                # Record the experience.
                self.agent.record(prev_state, action, reward, state, done)
                mins = np.minimum(mins, state)
                maxs = np.maximum(maxs, state)
                print(state, mins, maxs)


            # End of episode
            self.reward_viewer.update(total_reward)
            self.agent_viewer.display()
            print("Episode {} score: {}".format(episode, total_reward))


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

import numpy as np
import matplotlib.pyplot as plt


class Simulation:
    def __init__(self, env, agent, episodes):
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.rewards = []

    def train(self):
        """
            Train the model to take actions in an environment
            and maximize its rewards
        """
        max_total_reward = -float('inf')

        for episode in range(self.episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                self.env.render()
                # Take action.
                action = self.agent.pick_action(state)
                # Step environment.
                prev_state = state
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                # Record the experience.
                self.agent.record(prev_state, action, reward, state, done)

            # End of episode
            self.rewards.append(total_reward)
            if total_reward > max_total_reward:
                max_total_reward = total_reward
            print("Episode {} score: {} (max {})".format(episode, total_reward, max_total_reward))

            self.display()
            self.agent.display()

    def display(self):
        self.plot_rewards()

    def plot_rewards(self):
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
            plt.plot(0 * self.rewards)

        plt.pause(0.001)

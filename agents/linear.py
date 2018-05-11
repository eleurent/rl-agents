import numpy as np

class LinearAgent(object):
    def __init__(self, env, config):
        self.env = env
        self.config = config

    def test(self, num_episodes=3):
        for _ in range(num_episodes):
            done = False
            observation = self.env.reset()
            while not done:
                action = self.act(observation)
                observation, reward, done, _ = self.env.step(action)
                self.env.render()

    def act(self, observation):
        u = np.dot(self.config['K'], -observation)
        action = 1 if u < 0 else 0
        return action

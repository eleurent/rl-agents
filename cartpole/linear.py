
class LinearAgent(object):
    Kx = [0.5, 1]
    Ka = [5, 10]

    def __init__(self, env, config):
        self.env = env

    def test(self, num_episodes=3):
        for _ in range(num_episodes):
            done = False
            observation = self.env.reset()
            while not done:
                action = self.act(observation)
                observation, reward, done, _ = self.env.step(action)

    def act(self, observation):
        ux = self.Kx[0]*(0 - observation[0]) - self.Kx[1]*observation[1]
        ua = self.Ka[0]*(ux - observation[2]) - self.Ka[1]*observation[3]
        action = 1 if ua < 0 else 0
        return action

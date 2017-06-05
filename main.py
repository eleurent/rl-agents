import gym
from gym import wrappers
from agents.dqn import DQN

if __name__ == "__main__":
    envname = 'MountainCar-v0'
    env = gym.make(envname)
    env = wrappers.Monitor(env, 'tmp/'+envname, force=True)
    config = {
        "neuralNet": [100, 100],
        "memory": 50000,
        "batchSize": 100,
        "episodes": 2000,
        "gamma": 0.99,
        "epsilon": [1.0, 0.01],
        "tau": 6000,
    }
    agent = DQN(env, config)
    agent.train()
import gym
from gym import wrappers
import numpy as np

from cartpole.dqn import DQN
from cartpole.linear import LinearAgent


def make_env():
    envname = 'CartPole-v0'
    env = gym.make(envname)
    env = wrappers.Monitor(env, 'tmp/' + envname, force=True)
    return env


def run_dqn():
    config = {
        "neuralNet": [100, 100],
        "memory": 50000,
        "batchSize": 100,
        "episodes": 2000,
        "gamma": 0.99,
        "epsilon": [1.0, 0.01],
        "tau": 6000,
    }
    agent = DQN(make_env(), config)
    agent.train()


def run_linear():
    env = make_env()
    config = {
        'K': np.array([[1, 20, 20, 30]])
    }
    agent = LinearAgent(env, config)
    agent.test()
    env.close()


if __name__ == "__main__":
    run_dqn()

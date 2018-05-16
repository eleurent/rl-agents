import gym
from gym import wrappers
import numpy as np

import obstacle_env

from agents.dqn_keras import DqnKerasAgent
from agents.dqn_pytorch import DqnPytorchAgent
from agents.linear import LinearAgent
from agents.utils import ValueFunctionViewer
from trainer.simulation import Simulation
from trainer.state_sampler import ObstacleStateSampler


def make_env():
    env_name = 'obstacle-v0'
    env = gym.make(env_name)
    env = wrappers.Monitor(env, 'tmp/' + env_name, force=True)
    sampler = ObstacleStateSampler()
    return env, sampler


def dqn_keras(env):
    config = {
        "layers": [100, 100],
        "memory_capacity": 50000,
        "batch_size": 100,
        "gamma": 0.9,
        "epsilon": [1.0, 0.01],
        "epsilon_tau": 50000,
        "target_update": 1
    }
    return DqnKerasAgent(env, config)


def dqn_pytorch(env):
    config = {
        "layers": [256, 256],
        "memory_capacity": 50000*5,
        "batch_size": 200,
        "gamma": 0.9,
        "epsilon": [1.0, 0.01],
        "epsilon_tau": 50000*5,
        "target_update": 100
    }
    return DqnPytorchAgent(env, config)


if __name__ == "__main__":
    env, sampler = make_env()
    agent = dqn_pytorch(env)
    agent_viewer = ValueFunctionViewer(agent, sampler)
    sim = Simulation(env, agent, num_episodes=50000, agent_viewer=agent_viewer)
    sim.train()


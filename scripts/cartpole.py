import gym
import numpy as np

from rl_agents.agents.linear.linear import LinearAgent
from rl_agents.agents.dqn.dqn_keras import DqnKerasAgent
from rl_agents.agents.dqn.dqn_pytorch import DqnPytorchAgent
from rl_agents.agents.dqn.graphics import ValueFunctionViewer
from rl_agents.trainer.simulation import Simulation
from rl_agents.trainer.state_sampler import CartPoleStateSampler
from rl_agents.wrappers.monitor import MonitorV2


def make_env():
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env = MonitorV2(env, 'out/' + env_name)
    sampler = CartPoleStateSampler()
    return env, sampler


def dqn_keras(env):
    config = {
        "layers": [100, 100],
        "memory_capacity": 50000,
        "batch_size": 100,
        "gamma": 0.99,
        "epsilon": [1.0, 0.01],
        "epsilon_tau": 6000,
        "target_update": 1
    }
    return DqnKerasAgent(env, config)


def dqn_pytorch(env):
    config = {
        "layers": [100, 100],
        "memory_capacity": 50000,
        "batch_size": 100,
        "gamma": 0.99,
        "epsilon": [1.0, 0.01],
        "epsilon_tau": 6000,
        "target_update": 1
    }
    return DqnPytorchAgent(env, config)


def linear(env):
    config = {
        'K': np.array([[1, 20, 20, 30]])
    }
    return LinearAgent(env, config)


if __name__ == "__main__":
    env, sampler = make_env()
    agent = dqn_pytorch(env)
    agent_viewer = ValueFunctionViewer(agent, sampler)
    sim = Simulation(env, agent, num_episodes=200, agent_viewer=agent_viewer)
    sim.train()


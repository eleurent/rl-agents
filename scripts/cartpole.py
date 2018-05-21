import gym
import numpy as np

from rl_agents.agents.linear.linear import LinearAgent
from rl_agents.agents.dqn.dqn_keras import DqnKerasAgent
from rl_agents.agents.dqn.dqn_pytorch import DqnPytorchAgent
from rl_agents.agents.tree_search.mcts import MCTSAgent
from rl_agents.trainer.simulation import Simulation
from rl_agents.trainer.state_sampler import CartPoleStateSampler


def make_env():
    env_name = 'CartPole-v0'
    environment = gym.make(env_name)
    env_sampler = CartPoleStateSampler()
    return environment, env_sampler


def dqn_keras(environment):
    config = {
        "layers": [100, 100],
        "memory_capacity": 50000,
        "batch_size": 100,
        "gamma": 0.99,
        "epsilon": [1.0, 0.01],
        "epsilon_tau": 6000,
        "target_update": 1
    }
    return DqnKerasAgent(environment, config)


def dqn_pytorch(environment):
    config = {
        "layers": [100, 100],
        "memory_capacity": 50000,
        "batch_size": 100,
        "gamma": 0.99,
        "epsilon": [1.0, 0.01],
        "epsilon_tau": 6000,
        "target_update": 1
    }
    return DqnPytorchAgent(environment, config)


def linear(environment):
    config = {
        'K': np.array([[1, 20, 20, 30]])
    }
    return LinearAgent(environment, config)


def mcts(environment):
    return MCTSAgent(environment,
                     rollout_policy=MCTSAgent.random_policy,
                     prior_policy=MCTSAgent.random_policy,
                     iterations=40,
                     temperature=200,
                     max_depth=10)


if __name__ == "__main__":
    gym.logger.set_level(gym.logger.INFO)
    env, sampler = make_env()
    agent = dqn_pytorch(env)
    sim = Simulation(env, agent, num_episodes=200)
    sim.train()


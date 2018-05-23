import gym

from rl_agents.agents.dqn.dqn_keras import DQNKerasAgent
from rl_agents.agents.dqn.dqn_pytorch import DQNPytorchAgent
from rl_agents.agents.dqn.graphics import ValueFunctionViewer
from rl_agents.agents.tree_search.mcts import MCTSAgent
from rl_agents.trainer.simulation import Simulation
from rl_agents.trainer.state_sampler import MountainCarStateSampler


def make_env():
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    sampler = MountainCarStateSampler()
    return env, sampler


def dqn_keras(env):
    config = {
        "layers": [100, 100],
        "memory_capacity": 50000,
        "batch_size": 100,
        "gamma": 0.99,
        "epsilon": [1.0, 0.01],
        "epsilon_tau": 10000,
        "target_update": 1
    }
    return DQNKerasAgent(env, config)


def dqn_pytorch(env):
    config = {
        "layers": [100, 100],
        "memory_capacity": 50000,
        "batch_size": 100,
        "gamma": 0.99,
        "epsilon": [1.0, 0.01],
        "epsilon_tau": 10000,
        "target_update": 1
    }
    return DQNPytorchAgent(env, config)


def mcts(environment):
    return MCTSAgent(environment,
                     rollout_policy=MCTSAgent.random_policy,
                     prior_policy=MCTSAgent.random_policy,
                     iterations=50,
                     temperature=200,
                     max_depth=10)


if __name__ == "__main__":
    gym.logger.set_level(gym.logger.INFO)
    env, sampler = make_env()
    agent = mcts(env)
    sim = Simulation(env, agent, num_episodes=300)
    sim.test()


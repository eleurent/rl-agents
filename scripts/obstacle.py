import multiprocessing

import gym

from rl_agents.agents.dqn.dqn_pytorch import DQNPytorchAgent
from rl_agents.agents.tree_search.mcts import MCTSAgent
from rl_agents.trainer.simulation import Simulation
from rl_agents.trainer.state_sampler import ObstacleStateSampler

import obstacle_env


def make_env():
    env_name = 'obstacle-v0'
    environment = gym.make(env_name)
    env_sampler = ObstacleStateSampler()
    return environment, env_sampler


def dqn_pytorch(environment):
    config = {
        "layers": [100, 100],
        "memory_capacity": 50000,
        "batch_size": 100,
        "gamma": 0.9,
        "epsilon": [1.0, 0.01],
        "epsilon_tau": 50000,
        "target_update": 1
    }
    return DQNPytorchAgent(environment, config)


def mcts(environment):
    return MCTSAgent(environment,
                     rollout_policy=MCTSAgent.random_policy,
                     prior_policy=lambda state:MCTSAgent.preference_policy(state, 0, 0.5),
                     iterations=100,
                     temperature=150,
                     max_depth=5)


def main():
    gym.logger.set_level(gym.logger.INFO)
    env, sampler = make_env()
    # agent = dqn_pytorch(env)
    agent = mcts(env)
    sim = Simulation(env, agent, num_episodes=80)
    sim.train()


if __name__ == "__main__":
    for i in range(4):
        p = multiprocessing.Process(target=main)
        p.start()


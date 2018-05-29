import gym

from rl_agents.agents.dqn.dqn_pytorch import DQNPytorchAgent
from rl_agents.agents.tree_search.mcts import MCTSAgent
from rl_agents.trainer.simulation import Simulation


def dqn_pytorch(env):
    config = dict(
        model=dict(layers=[100, 100]),
        exploration=dict(tau=6000),
    )
    return DQNPytorchAgent(env, config)


def mcts(environment):
    return MCTSAgent(environment,
                     iterations=50,
                     temperature=200,
                     max_depth=10)


if __name__ == "__main__":
    gym.logger.set_level(gym.logger.INFO)
    env, sampler = gym.make('MountainCar-v0')
    agent = mcts(env)
    sim = Simulation(env, agent, num_episodes=300)
    sim.test()


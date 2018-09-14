import gym
from rl_agents.agents.tree_search.mcts import MCTSAgent


def test_cartpole():
    env = gym.make('CartPole-v0')
    agent = MCTSAgent(env, config=dict(budget=400, temperature=200, max_depth=10))

    state = env.reset()
    done = False
    steps = 0
    while not done:
        action = agent.act(state)
        assert action is not None

        next_state, reward, done, info = env.step(action)
        steps += 1

    assert steps == env._max_episode_steps

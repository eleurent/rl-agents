import gymnasium as gym
import pytest

torch = pytest.importorskip("torch")


def test_cartpole():
    from rl_agents.agents.deep_q_network.pytorch import DQNAgent

    env = gym.make('CartPole-v0')
    agent = DQNAgent(env, config=None)

    state, info = env.reset()
    n = 2 * agent.config['batch_size']
    for _ in range(n):
        action = agent.act(state)
        assert action is not None

        next_state, reward, done, truncated, info = env.step(action)
        agent.record(state, action, reward, next_state, done, info)

        if done:
            state, info = env.reset()
        else:
            state = next_state

    assert (len(agent.memory) == n or
            len(agent.memory) == agent.config['memory_capacity'])

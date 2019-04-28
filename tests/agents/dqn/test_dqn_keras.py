import gym
import pytest

keras = pytest.importorskip("keras")


@pytest.mark.skip(reason="only the pytorch DQNAgent is tested to free memory")
def test_cartpole():
    from rl_agents.agents.deep_q_network.keras import DQNAgent

    env = gym.make('CartPole-v0')
    agent = DQNAgent(env, config=None)

    state = env.reset()
    n = 2 * agent.config['batch_size']
    for _ in range(n):
        action = agent.act(state)
        assert action is not None

        next_state, reward, done, info = env.step(action)
        agent.record(state, action, reward, next_state, done)

        if done:
            state = env.reset()
        else:
            state = next_state

    assert (len(agent.memory) == n or
            len(agent.memory) == agent.config['memory_capacity'])

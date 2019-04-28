import gym

from rl_agents.agents.common.abstract import AbstractAgent
from rl_agents.trainer.evaluation import Evaluation


def test_evaluation(tmpdir):
    env = gym.make('CartPole-v0')
    agent = RandomAgent(env)
    evaluation = Evaluation(env,
                            agent,
                            directory=tmpdir.strpath,
                            num_episodes=3,
                            display_env=False,
                            display_agent=False,
                            display_rewards=False)
    evaluation.monitor._monitor = True  # TODO: dirty fix until merge of https://github.com/openai/gym/pull/1362
    evaluation.train()
    artifacts = tmpdir.listdir()
    assert any(['manifest' in file.basename for file in artifacts])
    assert any(['metadata' in file.basename for file in artifacts])
    assert any(['stats' in file.basename for file in artifacts])


class RandomAgent(AbstractAgent):

    def __init__(self, env):
        super(RandomAgent, self).__init__()
        self.env = env

    def record(self, state, action, reward, next_state, done, info):
        pass

    def act(self, state):
        return self.env.action_space.sample()

    def reset(self):
        pass

    def seed(self, seed=None):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

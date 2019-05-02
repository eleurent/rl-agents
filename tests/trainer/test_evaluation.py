from pathlib import Path
import gym

from rl_agents.agents.simple.random import RandomUniformAgent
from rl_agents.trainer.evaluation import Evaluation


def test_evaluation(tmpdir):
    env = gym.make('CartPole-v0')
    agent = RandomUniformAgent(env)
    evaluation = Evaluation(env,
                            agent,
                            directory=tmpdir,
                            num_episodes=3,
                            display_env=False,
                            display_agent=False,
                            display_rewards=False)
    evaluation.train()
    assert any(['manifest' in file.name for file in evaluation.run_directory.iterdir()])
    assert any(['metadata' in file.name for file in evaluation.run_directory.iterdir()])
    assert any(['stats' in file.name for file in evaluation.run_directory.iterdir()])

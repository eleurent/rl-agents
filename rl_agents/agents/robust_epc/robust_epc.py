
from rl_agents.agents.common.abstract import AbstractAgent


class RobustEPCAgent(AbstractAgent):
    """
        Cross-Entropy Method planner.
        The environment is copied and used as an oracle model to sample trajectories.
    """
    def __init__(self, env, config):
        super().__init__(config)
        self.env = env

    @classmethod
    def default_config(cls):
        return dict(gamma=0.9)

    # def step(self, action):
    #
    #
    # def estimate(self):
    #
    #
    # def plan(self):
    #     pass
    #
    # def estimate(self):
    #     pass





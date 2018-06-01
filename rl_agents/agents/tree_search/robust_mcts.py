import numpy as np

from rl_agents.agents.abstract import AbstractAgent
from rl_agents.agents.tree_search.mcts import MCTSAgent


class RobustMCTSAgent(AbstractAgent):
    def __init__(self,
                 env,
                 models,
                 config=None,
                 prior_policy=None,
                 rollout_policy=None,):
        """
            A new MCTS agent with multiple environment models.
        :param env: The true environment
        :param models: A list of env preprocessors that represent possible transition models
        :param config: The agent configuration
        :param prior_policy: The prior distribution over actions given a state
        :param rollout_policy: The distribution over actions used when evaluating leaves
        """
        super(RobustMCTSAgent, self).__init__(config)
        self.agents = [MCTSAgent(env, self.config, prior_policy, rollout_policy, env_preprocessor=model)
                       for model in models]
        self.__env = env

    @property
    def env(self):
        return self.__env

    @env.setter
    def env(self, env):
        self.__env = env
        for agent in self.agents:
            agent.env = env

    def plan(self, state):
        for agent in self.agents:
            agent.plan(state)

        min_action_values = {k: np.inf for k in self.env.ACTIONS.keys()}
        for agent in self.agents:
            min_action_values = {k: min(v, agent.mcts.root.children[k].value)
                                 for k, v in min_action_values.items()
                                 if k in agent.mcts.root.children}
        action = max(min_action_values.keys(), key=(lambda key: min_action_values[key]))
        for agent in self.agents:
            agent.previous_action = action

        return [action]

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def seed(self, seed=None):
        for agent in self.agents:
            seeds = agent.seed(seed)
            seed = seeds[0]
        return seed

    def record(self, state, action, reward, next_state, done):
        raise NotImplementedError()

    def act(self, state):
        return self.plan(state)[0]

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()

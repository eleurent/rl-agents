import numpy as np

from rl_agents.agents.common.abstract import AbstractAgent
from rl_agents.agents.common.factory import agent_factory
from rl_agents.agents.tree_search.mcts import MCTSAgent


class MCTSWithPriorPolicyAgent(MCTSAgent):
    """
        An MCTS agent that leverages another StochasticAgent as a prior and rollout policy to guide its tree
        expansions and leaf evaluations.
    """

    def __init__(self,
                 env,
                 config=None):
        """
        :param env: The environment
        :param config: The agent configuration. It has to contains the field:
                       - prior_agent is the config used to create the agent, whose class is specified in
                       its __class__ field.
        """
        super(AbstractAgent, self).__init__(config)
        self.prior_agent = agent_factory(env, config['prior_agent'])
        #  Load the prior agent from a file, if one is set
        if 'model_save' in config['prior_agent']:
            self.prior_agent.load(config['prior_agent']['model_save'])
        super(MCTSWithPriorPolicyAgent, self).__init__(
            env,
            self.config)
        self.planner.prior_policy = self.agent_policy_available
        self.planner.rollout_policy = self.agent_policy_available

    @classmethod
    def default_config(cls):
        """
            Use the PyTorch implementation of a DQN Agent as default prior agent
        :return: the default MCTSWithPriorPolicyAgent config
        """
        mcts_config = super(MCTSWithPriorPolicyAgent, cls).default_config()
        mcts_config.update({"prior_agent": {
                                "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
                                "exploration": {"method": "Boltzmann"}
        }})
        return mcts_config

    def agent_policy(self, state, observation):
        # Reset prior agent environment
        self.prior_agent.env = state
        # Trigger the computation of action distribution
        self.prior_agent.act(observation)
        distribution = self.prior_agent.action_distribution(observation)
        return list(distribution.keys()), list(distribution.values())

    def agent_policy_available(self, state, observation):
        actions, probs = self.agent_policy(state, observation)
        if hasattr(state, 'get_available_actions'):
            available_actions = state.get_available_actions()
            probs = np.array([probs[actions.index(a)] for a in available_actions])
            probs /= np.sum(probs)
            actions = available_actions
        return actions, probs

    def record(self, state, action, reward, next_state, done, info):
        raise NotImplementedError()

    def save(self, filename):
        return self.prior_agent.save(filename)

    def load(self, filename):
        return self.prior_agent.load(filename)

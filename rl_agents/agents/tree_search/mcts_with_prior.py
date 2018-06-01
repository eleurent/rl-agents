from rl_agents.agents.abstract import AbstractAgent
from rl_agents.agents.common import agent_factory
from rl_agents.agents.tree_search.mcts import MCTSAgent


class MCTSWithPriorPolicyAgent(MCTSAgent):
    """
        An MCTS agent that leverages another StochasticAgent as a prior and rollout policy to guide its tree
        expansions and leaf evaluations.
    """

    def __init__(self,
                 env,
                 config=None,
                 env_preprocessor=None):
        """
        :param env: The environment
        :param config: The agent configuration. It has to contains the fields:
                       - prior_agent is the config used to create the agent, whose class is specified in
                       its __class__ field.
        :param env_preprocessor: a preprocessor function to apply to the environment before planning
        """
        super(AbstractAgent, self).__init__(config)
        self.prior_agent = agent_factory(env, config['prior_agent'])
        #  Load the prior agent from a file, if one is set
        if 'model_save' in config['prior_agent']:
            self.prior_agent.load(config['prior_agent']['model_save'])
        super(MCTSWithPriorPolicyAgent, self).__init__(
            env,
            self.config,
            prior_policy=self.agent_policy,
            rollout_policy=self.agent_policy,
            env_preprocessor=env_preprocessor)

    @classmethod
    def default_config(cls):
        """
            Use the PyTorch implementation of a DQN Agent as default prior agent
        :return: the default MCTSWithPriorPolicyAgent config
        """
        mcts_config = super(MCTSWithPriorPolicyAgent, cls).default_config()
        mcts_config.update({"iterations": 10})
        mcts_config.update({"prior_agent": {
                                "__class__": "<class 'rl_agents.agents.dqn.pytorch.DQNAgent'>",
                                "exploration": {"method": "Boltzmann"}
        }})
        return mcts_config

    def agent_policy(self, state, observation):
        distribution = self.prior_agent.action_distribution(observation)
        return list(distribution.keys()), list(distribution.values())

    def record(self, state, action, reward, next_state, done):
        raise NotImplementedError()

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        self.prior_agent.load(filename)

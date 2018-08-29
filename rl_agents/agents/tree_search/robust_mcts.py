import numpy as np
from gym import logger

from rl_agents.agents.abstract import AbstractAgent
from rl_agents.agents.common import load_agent, preprocess_env
from rl_agents.agents.tree_search.mcts import MCTSAgent, MCTS, Node


class DiscreteRobustMCTSAgent(MCTSAgent):
    def __init__(self,
                 env,
                 config=None):
        super(DiscreteRobustMCTSAgent, self).__init__(env, config)
        self.__env = env
        self.mcts = RobustMCTS(self.mcts.prior_policy, self.mcts.rollout_policy, self.config)

    @classmethod
    def default_config(cls):
        config = super(DiscreteRobustMCTSAgent, cls).default_config()
        config.update(dict(envs_preprocessors=[]))
        return config

    def record(self, state, action, reward, next_state, done):
        raise NotImplementedError()

    def plan(self, observation):
        envs = [preprocess_env(self.__env, preprocessors) for preprocessors in self.config["envs_preprocessors"]]
        self.env = JointEnv(envs)
        return super(DiscreteRobustMCTSAgent, self).plan(observation)

    def act(self, state):
        return self.plan(state)[0]

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()


class JointEnv(object):
    def __init__(self, envs):
        self.joint_state = envs

    def step(self, action):
        transitions = [state.step(action) for state in self.joint_state]
        observations, rewards, terminals, info = zip(*transitions)
        return observations, np.array(rewards), np.array(terminals), info

    @property
    def action_space(self):
        return self.joint_state[0].action_space

    def get_available_actions(self):
        return list(set().union(*[s.get_available_actions() if hasattr(s, "get_available_actions") else []
                                  for s in self.joint_state]))


class RobustMCTS(MCTS):
    def __init__(self, prior_policy, rollout_policy, config=None):
        super(RobustMCTS, self).__init__(prior_policy, rollout_policy, config)
        self.root = RobustNode(parent=None, mcts=self)


class RobustNode(Node):
    def get_value(self):
        return np.min(self.value)


class IntervalRobustMCTS(AbstractAgent):
    def __init__(self, env, config=None):
        super(IntervalRobustMCTS, self).__init__(config)
        self.env = env
        self.sub_agent = load_agent(self.config['sub_agent_path'], env)

    @classmethod
    def default_config(cls):
        return dict(sub_agent_path="",
                    env_preprocessors=[],
                    enable_robust_planning=True)

    def act(self, observation):
        return self.plan(observation)[0]

    def plan(self, observation):
        if self.config["enable_robust_planning"]:
            self.sub_agent.env = preprocess_env(self.env, self.config["env_preprocessors"])
        return self.sub_agent.plan(observation)

    def reset(self):
        return self.sub_agent.reset()

    def seed(self, seed=None):
        return self.sub_agent.seed(seed)

    def save(self, filename):
        return self.sub_agent.save(filename)

    def load(self, filename):
        return self.sub_agent.load(filename)

    def record(self, state, action, reward, next_state, done):
        return self.sub_agent.record(state, action, reward, next_state, done)


class OneStepRobustMCTS(AbstractAgent):
    def __init__(self,
                 env,
                 config=None):
        """
            A new MCTS agent with multiple environment models.
        :param env: The true environment
        :param config: The agent configuration.
                       It should include an "agents" key giving a list of paths to several MCTSAgent configurations.
                       This class could be modified to directly load configuration dictionaries instead of agents
                       config files.
        """
        super(OneStepRobustMCTS, self).__init__(config)
        self.agents = [load_agent(agent_config_path, env) for agent_config_path in self.config["agents"]]
        self.__env = env

    @classmethod
    def default_config(cls):
        return dict(agents=[])

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

        min_action_values = {k: np.inf for k in range(self.env.action_space.n)}
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

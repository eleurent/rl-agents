import numpy as np

from rl_agents.agents.abstract import AbstractAgent
from rl_agents.agents.common import load_agent, preprocess_env
from rl_agents.agents.tree_search.mcts import MCTSAgent, MCTS, MCTSNode


class DiscreteRobustPlannerAgent(MCTSAgent):
    def __init__(self,
                 env,
                 config=None):
        self.__env = env
        super(DiscreteRobustPlannerAgent, self).__init__(env, config)

    def make_planner(self):
        return RobustMCTS(self.planner.prior_policy, self.planner.rollout_policy, self.config)

    @classmethod
    def default_config(cls):
        config = super(DiscreteRobustPlannerAgent, cls).default_config()
        config.update(dict(envs_preprocessors=[]))
        return config

    def plan(self, observation):
        envs = [preprocess_env(self.__env, preprocessors) for preprocessors in self.config["envs_preprocessors"]]
        self.env = JointEnv(envs)
        return super(DiscreteRobustPlannerAgent, self).plan(observation)


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
        return list(set().union(*[s.get_available_actions() if hasattr(s, "get_available_actions")
                                  else range(s.action_space.n)
                                  for s in self.joint_state]))


class RobustMCTS(MCTS):
    def make_root(self):
        self.root = RobustMCTSNode(parent=None, mcts=self)


class RobustMCTSNode(MCTSNode):
    def get_value(self):
        return np.min(self.value)


class IntervalRobustPlannerAgent(AbstractAgent):
    def __init__(self, env, config=None):
        super(IntervalRobustPlannerAgent, self).__init__(config)
        self.env = env
        self.sub_agent = load_agent(self.config['sub_agent_path'], env)

    @classmethod
    def default_config(cls):
        return dict(sub_agent_path="",
                    env_preprocessors=[])

    def act(self, observation):
        return self.plan(observation)[0]

    def plan(self, observation):
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

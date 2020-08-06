import numpy as np

from rl_agents.agents.common.abstract import AbstractAgent
from rl_agents.agents.common.factory import load_agent, preprocess_env
from rl_agents.agents.tree_search.deterministic import DeterministicPlannerAgent, OptimisticDeterministicPlanner, \
    DeterministicNode


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


class DiscreteRobustPlanner(OptimisticDeterministicPlanner):
    def reset(self):
        self.root = RobustNode(parent=None, planner=self)
        self.leaves = [self.root]


class RobustNode(DeterministicNode):
    def get_value_lower_bound(self):
        return np.min(self.value_lower)

    def get_value_upper_bound(self):
        return np.min(self.value_upper)


class DiscreteRobustPlannerAgent(DeterministicPlannerAgent):
    PLANNER_TYPE = DiscreteRobustPlanner

    def __init__(self,
                 env,
                 config=None):
        self.true_env = env
        super().__init__(env, config)

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update(dict(models=[]))
        return config

    def plan(self, observation):
        envs = [preprocess_env(self.true_env, preprocessors) for preprocessors in self.config["models"]]
        self.env = JointEnv(envs)
        return super().plan(observation)


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

    def get_plan(self):
        return self.sub_agent.planner.get_plan()

    def reset(self):
        return self.sub_agent.reset()

    def seed(self, seed=None):
        return self.sub_agent.seed(seed)

    def save(self, filename):
        return self.sub_agent.save(filename)

    def load(self, filename):
        return self.sub_agent.load(filename)

    def record(self, state, action, reward, next_state, done, info):
        return self.sub_agent.record(state, action, reward, next_state, done, info)

import copy

import gym
from gym import logger
import numpy as np
from gym.utils import seeding

from rl_agents.agents.abstract import AbstractAgent
from rl_agents.agents.common import preprocess_env, safe_deepcopy_env
from rl_agents.agents.tree_search.tree import Node
from rl_agents.configuration import Configurable


class DeterministicPlannerAgent(AbstractAgent):
    """
        An agent that performs optimistic planning in deterministic MDPs.
    """

    def __init__(self,
                 env,
                 config=None):
        super(DeterministicPlannerAgent, self).__init__(config)
        self.env = env
        self.olop = OptimisticDeterministicPlanner(env, self.config)
        self.previous_action = None

    @classmethod
    def default_config(cls):
        return dict(env_preprocessors=[])

    def plan(self, observation):
        """
            Plan an optimal sequence of actions.

            Start by updating the previously found tree with the last action performed.

        :param observation: the current state
        :return: the list of actions
        """
        self.olop.step(self.previous_action)
        env = preprocess_env(self.env, self.config["env_preprocessors"])
        actions = self.olop.plan(state=env)

        self.previous_action = actions[0]
        return actions

    def seed(self, seed=None):
        return [seed]

    def reset(self):
        self.olop.step_by_reset()

    def record(self, state, action, reward, next_state, done):
        raise NotImplementedError()

    def act(self, state):
        return self.plan(state)[0]

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()


class OptimisticDeterministicPlanner(Configurable):
    """
       An implementation of Open Loop Optimistic Planning.
    """
    def __init__(self, env, config=None):
        super(OptimisticDeterministicPlanner, self).__init__(config)
        self.root = DeterministicNode(None, planner=self)
        self.env = env
        self.np_random = None
        self.seed()

    @classmethod
    def default_config(cls):
        return dict(budget=100,
                    gamma=0.7,
                    step_strategy="reset")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def run(self, leaves):
        """
            Run an OptimisticDeterministicPlanner episode
        :param leaves: search tree leaves
        """
        leaf_to_expand = max(leaves, key=lambda n: n.value_upper_bound)
        leaf_to_expand.expand(leaves)
        self.root.backup_values()

    def plan(self, state):
        """
            Plan an optimal sequence of actions

        :param state: the initial environment state
        :return: the list of actions
        """
        self.root.state = state
        leaves = [self.root]
        for _ in np.arange(self.config["budget"] // self.env.action_space.n):
            self.run(leaves)

        # Return best action, tie randomly
        actions = list(self.root.children.keys())
        a = self.root.random_argmax([self.root.children[a].value for a in actions])
        return [actions[a]]

    def step(self, action):
        """
            Update the tree when the agent performs an action

        :param action: the chosen action from the root node
        """
        if self.config["step_strategy"] == "reset":
            self.step_by_reset()
        else:
            gym.logger.warn("Unknown step strategy: {}".format(self.config["step_strategy"]))
            self.step_by_reset()

    def step_by_reset(self):
        self.root = DeterministicNode(None, planner=self)


class DeterministicNode(Node):
    def __init__(self, parent, planner, state=None, depth=0):
        super(DeterministicNode, self).__init__(parent, planner)
        self.state = state
        self.depth = depth
        self.value_upper_bound = 0
        self.terminal = False

    def expand(self, leaves):
        if self.state is None:
            raise Exception("The state should be set before expanding a node")
        try:
            actions = self.state.get_available_actions()
        except AttributeError:
            actions = range(self.state.action_space.n)
        for action in actions:
            self.children[action] = type(self)(self,
                                               self.planner,
                                               state=safe_deepcopy_env(self.state),
                                               depth=self.depth + 1)
            _, reward, done, _ = self.children[action].state.step(action)
            reward = reward if not done else 0
            self.children[action].update(reward)

        leaves.remove(self)
        leaves.extend(self.children.values())

    def update(self, reward):
        if not 0 <= reward <= 1:
            raise ValueError("This planner assumes that all rewards are normalized in [0, 1]")
        gamma = self.planner.config["gamma"]
        self.value = self.parent.value + (gamma ** (self.depth - 1)) * reward
        self.value_upper_bound = self.value + (gamma ** self.depth) / (1 - gamma)

    def backup_values(self):
        if self.children:
            self.value = np.amax([child.backup_values() for child in self.children.values()])
            self.value_upper_bound = 0  # should be backed-up as well, but not used anyway.
        return self.value



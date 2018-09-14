import gym
import numpy as np

from rl_agents.agents.common import safe_deepcopy_env
from rl_agents.agents.tree_search.abstract import Node, AbstractTreeSearchAgent, AbstractPlanner


class DeterministicPlannerAgent(AbstractTreeSearchAgent):
    """
        An agent that performs optimistic planning in deterministic MDPs.
    """
    def make_planner(self):
        return OptimisticDeterministicPlanner(self.config)


class OptimisticDeterministicPlanner(AbstractPlanner):
    """
       An implementation of Open Loop Optimistic Planning.
    """
    def make_root(self):
        return DeterministicNode(None, planner=self)

    def run(self, leaves):
        """
            Run an OptimisticDeterministicPlanner episode
        :param leaves: search tree leaves
        """
        leaf_to_expand = max(leaves, key=lambda n: n.value_upper_bound)
        leaf_to_expand.expand(leaves)
        self.root.backup_values()

    def plan(self, state, observation):
        self.root.state = state
        leaves = [self.root]
        for _ in np.arange(self.config["budget"] // state.action_space.n):
            self.run(leaves)

        return self.get_plan()

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

    def selection_rule(self):
        if not self.children:
            return None
        actions = list(self.children.keys())
        index = self.random_argmax([self.children[a].value for a in actions])
        return actions[index]

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

import numpy as np
import logging
from rl_agents.agents.common.factory import safe_deepcopy_env
from rl_agents.agents.tree_search.abstract import Node, AbstractTreeSearchAgent, AbstractPlanner

logger = logging.getLogger(__name__)


class DeterministicNode(Node):
    def __init__(self, parent, planner, state=None, depth=0):
        super().__init__(parent, planner)
        self.state = state
        self.observation = None
        self.depth = depth
        self.reward = 0
        self.value_upper = 0
        self.value_lower = 0
        self.count = 1
        self.done = False

    def selection_rule(self):
        if not self.children:
            return None
        actions = list(self.children.keys())
        index = self.random_argmax([self.children[a].get_value_lower_bound() for a in actions])
        return actions[index]

    def expand(self):
        self.planner.leaves.remove(self)
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
            observation, reward, done, _ = self.planner.step(self.children[action].state, action)
            self.planner.leaves.append(self.children[action])
            self.children[action].update(reward, done, observation)

    def update(self, reward, done, observation=None):
        if not np.all(0 <= reward) or not np.all(reward <= 1):
            raise ValueError("This planner assumes that all rewards are normalized in [0, 1]")
        gamma = self.planner.config["gamma"]
        self.reward = reward
        self.observation = observation
        self.done = done
        self.value_lower = self.parent.value_lower + (gamma ** (self.depth - 1)) * reward
        self.value_upper = self.value_lower + (gamma ** self.depth) / (1 - gamma)
        if done:
            self.value_lower = self.value_upper = self.value_lower + \
                self.planner.config["terminal_reward"] * (gamma ** self.depth) / (1 - gamma)

        for node in self.sequence():
            node.count += 1

    def backup_values(self):
        if self.children:
            backup_children = [child.backup_values() for child in self.children.values()]
            self.value_lower = np.amax([b[0] for b in backup_children])
            self.value_upper = np.amax([b[1] for b in backup_children])
        return self.get_value_lower_bound(), self.get_value_upper_bound()

    def backup_to_root(self):
        if self.children:
            self.value_lower = np.amax([child.value_lower for child in self.children.values()])
            self.value_upper = np.amax([child.value_upper for child in self.children.values()])
            if self.parent:
                self.parent.backup_to_root()

    def get_value_lower_bound(self):
        return self.value_lower

    def get_value_upper_bound(self):
        return self.value_upper

    def get_value(self) -> float:
        return self.value_upper


class OptimisticDeterministicPlanner(AbstractPlanner):
    NODE_TYPE = DeterministicNode

    """
       An implementation of Open Loop Optimistic Planning.
    """
    def __init__(self, env, config=None):
        super(OptimisticDeterministicPlanner, self).__init__(config)
        self.env = env
        self.leaves = None

    def reset(self):
        self.root = self.NODE_TYPE(None, planner=self)
        self.leaves = [self.root]

    def run(self):
        """
            Run an OptimisticDeterministicPlanner episode
        """
        leaf_to_expand = max(self.leaves, key=lambda n: n.get_value_upper_bound())
        if leaf_to_expand.done:
            logger.warning("Expanding a terminal state")
        leaf_to_expand.expand()
        leaf_to_expand.backup_to_root()

    def plan(self, state, observation):
        self.root.state = state
        for epoch in np.arange(self.config["budget"] // state.action_space.n):
            logger.debug("Expansion {}/{}".format(epoch + 1, self.config["budget"] // state.action_space.n))
            self.run()

        return self.get_plan()

    def step_by_subtree(self, action):
        super(OptimisticDeterministicPlanner, self).step_by_subtree(action)
        if not self.root.children:
            self.leaves = [self.root]
        #  v0 = r0 + g r1 + g^2 r2 +... and v1 = r1 + g r2 + ... = (v0-r0)/g
        for leaf in self.leaves:
            leaf.value_lower = (leaf.value_lower - self.root.reward) / self.config["gamma"]
            leaf.value_upper_bound = (leaf.value_upper_bound - self.root.reward) / self.config["gamma"]
        self.root.backup_values()


class DeterministicPlannerAgent(AbstractTreeSearchAgent):
    """
        An agent that performs optimistic planning in deterministic MDPs.
    """
    PLANNER_TYPE = OptimisticDeterministicPlanner

import numpy as np
import logging
from rl_agents.agents.common.factory import safe_deepcopy_env
from rl_agents.agents.tree_search.abstract import Node, AbstractTreeSearchAgent, AbstractPlanner

logger = logging.getLogger(__name__)


class StateAwarePlannerAgent(AbstractTreeSearchAgent):
    """
        An agent that performs optimistic planning in deterministic MDPs.
    """
    def make_planner(self):
        return StateAwarePlanner(self.env, self.config)


class StateAwarePlanner(AbstractPlanner):
    """
       An implementation of State Aware Planning.
    """
    def __init__(self, env, config=None):
        super().__init__(config)
        self.env = env
        self.leaves = None

    def make_root(self):
        root = StateAwareNode(None, planner=self)
        self.leaves = [root]
        return root

    def run(self):
        """
            Run an OptimisticDeterministicPlanner episode
        """
        leaf_to_expand = max(self.leaves, key=lambda n: n.get_value_upper_bound())
        if not leaf_to_expand.done:
            leaf_to_expand.expand(self.leaves)

        leaf_to_expand.backup_to_root()

    def plan(self, state, observation):
        self.root.state = state
        self.root.observation = observation
        for _ in np.arange(self.config["budget"] // state.action_space.n):
            self.run()

        return self.get_plan()


class StateAwareNode(Node):
    def __init__(self, parent, planner, state=None, observation=None, depth=0):
        super().__init__(parent, planner)
        self.state = state
        self.observation = observation
        self.depth = depth
        self.reward = 0
        self.value = 0  # Sum of rewards along sequence
        self.future_value_upper_bound = 0  # Upper-bound on optimal future rewards
        self.count = 1
        self.done = False

    def selection_rule(self):
        if not self.children:
            return None
        actions = list(self.children.keys())
        index = self.random_argmax([self.children[a].get_value() for a in actions])
        return actions[index]

    def expand(self, leaves):
        leaves.remove(self)
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
            observation, reward, done, _ = self.children[action].state.step(action)
            leaves.append(self.children[action])
            self.children[action].update(reward, done, observation, leaves)



    def update(self, reward, done, observation, leaves):
        if not np.all(0 <= reward) or not np.all(reward <= 1):
            raise ValueError("This planner assumes that all rewards are normalized in [0, 1]")
        gamma = self.planner.config["gamma"]
        self.reward = reward
        self.observation = observation
        self.value = self.parent.value + (gamma ** (self.depth - 1)) * reward
        self.done = done
        if self.done:
            self.future_value_upper_bound = 0
        else:
            # Search the tree for similar states to decrease this optimistic UCB for future rewards:
            is_same_state_node = lambda node: str(node.observation) == str(self.observation)
            same_state_nodes = list(StateAwareNode.breadth_first_search(self.planner.root, None, is_same_state_node,
                                                                        condition_blocking=False))
            # Set min state UCB for all occurences
            future_value_upper_bound = min([1/(1 - gamma)] + [node.future_value_upper_bound for node, path in same_state_nodes])
            for node, _ in same_state_nodes:
                node.future_value_upper_bound = future_value_upper_bound

            # Pick the one with highest sequence-value, and kill the others
            best = max([n for n, _ in same_state_nodes], key=lambda n: n.get_value_upper_bound())
            for node, _ in same_state_nodes:
                if node is not best and not node.children and node in leaves:
                    leaves.remove(node)

    def backup_to_root(self):
        if self.parent:
            gamma = self.planner.config["gamma"]
            best_child = max(self.parent.children.values(), key=lambda child: child.get_value_upper_bound())

            # Here, we could overwrite a state value U_a^s that was previously transfered from elsewhere
            # by another value backed up from the children. This is probably fine.
            self.future_value_upper_bound = best_child.reward + gamma*best_child.future_value_upper_bound

            self.parent.backup_to_root()
            self.count += 1

    def get_value_upper_bound(self):
        return self.value + (self.planner.config["gamma"] ** self.depth) * self.future_value_upper_bound

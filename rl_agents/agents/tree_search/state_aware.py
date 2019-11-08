import numpy as np
import logging
from rl_agents.agents.tree_search.deterministic import DeterministicPlannerAgent, OptimisticDeterministicPlanner, \
    DeterministicNode

logger = logging.getLogger(__name__)


class StateAwarePlannerAgent(DeterministicPlannerAgent):
    """
        An agent that performs state-aware optimistic planning in deterministic MDPs.
    """
    def make_planner(self):
        return StateAwarePlanner(self.env, self.config)


class StateAwarePlanner(OptimisticDeterministicPlanner):
    """
       An implementation of State Aware Planning.
    """
    def __init__(self, env, config=None):
        super().__init__(env, config)
        self.state_nodes = {}  # Mapping of states to tree nodes that lead to this state
        self.state_values = {}  # Mapping of states to an upper confidence bound of the state-value

    def make_root(self):
        root = StateAwareNode(None, planner=self)
        self.leaves = [root]
        return root

    def run(self):
        leaf_to_expand = max(self.leaves, key=lambda n: n.get_value_upper_bound())
        if leaf_to_expand.done:
            logger.warning("Expanding a terminal state")
        leaf_to_expand.expand()
        leaf_to_expand.updated_nodes = leaf_to_expand.backup_to_root()
        print("{} updated nodes for state {} from path {}".format(
            leaf_to_expand.updated_nodes,
            leaf_to_expand.observation,
            list(leaf_to_expand.path())))

    def update_value(self, observation, value):
        """
            Update the upper-confidence-bound for the value of a state with a possibly tighter candidate

        :param observation: an observed state
        :param value: a candidate upper-confidence bound
        :return: whether or not the value was updated, and needs to be backed-up
        """
        if str(observation) not in self.state_values or value < self.state_values[str(observation)]:
            self.state_values[str(observation)] = value
            return True
        else:
            return False

    def plan(self, state, observation):
        # Initialize root
        self.root.observation = observation
        self.state_nodes[str(observation)] = [self.root]
        self.state_values[str(observation)] = 1 / (1 - self.config["gamma"])
        super().plan(state, observation)

        logger.debug("{} expansions".format(self.config["budget"] // state.action_space.n))
        logger.debug("{} states explored".format(len(self.state_nodes)))

        return self.get_plan()


class StateAwareNode(DeterministicNode):
    def __init__(self, parent, planner, state=None, depth=0):
        super().__init__(parent, planner, state, depth)
        self.observation = None  # Store observations

    def update(self, reward, done, observation=None):
        super().update(reward, done)
        self.observation = observation

        # Add to list of nodes with this observation
        if str(observation) not in self.planner.state_nodes:
            self.planner.state_nodes[str(observation)] = []
        self.planner.state_nodes[str(observation)].append(self)

        # Update the value of this state - no gain of information
        future_value_ucb = 1/(1 - self.planner.config["gamma"]) if not self.done else 0  # Default value
        self.planner.update_value(observation, future_value_ucb)  # Aggregate from other nodes

        # Among sequences that lead to this state, remove all suboptimal leaves
        state_leaves = [node for node in self.planner.state_nodes[str(observation)]
                        if not node.children and node in self.planner.leaves]
        best = max(state_leaves, key=lambda n: n.get_value_upper_bound())
        [self.planner.leaves.remove(node) for node in state_leaves if node is not best]

    def backup_to_root(self):
        if self.children:
            updated_nodes = 1
            best_child = max(self.children.values(), key=lambda child: child.get_value_upper_bound())
            updated = self.planner.update_value(self.observation, best_child.reward + self.planner.config["gamma"] *
                                                self.planner.state_values[str(best_child.observation)])
            if updated:
                if self.parent:  # Backup this node's parent first
                    updated_nodes += self.parent.backup_to_root()
                for node in self.planner.state_nodes[str(self.observation)]:  # And other updated nodes then
                    if node is not self and node.parent:
                        updated_nodes += node.parent.backup_to_root()
        return updated_nodes

    def get_value_upper_bound(self):
        return self.value + \
               (self.planner.config["gamma"] ** self.depth) * self.planner.state_values[str(self.observation)]

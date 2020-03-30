from collections import defaultdict

import logging
from rl_agents.agents.tree_search.deterministic import DeterministicPlannerAgent, OptimisticDeterministicPlanner, \
    DeterministicNode

logger = logging.getLogger(__name__)


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

        # Handle terminal states
        if self.done:
            self.planner.update_value(observation, 0)

    def prune(self):
        """
            Among sequences that lead to this state, check if one is better than this one.
            If so, remove this leaf.
        """
        if self.planner.config["prune_suboptimal_leaves"]:
            value_upper_bound = self.get_value_upper_bound()
            for node in self.planner.state_nodes[str(self.observation)]:
                if node is not self and \
                        node.get_value_upper_bound() >= value_upper_bound and \
                        node.depth >= self.depth and \
                        (node.children or node in self.planner.leaves):
                    self.planner.leaves.remove(self)
                    break

    def backup_to_root(self):
        gamma = self.planner.config["gamma"]
        updates_count = 0
        queue = [self]
        while queue:
            node = queue.pop(0)
            delta = 0
            # Bellman backup
            if node.children:
                best_child = max(node.children.values(), key=lambda child: child.get_value_upper_bound())
                backup = best_child.reward + gamma * self.planner.state_values[str(best_child.observation)]
                # Update state ucb with this new bound
                delta = self.planner.update_value(node.observation, backup)
                updates_count += 1

            # Should we propagate the update by backing-up the parents?
            for neighbour in self.planner.state_nodes[str(node.observation)]:
                if neighbour.parent and \
                        (neighbour is node or self.planner.config["backup_aggregated_nodes"]) and \
                        delta > self.planner.config["accuracy"] * (1 - gamma) * gamma ** (neighbour.depth - 1):
                    queue.append(neighbour.parent)
        return updates_count

    def get_value_upper_bound(self):
        return self.value_lower + \
               (self.planner.config["gamma"] ** self.depth) * self.planner.state_values[str(self.observation)]


class StateAwarePlanner(OptimisticDeterministicPlanner):
    NODE_TYPE = StateAwareNode
    """
       An implementation of State Aware Planning.
    """
    def __init__(self, env, config=None):
        super().__init__(env, config)

        # Mapping of states to tree nodes that lead to this state
        self.state_nodes = {}
        # Mapping of states to an upper confidence bound of the state-value
        self.state_values = defaultdict(lambda: 1 / (1 - self.config["gamma"]))

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update({
            "backup_aggregated_nodes": True,
            "prune_suboptimal_leaves": True,
            "accuracy": 0,
        })
        return cfg

    def run(self):
        leaf_to_expand = max(self.leaves, key=lambda n: n.get_value_upper_bound())
        if leaf_to_expand.done:
            logger.warning("Expanding a terminal state")
        leaf_to_expand.expand()
        leaf_to_expand.updates_count = leaf_to_expand.backup_to_root()
        logger.debug("{} updated nodes for state {} from path {}".format(
            leaf_to_expand.updates_count,
            leaf_to_expand.observation,
            list(leaf_to_expand.path())))
        for leaf in reversed(self.leaves.copy()):
            leaf.prune()

    def update_value(self, observation, value):
        """
            Update the upper-confidence-bound for the value of a state with a possibly tighter candidate

        :param observation: an observed state
        :param value: a candidate upper-confidence bound
        :return: the value difference
        """
        delta = self.state_values[str(observation)] - value
        if delta > 0:
            self.state_values[str(observation)] = value
        return delta

    def plan(self, state, observation):
        # Initialize root
        self.root.observation = observation
        self.state_nodes[str(observation)] = [self.root]
        self.state_values[str(observation)] = 1 / (1 - self.config["gamma"])
        super().plan(state, observation)

        logger.debug("{} expansions".format(self.config["budget"] // state.action_space.n))
        logger.debug("{} states explored".format(len(self.state_nodes)))

        return self.get_plan()


class StateAwarePlannerAgent(DeterministicPlannerAgent):
    """
        An agent that performs state-aware optimistic planning in deterministic MDPs.
    """
    PLANNER_TYPE = StateAwarePlanner

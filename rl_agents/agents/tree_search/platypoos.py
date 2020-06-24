import logging
from operator import attrgetter
import numpy as np

from rl_agents.agents.common.factory import safe_deepcopy_env
from rl_agents.agents.tree_search.abstract import Node, AbstractTreeSearchAgent, AbstractPlanner

logger = logging.getLogger(__name__)


class PlaTyPOOS(AbstractPlanner):
    """
       An implementation of "Planning with y Plus an Online Optimization Strategy".
       Reference: Scale-free adaptive planning for deterministic dynamics & discounted rewards (2019).
    """
    def __init__(self, env, config=None):
        super(PlaTyPOOS, self).__init__(config)
        self.env = env
        self.candidates = {}
        self.openings = 0

        if "horizon" not in self.config:
            expansion_budget = self.config["budget"] / env.action_space.n
            self.config["horizon"] = int(np.floor(expansion_budget /
                                                  (2 * (np.log2(expansion_budget) + 1)**2)))

    def reset(self):
        self.root = PlaTyPOOSNode(parent=None, planner=self, state=None)

    def explore(self, depth, current_layer):
        """
            Explore the nodes at the current depth
        :param int depth: current depth
        :param list current_layer: a list of nodes at depth h
        """
        # Sort nodes by values
        current_layer = sorted(current_layer, key=attrgetter('value'), reverse=True)

        # Select nodes to expand
        h, h_max, gamma = depth, self.config["horizon"], self.config["gamma"]
        p_top = max(int(np.floor(np.log2(h_max / np.ceil(h ** 2 * gamma ** (2 * h))))), 0)
        to_expand = []
        for p in range(p_top, -1, -1):
            nodes_count = int(np.floor(h_max / h * np.ceil(h * 2 ** p * gamma ** (2 * h))))
            evaluations = int(np.ceil(h * 2 ** p * gamma ** (2 * h)))
            min_visits = int(np.ceil((h - 1) * 2 ** p * gamma ** (2 * (h - 1))))

            # Pick first nodes with enough visits for evaluation
            for node in current_layer:
                if node.count > min_visits and not node.to_expand:
                    node.to_expand = True
                    to_expand.append((node, evaluations, p))
                if len(to_expand) >= nodes_count:
                    break

        # Expand selected nodes
        next_layer = []
        for node, evaluations, p in to_expand:
            node.expand(next_layer, evaluations)

            # Keep track of best nodes for cross-validation
            if p not in self.candidates or node.value > self.candidates[p].value:
                self.candidates[p] = node

        return next_layer

    def cross_validate(self):
        """
            Cross-validate the candidate action sequences with highest values
        """
        h_max, gamma = self.config["horizon"], self.config["gamma"]
        for node in self.candidates.values():
            while node:
                evaluations = int(
                    np.floor((node.depth + 1) * 5 * h_max * gamma ** (2 * node.depth) * (1 - gamma ** 2) ** 2))
                node.expand([], evaluations)
                node = node.parent

    def get_plan(self):
        """
            Get the optimal action sequence by following the best candidate up to the root.
        :return: the list of actions
        """
        actions = []
        candidate = max(self.candidates.values(), key=attrgetter("value"))
        while candidate.parent is not None:
            actions.insert(0, [a for a, node in candidate.parent.children.items() if node == candidate][0])
            candidate = candidate.parent
        return actions

    def plan(self, state, observation):
        # Initialization: expand the root
        current_layer, self.candidates, self.openings = [], {}, 0
        self.root.state = state
        self.root.expand(current_layer, self.config["horizon"])
        # Exploration and cross-validation
        for h in range(1, self.config["horizon"]):
            current_layer = self.explore(h, current_layer)
        self.cross_validate()
        logger.info("Total number of openings: {}".format(self.openings))
        return self.get_plan()


class PlaTyPOOSNode(Node):
    STOP_ON_ANY_TERMINAL_STATE = True

    def __init__(self, parent, planner, state, depth=0):
        super(PlaTyPOOSNode, self).__init__(parent, planner)

        self.state = state
        """ Environment state associated with the node."""

        self.depth = depth
        """ Node depth."""

        self.cumulative_reward = 0
        """ Sum of all rewards received at this node."""

        self.done = False
        """ Is this node a terminal node?"""

        self.to_expand = False

    def update(self, reward, done):
        """
            Update the node value given a new transition from oracle
        :param reward: the reward received
        :param done: is the state terminal
        """
        self.cumulative_reward += reward
        self.count += 1
        self.value = self.parent.value + self.planner.config["gamma"] ** (self.depth - 1) * (
                    self.cumulative_reward / self.count)
        self.done = done

    def expand(self, next_layer, count=1):
        """
            Expand the node by querying the oracle model for every possible action
        :param next_layer: list of nodes at the next depth, to be updated with new children nodes
        :param count: number of times each transition must be evaluated
        """
        if self.state is None:
            raise Exception("The state should be set before expanding a node")
        try:
            actions = self.state.get_available_actions()
        except AttributeError:
            actions = range(1, self.state.action_space.n)

        self.planner.openings += count

        if self.done and PlaTyPOOSNode.STOP_ON_ANY_TERMINAL_STATE:
            return

        for _ in range(count):
            for action in actions:
                state = safe_deepcopy_env(self.state)
                state.seed(self.planner.np_random.randint(2**30))
                _, reward, done, _ = state.step(action)

                if action not in self.children:
                    self.children[action] = type(self)(self,
                                                       self.planner,
                                                       state,
                                                       depth=self.depth + 1)
                    next_layer.append(self.children[action])

                self.children[action].update(reward, done)

    def get_value(self):
        if self.done:
            return self.value
        return self.value + self.planner.config["gamma"] ** self.depth / (1 - self.planner.config["gamma"])

    def selection_rule(self):
        """
            Select the subtree containing the best candidate.
            Or raise ValueError if the best candidate is not a descendant of this node
        :return: the action to perform in this node
        """
        candidate = max(self.planner.candidates.values(), key=attrgetter("value"))
        while candidate.parent and candidate.parent is not self:
            candidate = candidate.parent
        if not candidate.parent:
            raise ValueError("Best candidate is not a descendant of this node")
        return [a for a, node in candidate.parent.children.items() if node == candidate][0]


class PlaTyPOOSAgent(AbstractTreeSearchAgent):
    """
        An agent that uses PlaTyPOOS to plan a sequence of actions in an MDP.
    """
    PLANNER_TYPE = PlaTyPOOS
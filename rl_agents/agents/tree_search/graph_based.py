import operator
from collections import defaultdict

import numpy as np
import logging
from rl_agents.agents.common.factory import safe_deepcopy_env
from rl_agents.agents.tree_search.abstract import Node, AbstractTreeSearchAgent, AbstractPlanner
from rl_agents.agents.tree_search.mdp_gape import DecisionNode
from rl_agents.utils import kl_upper_bound

logger = logging.getLogger(__name__)


class GraphBasedPlannerAgent(AbstractTreeSearchAgent):
    def make_planner(self):
        return GraphBasedPlanner(self.env, self.config)

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update({
            "sampling_timeout": 100
        })
        return cfg


class GraphBasedPlanner(AbstractPlanner):
    def __init__(self, env, config=None):
        super().__init__(config)
        self.env = env
        self.nodes = {}

    def make_root(self):
        root = GraphNode(planner=self, state=None, observation=None)
        return root

    def run(self, observation):
        node = self.nodes[str(observation)]
        for k in range(self.config["sampling_timeout"]):
            if not node.children:
                self.expand(node)
                node.partial_value_iteration()
                break
            else:
                optimistic_action = node.sampling_rule()
                node = node.children[optimistic_action]
        else:
            logger.info("The optimistic sampling strategy could not find a sink. We probably found an optimal loop.")

    def expand(self, node):
        try:
            actions = node.state.get_available_actions()
        except AttributeError:
            actions = range(node.state.action_space.n)
        for action in actions:
            next_state = safe_deepcopy_env(node.state)
            next_observation, reward, done, _ = next_state.step(action)
            # Add new state node
            if str(next_observation) not in self.nodes:
                self.nodes[str(next_observation)] = GraphNode(self, next_state, next_observation)
            node.rewards[action] = reward
            node.children[action] = self.nodes[str(next_observation)]
            self.nodes[str(next_observation)].parents.append(node)

    def plan(self, state, observation):
        if str(observation) not in self.nodes:
            self.root = self.nodes[str(observation)] = GraphNode(self, state, observation)
        for epoch in np.arange(self.config["budget"] // state.action_space.n):
            logger.debug("Expansion {}/{}".format(epoch + 1, self.config["budget"] // state.action_space.n))
            self.run(observation)

        return self.get_plan()

    def get_plan(self):
        node = self.root
        actions = []
        for _ in range(self.config["sampling_timeout"]):
            if not node.children:
                break
            action = node.selection_rule()
            actions.append(action)
            node = node.children[action]
        return actions


class GraphNode(Node):
    def __init__(self, planner, state, observation):
        super().__init__(parent=None, planner=planner)
        self.state = state
        self.observation = observation
        self.value_lower = 0
        self.value_upper = 1 / (1 - self.planner.config["gamma"])
        self.rewards = {}
        self.parents = []
        self.updates_count = 0

    def sampling_rule(self):
        """
            Optimistic action sampling
        """
        q_values_upper_bound = self.backup("value_upper")
        return max(q_values_upper_bound.items(), key=operator.itemgetter(1))[0]

    def selection_rule(self):
        """
            Conservative action selection
        """
        action_values_bound = self.backup("value_lower")
        return max(action_values_bound.items(), key=operator.itemgetter(1))[0]

    def backup(self, field):
        gamma = self.planner.config["gamma"]
        return {action: self.rewards[action] + gamma * getattr(self.children[action], field)
                for action in self.children.keys()}

    def get_value(self):
        return self.value_lower

    def get_obs_visits(self):
        visits = defaultdict(int)
        updates = defaultdict(int)
        for obs in self.planner.nodes.keys():
            visits[obs] += 1
            updates[obs] += self.planner.nodes[obs].updates_count
        return visits, updates

    def get_trajectories(self, full_trajectories=True, include_leaves=True):
        return []

    def partial_value_iteration(self, eps=1e-2):
        queue = [self]
        while queue:
            self.updates_count += 1
            node = queue.pop(0)
            delta = 0
            for field in ["value_lower", "value_upper"]:
                action_value_bound = node.backup(field)
                state_value_bound = np.amax(list(action_value_bound.values()))
                delta = max(delta, abs(getattr(node, field) - state_value_bound))
                setattr(node, field, state_value_bound)
            if delta > eps:
                queue.extend(node.parents)

    def __str__(self):
        return "{} (L:{:.2f}, U:{:.2f})".format(str(self.observation), self.value_lower, self.value_upper)


class GraphDecisionNode(GraphNode):
    """
        Decision nodes have different meanings depending on their location:
            - planner.nodes[s] stores a DecisionsNode holding information about s: N(s), V(s)
            - DecisionNode.transition[a] stores a ChanceNode holding information about (s,a): N(s,a), Q(s,a), p(s'|s,a)
            - ChanceNode.children[s'] stores a DecisionNode holding information about (s,a,s'): N(s,a,s'), R(s,a,s')


                                              planner.nodes
                                                    |
                                              DecisionNode(s)
                                                 |     |
                                        ActionNode(s,a) ...
                                          |      |
                                  DecisonNode(s,a,s')   ...
    """
    def __init__(self, planner, state, observation):
        super().__init__(planner, state, observation)
        self.count = 0
        """ Visit count N(s) (when in planner.nodes) or N(s,a,s') (when child of a chance node)"""
        self.cumulative_reward = 0
        """ Sum of all rewards r(s,a,s') (when child of a chance node). """
        self.mu_ucb = 1
        """ Upper bound on mean r(s,a,s') (when child of a chance node). """
        self.mu_lcb = 0
        """ Lower bound on mean r(s,a,s') (when child of a chance node)"""

    def selection_rule(self):
        """
            Conservative action selection
        """
        action_values_bound = self.backup("value_lower")
        return max(action_values_bound.items(), key=operator.itemgetter(1))[0]

    def update(self, reward=None):
        self.count += 1
        if reward is not None:
            self.cumulative_reward += reward
            self.compute_reward_ucb()

    def compute_reward_ucb(self):
        if self.planner.config["upper_bound"]["type"] == "kullback-leibler":
            # Variables available for threshold evaluation
            horizon = self.planner.config["horizon"]
            actions = self.planner.env.action_space.n
            confidence = self.planner.config["confidence"]
            count = self.count
            time = self.planner.config["episodes"]
            threshold = eval(self.planner.config["upper_bound"]["threshold"])
            self.mu_ucb = kl_upper_bound(self.cumulative_reward, self.count, 0,
                                         threshold=str(threshold))
            self.mu_lcb = kl_upper_bound(self.cumulative_reward, self.count, 0,
                                         threshold=str(threshold), lower=True)
        else:
            logger.error("Unknown upper-bound type")

    def __str__(self):
        return "{} (L:{:.2f}, U:{:.2f})".format(str(self.observation), self.value_lower, self.value_upper)

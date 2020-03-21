import operator
from collections import defaultdict

import numpy as np
import logging
from rl_agents.agents.common.factory import safe_deepcopy_env
from rl_agents.agents.tree_search.abstract import Node, AbstractTreeSearchAgent, AbstractPlanner

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

    def get_node(self, observation, state=None):
        # Get or create node
        if str(observation) not in self.nodes:
            self.nodes[str(observation)] = GraphNode(self, state, observation)
        return self.nodes[str(observation)]

    def plan(self, state, observation):
        self.root = self.get_node(observation, state=state)
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
        q_values_upper = self.backup("value_upper")
        return max(q_values_upper.items(), key=operator.itemgetter(1))[0]

    def selection_rule(self):
        """
            Conservative action selection
        """
        q_values_lower = self.backup("value_lower")
        return max(q_values_lower.items(), key=operator.itemgetter(1))[0]

    def expand(self):
        try:
            actions = self.state.get_available_actions()
        except AttributeError:
            actions = range(self.state.action_space.n)
        for action in actions:
            # Simulate transition
            next_state = safe_deepcopy_env(self.state)
            next_observation, reward, done, _ = next_state.step(action)
            # Record the transition
            next_node = self.planner.get_node(next_observation)
            next_node.state = next_state
            next_node.parents.append(self)
            self.rewards[action] = reward
            self.children[action] = next_node

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

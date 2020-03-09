import logging
import numpy as np
from gym.utils import seeding

from rl_agents.agents.common.abstract import AbstractAgent
from rl_agents.agents.common.factory import preprocess_env, safe_deepcopy_env
from rl_agents.configuration import Configurable
from rl_agents.agents.tree_search.graphics import TreePlot

logger = logging.getLogger(__name__)


class AbstractTreeSearchAgent(AbstractAgent):
    def __init__(self,
                 env,
                 config=None):
        """
            A new Tree Search agent.
        :param env: The environment
        :param config: The agent configuration. Use default if None.
        """
        super(AbstractTreeSearchAgent, self).__init__(config)
        self.env = env
        self.planner = self.make_planner()
        self.previous_actions = []
        self.remaining_horizon = 0
        self.steps = 0

    @classmethod
    def default_config(cls):
        return {
            "env_preprocessors": [],
            "display_tree": False
        }

    def make_planner(self):
        raise NotImplementedError()

    def plan(self, observation):
        """
            Plan an optimal sequence of actions.

            Start by updating the previously found tree with the last action performed.

        :param observation: the current state
        :return: the list of actions
        """
        self.steps += 1
        replanning_required = self.step(self.previous_actions)
        if replanning_required:
            env = preprocess_env(self.env, self.config["env_preprocessors"])
            actions = self.planner.plan(state=env, observation=observation)
        else:
            actions = self.previous_actions[1:]
        self.write_tree()

        self.previous_actions = actions
        return actions

    def step(self, actions):
        """
            Handle receding horizon mechanism
        :return: whether a replanning is required
        """
        replanning_required = self.remaining_horizon == 0 or len(actions) <= 1
        if replanning_required:
            self.remaining_horizon = self.config["receding_horizon"] - 1
        else:
            self.remaining_horizon -= 1

        self.planner.step(actions)
        return replanning_required

    def reset(self):
        self.planner.step_by_reset()
        self.steps = 0

    def seed(self, seed=None):
        return self.planner.seed(seed)

    def record(self, state, action, reward, next_state, done, info):
        pass

    def act(self, state):
        return self.plan(state)[0]

    def save(self, filename):
        return False

    def load(self, filename):
        return False

    def write_tree(self):
        if self.config["display_tree"] and self.writer:
            TreePlot(self.planner, max_depth=self.config["max_depth"]).plot_to_writer(self.writer, epoch=self.steps, show=True)


class AbstractPlanner(Configurable):
    def __init__(self, config=None):
        super(AbstractPlanner, self).__init__(config)
        self.np_random = None
        self.root = self.make_root()
        self.seed()

    @classmethod
    def default_config(cls):
        return dict(budget=500,
                    gamma=0.8,
                    max_depth=6,
                    step_strategy="reset",
                    receding_horizon=1)

    def make_root(self):
        raise NotImplementedError()

    def seed(self, seed=None):
        """
            Seed the planner randomness source, e.g. for rollout policy
        :param seed: the seed to be used
        :return: the used seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def plan(self, state, observation):
        """
            Plan an optimal sequence of actions.

        :param state: the initial environment state
        :param observation: the corresponding state observation
        :return: the actions sequence
        """
        raise NotImplementedError()

    def get_plan(self):
        """
            Get the optimal action sequence of the current tree by recursively selecting the best action within each
            node with no exploration.

        :return: the list of actions
        """
        actions = []
        node = self.root
        while node.children:
            action = node.selection_rule()
            actions.append(action)
            node = node.children[action]
        return actions

    def step(self, actions):
        """
            Update the planner tree when the agent performs an action

        :param actions: a sequence of actions to follow from the root node
        """
        if self.config["step_strategy"] == "reset":
            self.step_by_reset()
        elif self.config["step_strategy"] == "subtree":
            if actions:
                self.step_by_subtree(actions[0])
            else:
                self.step_by_reset()
        else:
            logger.warning("Unknown step strategy: {}".format(self.config["step_strategy"]))
            self.step_by_reset()

    def step_by_reset(self):
        """
            Reset the planner tree to a root node for the new state.
        """
        self.root = self.make_root()

    def step_by_subtree(self, action):
        """
            Replace the planner tree by its subtree corresponding to the chosen action.

        :param action: a chosen action from the root node
        """
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            # The selected action was never explored, start a new tree.
            self.step_by_reset()


class Node(object):
    """
        A tree node
    """

    def __init__(self, parent, planner):
        """
            New node.

        :param parent: its parent node
        :param planner: the planner using the node
        """
        self.parent = parent
        self.planner = planner

        self.children = {}
        """ Dict of children nodes, indexed by action labels"""

        self.count = 0
        """ Number of times the node was visited."""

        self.value = 0
        """ Estimated value of the node's action sequence"""

    def get_value(self):
        return self.value

    def expand(self, branching_factor):
        for a in range(branching_factor):
            self.children[a] = type(self)(self, self.planner)

    def selection_rule(self):
        raise NotImplementedError()

    @staticmethod
    def breadth_first_search(root, operator=None, condition=None, condition_blocking=True):
        """
            Breadth-first search of all paths to nodes that meet a given condition

        :param root: starting node
        :param operator: will be applied to all traversed nodes
        :param condition: nodes meeting that condition will be returned
        :param condition_blocking: do not explore a node which met the condition
        :return: list of paths to nodes that met the condition
        """
        queue = [(root, [])]
        while queue:
            (node, path) = queue.pop(0)
            if (condition is None) or condition(node):
                returned = operator(node, path) if operator else (node, path)
                yield returned
            if (condition is None) or not condition_blocking or not condition(node):
                for next_key, next_node in node.children.items():
                    queue.append((next_node, path + [next_key]))

    def is_leaf(self):
        return not self.children

    def path(self):
        """
        :return: sequence of action labels from the root to the node
        """
        node = self
        path = []
        while node.parent:
            for a in node.parent.children:
                if node.parent.children[a] == node:
                    path.append(a)
                    break
            node = node.parent
        return reversed(path)

    def sequence(self):
        """
        :return: sequence of nodes from the root to the node
        """
        node = self
        path = [node]
        while node.parent:
            path.append(node.parent)
            node = node.parent
        return reversed(path)

    @staticmethod
    def all_argmax(x):
        """
        :param x: a set
        :return: the list of indexes of all maximums of x
        """
        m = np.amax(x)
        return np.nonzero(x == m)[0]

    def random_argmax(self, x):
        """
            Randomly tie-breaking arg max
        :param x: an array
        :return: a random index among the maximums
        """
        indices = Node.all_argmax(x)
        return self.planner.np_random.choice(indices)

    def __str__(self):
        return "{} ({})".format(list(self.path()), self.value)

    def __repr__(self):
        return '<node {}>'.format(id(self))

    def get_trajectories(self, initial_state, initial_observation=None,
                         as_observations=True, full_trajectories=True, include_leaves=True):
        """
            Get a list of visited nodes/states/trajectories corresponding to the node subtree

        :param initial_state: the state at the root
        :param initial_observation: the observation for the root state
        :param as_observations: return nodes instead of observations
        :param full_trajectories: return a list of observation sequences, else a list of observations
        :param include_leaves: include leaves or only expanded nodes
        :return: the list of trajectories
        """
        trajectories = []
        if initial_observation is None:
            initial_observation = initial_state.reset()
        if not as_observations:
            initial_observation = self  # Return this node instead of this observation
        if self.children:
            for action, child in self.children.items():
                next_state = safe_deepcopy_env(initial_state)
                next_observation, _, _, _ = next_state.step(action)
                child_trajectories = child.get_trajectories(next_state, next_observation,
                                                            as_observations, full_trajectories, include_leaves)
                if full_trajectories:
                    trajectories.extend([[initial_observation] + trajectory for trajectory in child_trajectories])
                else:
                    trajectories.extend(child_trajectories)
            if not full_trajectories:
                trajectories.append(initial_observation)
        elif include_leaves:
            trajectories = [[initial_observation]] if full_trajectories else [initial_observation]
        return trajectories

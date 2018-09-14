import copy

import gym
from gym import logger
import numpy as np

from rl_agents.agents.abstract import AbstractAgent
from rl_agents.agents.common import preprocess_env, safe_deepcopy_env
from rl_agents.agents.tree_search.tree import Node
from rl_agents.configuration import Configurable


class OLOPAgent(AbstractAgent):
    """
        An agent that uses Open Loop Optimistic Planning to plan a sequence of actions in an MDP.
    """

    def __init__(self,
                 env,
                 config=None):
        super(OLOPAgent, self).__init__(config)
        self.env = env
        self.olop = OLOP(env, self.config)
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


class OLOP(Configurable):
    """
       An implementation of Open Loop Optimistic Planning.
    """
    def __init__(self, env, config=None):
        super(OLOP, self).__init__(config)
        self.allocate_budget()
        self.root, self.leaves = self.build_tree(env.action_space.n)

    @classmethod
    def default_config(cls):
        return dict(budget=100,
                    gamma=0.7,
                    step_strategy="reset")

    @staticmethod
    def horizon(episodes, gamma):
        return int(np.ceil(np.log(episodes) / (2 * np.log(1 / gamma))))

    def allocate_budget(self):
        for episodes in range(1, 1000):
            if episodes * OLOP.horizon(episodes, self.config["gamma"]) > self.config["budget"]:
                self.config["episodes"] = episodes - 1
                self.config["horizon"] = OLOP.horizon(self.config["episodes"], self.config["gamma"])
                break
        else:
            raise ValueError("Could not split budget {} with gamma {}".format(self.config["budget"], self.config["gamma"]))

    def build_tree(self, branching_factor):
        root = OLOPNode(parent=None, planner=self)
        leaves = [root]
        for _ in range(self.config["horizon"]):
            next_leaves = []
            for leaf in leaves:
                leaf.expand(branching_factor)
                next_leaves += leaf.children.values()
            leaves = next_leaves
        return root, leaves

    def run(self, state):
        """
            Run an OLOP episode

        :param state: the initial environment state
        """

        # Compute B-values
        list(Node.breadth_first_search(self.root, operator=self.accumulate_ucb, condition=None))
        sequences = list(map(OLOP.sharpen_ucb, self.leaves))
        # Pick best action sequence
        best_sequence = list(self.leaves[np.argmax(sequences)].path())

        # Execute sequence and collect rewards
        node = self.root
        terminal = False
        for action in best_sequence:
            observation, reward, done, _ = state.step(action)
            terminal = terminal or done
            node = node.children[action]
            node.update(reward if not terminal else 0)

    def accumulate_ucb(self, node, path):
        node_t = node
        node.value = self.config["gamma"] ** (len(path) + 1) / (1 - self.config["gamma"])
        try:
            for t in np.arange(len(path), 0, -1):
                node.value += self.config["gamma"]**t * \
                              (node_t.total_reward / node_t.count
                               + np.sqrt(2*np.log(self.config["episodes"])/node_t.count))
                node_t = node_t.parent
        except ZeroDivisionError:
            node.value = np.infty
        return path, node.value

    @staticmethod
    def sharpen_ucb(node):
        node_t = node
        min_ucb = node.value
        while node_t.parent:
            min_ucb = min(min_ucb, node_t.value)
            node_t = node_t.parent
        node.value = min_ucb
        return node.value

    def plan(self, state):
        """
            Plan an optimal sequence of actions

        :param state: the initial environment state
        :return: the list of actions
        """
        for i in range(self.config['episodes']):
            if (i+1) % 10 == 0:
                logger.debug('{} / {}'.format(i+1, self.config['episodes']))
            # Copy the environment state
            try:
                state_copy = safe_deepcopy_env(state)
            except ValueError:
                state_copy = copy.deepcopy(state)
            self.run(state_copy)
        actions = list(self.root.children.keys())
        counts = Node.all_argmax([self.root.children[a].count for a in actions])
        action = actions[max(counts, key=(lambda k: self.root.children[actions[k]].get_value()))]
        print(action, self.root.children[action].value)
        return [action]

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
        self.root, self.leaves = self.build_tree(len(self.root.children))


class OLOPNode(Node):
    def __init__(self, parent, planner):
        super(OLOPNode, self).__init__(parent, planner)
        self.total_reward = 0

    def update(self, reward):
        self.total_reward += reward
        self.count += 1

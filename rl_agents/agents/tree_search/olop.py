from gym import logger
import numpy as np

from rl_agents.agents.common import safe_deepcopy_env
from rl_agents.agents.tree_search.abstract import Node, AbstractTreeSearchAgent, AbstractPlanner


class OLOPAgent(AbstractTreeSearchAgent):
    """
        An agent that uses Open Loop Optimistic Planning to plan a sequence of actions in an MDP.
    """
    def make_planner(self):
        return OLOP(self.env, self.config)


class OLOP(AbstractPlanner):
    """
       An implementation of Open Loop Optimistic Planning.
    """
    def __init__(self, env, config=None):
        self.leaves = None
        super(OLOP, self).__init__(config)

    def make_root(self):
        root, self.leaves = self.build_tree()
        return root

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
        if "horizon" not in self.config:
            self.allocate_budget()
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

    def plan(self, state, observation):
        for i in range(self.config['episodes']):
            if (i+1) % 10 == 0:
                logger.debug('{} / {}'.format(i+1, self.config['episodes']))
            self.run(safe_deepcopy_env(state))

        return self.get_plan()


class OLOPNode(Node):
    def __init__(self, parent, planner):
        super(OLOPNode, self).__init__(parent, planner)
        self.total_reward = 0

    def selection_rule(self):
        # Tie best counts by best value
        actions = list(self.children.keys())
        counts = Node.all_argmax([self.children[a].count for a in actions])
        return actions[max(counts, key=(lambda i: self.children[actions[i]].get_value()))]

    def update(self, reward):
        if not 0 <= reward <= 1:
            raise ValueError("This planner assumes that all rewards are normalized in [0, 1]")
        self.total_reward += reward
        self.count += 1

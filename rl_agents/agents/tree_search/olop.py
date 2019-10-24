import logging
import numpy as np

from rl_agents.agents.common.factory import safe_deepcopy_env
from rl_agents.agents.tree_search.abstract import Node, AbstractTreeSearchAgent, AbstractPlanner
from rl_agents.utils import hoeffding_upper_bound, kl_upper_bound, laplace_upper_bound

logger = logging.getLogger(__name__)


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
        self.env = env
        super(OLOP, self).__init__(config)

    @classmethod
    def default_config(cls):
        cfg = super(OLOP, cls).default_config()
        cfg.update(
            {
                "upper_bound":
                {
                    "type": "hoeffding",
                    "time": "global",
                    "c": 4
                },
                "lazy_tree_construction": True,
                "continuation_type": "zeros"
            }
        )
        return cfg

    def make_root(self):
        root = OLOPNode(parent=None, planner=self)
        self.leaves = [root]
        if "horizon" not in self.config:
            self.allocate_budget()

        if not self.config["lazy_tree_construction"]:
            self.prebuild_tree(self.env.action_space.n)

        return root

    def prebuild_tree(self, branching_factor):
        """
            Build a full search tree with a given branching factor and depth.

            In the original OLOP paper, the tree is built in advance so that leaves at depth L can be seen
            as arms of a structured bandit problem.
        :param branching_factor: The number of actions in each state
        """
        for _ in range(self.config["horizon"]):
            next_leaves = []
            for leaf in self.leaves:
                super(OLOPNode, leaf).expand(branching_factor)
                next_leaves += leaf.children.values()
            self.leaves = next_leaves

    @staticmethod
    def horizon(episodes, gamma):
        return max(int(np.ceil(np.log(episodes) / (2 * np.log(1 / gamma)))), 1)

    def allocate_budget(self):
        """
            Allocate the computational budget into M episodes of fixed horizon L.
        """
        if self.config["budget"] < self.env.action_space.n:
            self.config["budget"] = self.env.action_space.n
        for episodes in range(1, int(self.config["budget"])):
            if episodes * OLOP.horizon(episodes, self.config["gamma"]) > self.config["budget"]:
                self.config["episodes"] = max(episodes - 1, 1)
                self.config["horizon"] = OLOP.horizon(self.config["episodes"], self.config["gamma"])
                break
        else:
            raise ValueError("Could not split budget {} with gamma {}".format(self.config["budget"], self.config["gamma"]))

    def run(self, state):
        """
            Run an OLOP episode.

            Find the leaf with highest upper bound value, and sample the corresponding action sequence.

        :param state: the initial environment state
        """
        # Compute B-values
        list(Node.breadth_first_search(self.root, operator=self.compute_ucbs, condition=None))
        list(Node.breadth_first_search(self.root, operator=self.compute_u_values, condition=None))
        sequences_upper_bounds = list(map(OLOP.sharpen_b_values, self.leaves))

        # Pick best sequence of actions
        best_leaf_index = self.np_random.choice(Node.all_argmax(sequences_upper_bounds))
        best_sequence = list(self.leaves[best_leaf_index].path())

        # If the sequence length is shorter than the horizon (which can happen with lazy tree construction),
        # all continuations have the same upper-bounds. Pick one continuation arbitrarily.
        if self.config["continuation_type"] == "zeros":
            # Here, pad with the sequence [0, ..., 0].
            best_sequence = best_sequence[:self.config["horizon"]] + [0]*(self.config["horizon"] - len(best_sequence))
        elif self.config["continuation_type"] == "uniform":
            best_sequence = best_sequence[:self.config["horizon"]]\
                            + self.np_random.choice(range(state.action_space.n),
                                                    self.config["horizon"] - len(best_sequence)).tolist()

        # Execute sequence, expand tree if needed, collect rewards and update upper confidence bounds.
        node = self.root
        for action in best_sequence:
            if not node.children:
                self.leaves = node.expand(state, self.leaves, update_children=False)
            if action not in node.children:  # Default action may not be available
                action = list(node.children.keys())[0]  # Pick first available action instead
            observation, reward, done, _ = state.step(action)
            node = node.children[action]
            node.update(reward, done)
            # if node.done:
            #     break

    def compute_ucbs(self, node, path):
        """
            Update the nodes mean reward UCB.

            This should only be needed for upper-bounds that depend on global times.
            Bounds with only local time dependency can be computed when updating sampled nodes, without requiring
            a global update of the rest of the tree.
        """
        if self.config["upper_bound"]["time"] == "local":
            node.compute_ucb()

    def compute_u_values(self, node, path):
        """
            Compute the upper bound of the value of the action sequence at a given node.

            It represents the maximum admissible mean reward over trajectories that start with this particular sequence.
            It is computed by summing upper bounds of intermediate mean rewards along the sequence, and an upper bound
            of the remaining rewards over possible continuations of the sequence.
        :param node: a node in the look-ahead tree
        :param path: the path from the root to the node
        :return: the path from the root to the node, and the node value.
        """
        # Upper bound of the reward-to-go after this node
        node.value = self.config["gamma"] ** (len(path) + 1) / (1 - self.config["gamma"])
        node_t = node
        for t in np.arange(len(path), 0, -1):  # from current node up to the root
            node.value += self.config["gamma"]**t * node_t.mu_ucb  # upper bound of the node mean reward
            node_t = node_t.parent
        return path, node.value

    @staticmethod
    def sharpen_b_values(node):
        """
            Sharpen the upper-bound value of the action sequences at the tree leaves.

            By computing the min over intermediate upper-bounds along the sequence, that must all be satisfied.
            If the KL-UCB are used, the leaf always has the lowest value UCB and no further sharpening can be achieved.
        :param node: a node in the look-ahead tree
        :return:an upper-bound of the sequence value
        """
        if node.planner.config["upper_bound"] == "kullback-leibler":
            return node.value
        else:
            node_t = node
            min_value = node.value
            while node_t.parent:
                min_value = min(min_value, node_t.value)
                node_t = node_t.parent
            return min_value

    def plan(self, state, observation):
        for self.episode in range(self.config['episodes']):
            if (self.episode+1) % max(self.config['episodes'] // 10, 1) == 0:
                logger.debug('{} / {}'.format(self.episode+1, self.config['episodes']))
            self.run(safe_deepcopy_env(state))

        return self.get_plan()


class OLOPNode(Node):
    STOP_ON_ANY_TERMINAL_STATE = True

    def __init__(self, parent, planner):
        super(OLOPNode, self).__init__(parent, planner)

        self.cumulative_reward = 0
        """ Sum of all rewards received at this node. """

        self.mu_ucb = np.infty
        """ Upper bound of the node mean reward. """

        if self.planner.config["upper_bound"] == "kullback-leibler":
            self.mu_ucb = 1

        self.done = False
        """ Is this node a terminal node, for all random realizations (!)"""

    def selection_rule(self):
        # Tie best counts by best value
        actions = list(self.children.keys())
        counts = Node.all_argmax([self.children[a].count for a in actions])
        return actions[max(counts, key=(lambda i: self.children[actions[i]].get_value()))]

    def update(self, reward, done):
        if not 0 <= reward <= 1:
            raise ValueError("This planner assumes that all rewards are normalized in [0, 1]")

        if done or (self.parent and self.parent.done) and OLOPNode.STOP_ON_ANY_TERMINAL_STATE:
            self.done = True
        if self.done:
            reward = 0
        self.cumulative_reward += reward
        self.count += 1
        self.compute_ucb()



    def compute_ucb(self):
        if self.planner.config["upper_bound"]["time"] == "local":
            time = self.planner.episode + 1
        elif self.planner.config["upper_bound"]["time"] == "global":
            time = self.planner.config["episodes"]
        else:
            logger.error("Unknown upper-bound time reference")

        if self.planner.config["upper_bound"]["type"] == "hoeffding":
            self.mu_ucb = hoeffding_upper_bound(self.cumulative_reward, self.count, time,
                                                c=self.planner.config["upper_bound"]["c"])
        elif self.planner.config["upper_bound"]["type"] == "laplace":
            self.mu_ucb = laplace_upper_bound(self.cumulative_reward, self.count, time,
                                              c=self.planner.config["upper_bound"]["c"])
        elif self.planner.config["upper_bound"]["type"] == "kullback-leibler":
            self.mu_ucb = kl_upper_bound(self.cumulative_reward, self.count, time,
                                         c=self.planner.config["upper_bound"]["c"])
        else:
            logger.error("Unknown upper-bound type")

    def expand(self, state, leaves, update_children=False):
        if state is None:
            raise Exception("The state should be set before expanding a node")
        try:
            actions = state.get_available_actions()
        except AttributeError:
            actions = range(state.action_space.n)
        for action in actions:
            self.children[action] = type(self)(self,
                                               self.planner)
            if update_children:
                _, reward, done, _ = safe_deepcopy_env(state).step(action)
                self.children[action].update(reward, done)

        idx = leaves.index(self)
        leaves = leaves[:idx] + list(self.children.values()) + leaves[idx+1:]
        return leaves

import logging
import numpy as np

from rl_agents.agents.common.factory import safe_deepcopy_env
from rl_agents.agents.tree_search.abstract import Node, AbstractTreeSearchAgent, AbstractPlanner
from rl_agents.utils import kl_upper_bound

logger = logging.getLogger(__name__)


class OLOP(AbstractPlanner):
    """
       An implementation of Open Loop Optimistic Planning.
    """
    def __init__(self, env, config=None):
        self.leaves = None
        self.env = env
        super().__init__(config)

    @classmethod
    def default_config(cls):
        cfg = super(OLOP, cls).default_config()
        cfg.update(
            {
                "upper_bound":
                {
                    "type": "hoeffding",
                    "time": "global",
                    "threshold": "4*np.log(time)"
                },
                "continuation_type": "zeros"
            }
        )
        return cfg

    def reset(self):
        if "horizon" not in self.config:
            self.allocate_budget()
        self.root = OLOPNode(parent=None, planner=self)
        self.leaves = [self.root]

    @staticmethod
    def horizon(episodes, gamma):
        return max(int(np.ceil(np.log(episodes) / (2 * np.log(1 / gamma)))), 1)

    def allocate_budget(self):
        budget = max(self.env.action_space.n, self.config["budget"])
        self.config["episodes"], self.config["horizon"] = self.allocation(budget, self.config["gamma"])

    @staticmethod
    def allocation(budget, gamma):
        """
            Allocate the computational budget into M episodes of fixed horizon L.
        """
        for episodes in range(1, int(budget)):
            if episodes * OLOP.horizon(episodes, gamma) > budget:
                episodes = max(episodes - 1, 1)
                horizon = OLOP.horizon(episodes, gamma)
                break
        else:
            raise ValueError("Could not split budget {} with gamma {}".format(budget, gamma))
        return episodes, horizon

    def run(self, state):
        """
            Run an OLOP episode.

            Find the leaf with highest upper bound value, and sample the corresponding action sequence.

        :param state: the initial environment state
        """
        # We need randomness
        state.seed(self.np_random.randint(2**30))

        # Follow selection policy, expand tree if needed, collect rewards and update confidence bounds.
        node = self.root
        for h in range(self.config["horizon"]):
            # Select action
            if not node.children:  # Break ties at leaves
                node.expand(state)
                action = self.np_random.choice(list(node.children.keys())) \
                    if self.config["continuation_type"] == "uniform" else 0
            else:  # Run UCB elsewhere
                action, _ = max([child for child in node.children.items()], key=lambda c: c[1].value_upper)

            # Perform transition
            observation, reward, done, _ = self.step(state, action)
            node = node.children[action]
            node.update(reward, done)

        # Backup global statistics
        node.backup_to_root()

    def plan(self, state, observation):
        for self.episode in range(self.config['episodes']):
            if (self.episode+1) % max(self.config['episodes'] // 10, 1) == 0:
                logger.debug('{} / {}'.format(self.episode+1, self.config['episodes']))
            self.run(safe_deepcopy_env(state))

        return self.get_plan()


class OLOPNode(Node):
    STOP_ON_ANY_TERMINAL_STATE = False

    def __init__(self, parent, planner):
        super(OLOPNode, self).__init__(parent, planner)

        self.cumulative_reward = 0
        """ Sum of all rewards received at this node. """

        self.mu_ucb = np.infty
        """ Upper bound of the node mean reward. """

        if self.planner.config["upper_bound"]["type"] == "kullback-leibler":
            self.mu_ucb = 1

        gamma = self.planner.config["gamma"]

        self.depth = self.parent.depth + 1 if self.parent is not None else 0
        self.value_upper = (1 - gamma ** (self.planner.config["horizon"] + 1 - self.depth)) / (1 - gamma)

        self.done = False
        """ Is this node a terminal node, for all random realizations (!)"""

    def selection_rule(self):
        # Tie best counts by best value upper bound
        actions = list(self.children.keys())
        counts = Node.all_argmax([self.children[a].count for a in actions])
        return actions[max(counts, key=(lambda i: self.children[actions[i]].value_upper))]

    def update(self, reward, done):
        if not 0 <= reward <= 1:
            raise ValueError("This planner assumes that all rewards are normalized in [0, 1]")

        if done or (self.parent and self.parent.done) and OLOPNode.STOP_ON_ANY_TERMINAL_STATE:
            self.done = True
        if self.done:
            reward = 0
        self.cumulative_reward += reward
        self.count += 1
        self.compute_reward_ucb()

    def compute_reward_ucb(self):
        if self.planner.config["upper_bound"]["time"] == "local":
            time = self.planner.episode + 1
        elif self.planner.config["upper_bound"]["time"] == "global":
            time = self.planner.config["episodes"]
        else:
            time = np.nan
            logger.error("Unknown upper-bound time reference")

        # if self.planner.config["upper_bound"]["type"] == "hoeffding":
        #     self.mu_ucb = hoeffding_upper_bound(self.cumulative_reward, self.count, time,
        #                                         c=self.planner.config["upper_bound"]["c"])
        # elif self.planner.config["upper_bound"]["type"] == "laplace":
        #     self.mu_ucb = laplace_upper_bound(self.cumulative_reward, self.count, time,
        #                                       c=self.planner.config["upper_bound"]["c"])
        if self.planner.config["upper_bound"]["type"] == "kullback-leibler":
            threshold = eval(self.planner.config["upper_bound"]["threshold"])
            self.mu_ucb = kl_upper_bound(self.cumulative_reward, self.count, threshold)
        else:
            logger.error("Unknown upper-bound type")

    def expand(self, state):
        if state is None:
            raise Exception("The state should be set before expanding a node")
        try:
            actions = state.get_available_actions()
        except AttributeError:
            actions = range(state.action_space.n)
        for action in actions:
            self.children[action] = type(self)(self,
                                               self.planner)

        # Replace the former leaf by its children, but keep the ordering
        idx = self.planner.leaves.index(self)
        self.planner.leaves = self.planner.leaves[:idx] + \
                              list(self.children.values()) + \
                              self.planner.leaves[idx+1:]

    def backup_to_root(self):
        """
            Bellman V(s) = max_a Q(s,a)
        """
        if self.children:
            gamma = self.planner.config["gamma"]
            self.value_upper = self.mu_ucb + gamma * np.amax([c.value_upper for c in self.children.values()])
        else:
            assert self.depth == self.planner.config["horizon"]
            self.value_upper = self.mu_ucb
        if self.parent:
            self.parent.backup_to_root()


class OLOPAgent(AbstractTreeSearchAgent):
    """
        An agent that uses Open Loop Optimistic Planning to plan a sequence of actions in an MDP.
    """
    PLANNER_TYPE = OLOP

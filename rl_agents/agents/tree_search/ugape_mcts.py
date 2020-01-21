import logging
import numpy as np

from rl_agents.agents.common.factory import safe_deepcopy_env
from rl_agents.agents.tree_search.abstract import Node, AbstractTreeSearchAgent, AbstractPlanner
from rl_agents.agents.tree_search.olop import OLOPAgent, OLOP, OLOPNode
from rl_agents.utils import hoeffding_upper_bound, kl_upper_bound, laplace_upper_bound

logger = logging.getLogger(__name__)


class UgapEMCTSAgent(OLOPAgent):
    """
        An agent that uses Open Loop Optimistic Planning to plan a sequence of actions in an MDP.
    """
    def make_planner(self):
        return UgapEMCTS(self.env, self.config)


class UgapEMCTS(OLOP):
    """
       Best-Arm Identification MCTS.
    """
    @classmethod
    def default_config(cls):
        cfg = super(OLOP, cls).default_config()
        cfg.update(
            {
                "accuracy": 1.0,
                "confidence": 0.9,
                "upper_bound":
                {
                    "type": "kullback-leibler",
                    "time": "global",
                    "threshold": "3*np.log(1 + np.log(count))"
                                 "+ horizon*np.log(actions)"
                                 "+ np.log(1/(1-confidence))"
                },
                "continuation_type": "uniform",
                "horizon_from_accuracy": False
            }
        )
        return cfg

    def make_root(self):
        if "horizon" not in self.config:
            self.allocate_budget()
        root = UGapEMCTSNode(parent=None, planner=self)
        self.leaves = [root]
        return root

    def allocate_budget(self):
        """
            Allocate the computational budget into M episodes of fixed horizon L.
        """
        if self.config["horizon_from_accuracy"]:
            self.config["horizon"] = int(np.ceil(np.log(self.config["accuracy"] * (1 - self.config["gamma"]) / 2) \
                                     / np.log(self.config["gamma"])))
            self.config["episodes"] = self.config["budget"] // self.config["horizon"]
            assert self.config["episodes"] > 1
            logger.debug("Planning at depth H={}".format(self.config["horizon"]))
        else:
            super().allocate_budget()

    def run(self, state):
        """
            Run an OLOP episode.

            Find the leaf with highest upper bound value, and sample the corresponding action sequence.

        :param state: the initial environment state
        """
        if self.root.children:
            logger.debug(" / ".join(["a{} ({}): [{:.3f}, {:.3f}]".format(k, n.count, n.value_lower, n.value)
                                   for k, n in self.root.children.items()]))
            # Run UGapE for first action selection
            selected_child, best, challenger = self.root.best_arm_identification_selection()
            selected_action = next(selected_child.path())

            # Run UCB for the rest of the sequence
            best_sequence = [selected_action]
            while selected_child.children:
                action, selected_child = max([child for child in selected_child.children.items()], key=lambda c: c[1].value)
                best_sequence.append(action)
        else:
            best_sequence, best, challenger = [], None, None

        # If the sequence length is shorter than the horizon (which can happen with lazy tree construction),
        # all continuations have the same upper-bounds. Pick one continuation arbitrarily.
        if self.config["continuation_type"] == "zeros":
            # Here, pad with the sequence [0, ..., 0].
            best_sequence = best_sequence[:self.config["horizon"]] + [0]*(self.config["horizon"] - len(best_sequence))
        elif self.config["continuation_type"] == "uniform":
            best_sequence = best_sequence[:self.config["horizon"]]\
                            + self.np_random.choice(range(state.action_space.n),
                                                    self.config["horizon"] - len(best_sequence)).tolist()

        # Execute sequence, expand tree if needed, collect rewards and update confidence bounds.
        node = self.root
        for action in best_sequence:
            if not node.children:
                node.expand(state)
            if action not in node.children:  # Default action may not be available
                action = list(node.children.keys())[0]  # Pick first available action instead
            observation, reward, done, _ = state.step(action)
            node = node.children[action]
            node.update(reward, done)
        node.backup_to_root()

        return best, challenger

    def plan(self, state, observation):
        done = False
        episode = 0
        while not done:
            best, challenger = self.run(safe_deepcopy_env(state))

            # Stopping rule
            done = challenger.value - best.value_lower < self.config["accuracy"] if best is not None else False
            done = done or episode > self.config["episodes"]

            episode += 1
            if episode % 10 == 0:
                logger.debug('Episode {}: delta = {}/{}'.format(episode,
                                                                challenger.value - best.value_lower,
                                                                self.config["accuracy"]))

        return self.get_plan()


class UGapEMCTSNode(OLOPNode):
    def __init__(self, parent, planner):
        super().__init__(parent, planner)

        self.depth = 0 if parent is None else parent.depth + 1

        self.mu_lcb = -np.infty
        """ Lower bound of the node mean reward. """

        if self.planner.config["upper_bound"] == "kullback-leibler":
            self.mu_lcb = 0

        gamma = self.planner.config["gamma"]
        H = self.planner.config["horizon"]
        self.value = (1 - gamma ** (H-self.depth)) / (1 - gamma)

        """ Lower bound on the node optimal reward-to-go """
        self.value_lower = 0

        self.gap = -np.infty
        """ Maximum possible gap from this node to its neighbours, based on their value confidence intervals """

    def selection_rule(self):
        # Best arm identification at the root
        if self.planner.root == self:
            _, best_node, _ = self.best_arm_identification_selection()
            return next(best_node.path())

        # Then follow the optimistic values
        actions = list(self.children.keys())
        index = self.random_argmax([self.children[a].value for a in actions])
        return actions[index]

    def compute_ucb(self):
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

    def backup_to_root(self):
        if self.parent:
            children = self.parent.children.values()
            gamma = self.planner.config["gamma"]
            self.parent.value = self.parent.mu_ucb + gamma * np.amax([c.value for c in children])
            self.parent.value_lower = self.parent.mu_lcb + gamma * np.amax([c.value_lower for c in children])
            self.parent.backup_to_root()

    def compute_children_gaps(self):
        """
            For best arm identification: compute for each child how much the other actions are potentially better.
        """
        for child in self.children.values():
            child.gap = -np.infty
            for other in self.children.values():
                if other is not child:
                    child.gap = max(child.gap, other.value - child.value_lower)

    def best_arm_identification_selection(self):
        """
            Run UGapE on the children on this node, based on their value confidence intervals.
        :return: selected arm, best candidate, challenger
        """
        # Best candidate child has the lowest potential gap
        self.compute_children_gaps()
        best = min(self.children.values(), key=lambda c: c.gap)
        # Challenger: not best and highest value upper bound
        challenger = max([c for c in self.children.values() if c is not best], key=lambda c: c.value)
        # Selection: the one with highest uncertainty
        return max([best, challenger], key=lambda n: n.value - n.value_lower), best, challenger



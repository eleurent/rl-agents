import logging
import numpy as np

from rl_agents.agents.common.factory import safe_deepcopy_env
from rl_agents.agents.tree_search.abstract import Node, AbstractTreeSearchAgent, AbstractPlanner
from rl_agents.agents.tree_search.olop import OLOP

logger = logging.getLogger(__name__)


class SparseSampling(AbstractPlanner):
    """
       A Sparse Sampling Algorithm for Near-Optimal Planning in Large Markov Decision Processes.
    """
    def __init__(self, env, config=None):
        super().__init__(config)

    def reset(self):
        self.root = DecisionNode(parent=None, planner=self)

    def plan(self, state, observation):
        self.root.estimateV(state)
        logger.debug(f"{len(self.observations)} samples")
        return self.get_plan()

    def get_plan(self):
        """Only return the first action, the rest is conditioned on observations"""
        return [self.root.selection_rule()]


class DecisionNode(Node):
    def __init__(self, parent, planner):
        super().__init__(parent, planner)
        self.depth = self.parent.depth + 1 if self.parent is not None else 0
        self.value = 0
        self.count = 0

    def estimateV(self, state):
        # logger.debug(f"Run estimateV at {state.mdp.state} with depth {self.depth}")
        try:
            actions = state.get_available_actions()
        except AttributeError:
            actions = range(state.action_space.n)

        if self.depth == self.planner.config["horizon"]:
            return

        for action in actions:
            chance_node = self.get_child(action)
            chance_node.estimateQ(state, action)
        self.value = np.amax([c.value for c in self.children.values()])

    def selection_rule(self):
        actions = list(self.children.keys())
        index = self.random_argmax([self.children[a].value for a in actions])
        return actions[index]

    def get_child(self, action):
        if action not in self.children:
            self.children[action] = ChanceNode(self, self.planner)
        return self.children[action]


class ChanceNode(Node):
    def __init__(self, parent, planner):
        assert parent is not None
        super().__init__(parent, planner)
        self.depth = self.parent.depth
        self.value = 0

    def estimateQ(self, state, action):
        if self.depth == self.planner.config["horizon"]:
            return
        # logger.debug(f"Run estimateQ at {state.mdp.state},{action} with depth {self.depth}")

        for i in range(self.planner.config["C"]):
            next_state = safe_deepcopy_env(state)
            # We need randomness
            next_state.seed(self.planner.np_random.randint(2**30))

            observation, reward, done, _ = next_state.step(action)
            # observation = str(observation) + str(i)  # Prevent state merge
            self.get_child(observation).count += 1
            self.get_child(observation).state = next_state
        for next_state_node in self.children.values():
            next_state_node.estimateV(next_state_node.state)
        self.value = reward + self.planner.config["gamma"] * sum(next_state_node.value * next_state_node.count
                                for next_state_node in self.children.values()) / self.planner.config["C"]

    def selection_rule(self):
        raise AttributeError("Selection is done in DecisionNodes, not ChanceNodes")

    def get_child(self, obs):
        if str(obs) not in self.children:
            self.children[str(obs)] = DecisionNode(self, self.planner)
        return self.children[str(obs)]


class SparseSamplingAgent(AbstractTreeSearchAgent):
    """
        An agent that uses SparseSampling to plan a sequence of actions in an MDP.
    """
    PLANNER_TYPE = SparseSampling
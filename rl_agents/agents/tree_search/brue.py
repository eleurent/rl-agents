import logging
import numpy as np

from rl_agents.agents.common.factory import safe_deepcopy_env
from rl_agents.agents.tree_search.abstract import Node, AbstractTreeSearchAgent, AbstractPlanner
from rl_agents.utils import hoeffding_upper_bound, kl_upper_bound, laplace_upper_bound

logger = logging.getLogger(__name__)


class BRUEAgent(AbstractTreeSearchAgent):
    """
        An agent that uses BRUE to plan a sequence of actions in an MDP.
    """
    def make_planner(self):
        return BRUE(self.env, self.config)


class BRUE(AbstractPlanner):
    """
       Best Recommendation with Uniform Exploration algorithm.
    """
    def __init__(self, env, config=None):
        self.env = env
        self.available_budget = 0
        super().__init__(config)

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update({
            "horizon": 6
        })
        return cfg

    def make_root(self):
        root = DecisionNode(parent=None, planner=self)
        return root

    def rollout(self, state, observation):
        for h in range(self.config["horizon"]):
            action = self.np_random.randint(state.action_space.n)
            next_observation, reward, done, _ = state.step(action)
            yield observation, action, reward, next_observation, done
            observation = next_observation
            self.available_budget -= 1
            if done:
                break

    def update(self, rollout):
        state_node = self.root
        to_update = []

        # Get or create the sequence of visited nodes
        for obs, action, reward, next_obs, done in rollout:
            chance_node = state_node.get_child(action)
            next_state_node = chance_node.get_child(next_obs)
            to_update.append((state_node, chance_node, reward, next_state_node))
            state_node = next_state_node

        # Update counts, rewards and estimated returns
        for state_node, chance_node, reward, next_state_node in reversed(to_update):
            next_state_node.update(reward)
            estimated_return = reward + self.config["gamma"] * self.estimate(next_state_node)
            chance_node.update(estimated_return)

    def estimate(self, state_node):
        return_ = 0
        for d in range(self.config["horizon"] - state_node.depth):
            if not state_node.children:
                break
            chance_node = max(state_node.children.values(), key=lambda child: child.value)
            next_states = list(chance_node.children.values())
            counts = np.array([state.count for state in next_states])
            state_node = self.np_random.choice(next_states, p=counts / counts.sum())
            return_ += self.config["gamma"]**d * state_node.reward
        return return_

    def plan(self, state, observation):
        self.available_budget = self.config["budget"]
        while self.available_budget > 0:
            rollout = self.rollout(safe_deepcopy_env(state), observation)
            self.update(rollout)
        return self.get_plan()

    def get_plan(self):
        """Only return the first action, the rest is conditioned on observations"""
        return [self.root.selection_rule()]


class DecisionNode(Node):
    def __init__(self, parent, planner):
        super().__init__(parent, planner)
        self.depth = self.parent.depth + 1 if self.parent is not None else 0
        self.reward = 0

    def update(self, reward):
        if reward is not None:
            self.count += 1
            self.reward = (self.count - 1) / self.count * self.reward + reward / self.count

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

    def update(self, return_):
        self.count += 1
        self.value = (self.count - 1) / self.count * self.value + return_ / self.count

    def selection_rule(self):
        raise AttributeError("Selection is done in DecisionNodes, not ChanceNodes")

    def get_child(self, obs):
        if str(obs) not in self.children:
            self.children[str(obs)] = DecisionNode(self, self.planner)
        return self.children[str(obs)]

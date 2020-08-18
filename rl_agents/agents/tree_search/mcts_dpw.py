import logging
import numpy as np
from functools import partial
import time
from rl_agents.agents.common.factory import safe_deepcopy_env
# from rl_agents.agents.tree_search.abstract import Node, AbstractTreeSearchAgent, AbstractPlanner
from rl_agents.agents.tree_search.mcts import MCTSNode, MCTSAgent, MCTS

logger = logging.getLogger(__name__)
class MCTSDPWAgent(MCTSAgent):
    """
        An MCTS_DPW agent that uses Double Progressive Widenning for handling continuous
        stochastic MDPs.
    """
    def make_planner(self):
        rollout_policy = MCTSDPWAgent.policy_factory(self.config["rollout_policy"])
        prior_policy = MCTSDPWAgent.policy_factory(self.config["prior_policy"])
        return MCTSDPW(self.env, prior_policy, rollout_policy, self.config)

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "budget": 100,
            "gamma": 0.95,
         })
        return config

class MCTSDPW(MCTS):
    """
       An implementation of Monte-Carlo Tree Search with Upper Confidence Tree exploration
       and Double Progressive Widenning.
    """
    def __init__(self, env, prior_policy, rollout_policy, config=None):
        """
            New MCTSDPW instance.

        :param config: the mcts configuration. Use default if None.
        :param rollout_policy: the rollout policy used to estimate the value of a leaf node
        """
        super().__init__(env, prior_policy, rollout_policy, config)

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update({
            "temperature": 1,
            "closed_loop": False,
            "k_state": 1,
            "alpha_state": 0.3,
            "k_action": 3,
            "alpha_action": 0.3,
        })
        return cfg

    def reset(self):
        self.root = DecisionNode(parent=None, planner=self)

    def run(self, state, observation):
        """
            Run an iteration of MCTSDPW, starting from a given state
        :param state: the initial environment state
        :param observation: the corresponding observation
        """
        decision_node = self.root
        total_reward = 0
        depth = 0
        terminal = False
        state.seed(self.np_random.randint(2**30))
        while depth < self.config['horizon'] and not terminal and \
                        (decision_node.count != 0 or decision_node == self.root):


            # perform an action followed by a transition
            chance_node, action = decision_node.get_child(state, temperature=self.config['temperature'])
            observation, reward, terminal, _ = self.step(state, action)


            node_observation = observation if self.config["closed_loop"] else None
            decision_node = chance_node.get_child(node_observation)

            total_reward += self.config["gamma"] ** depth * reward
            depth += 1


        if not terminal:
            total_reward = self.evaluate(state, observation, total_reward, depth=depth)

        # Backup global statistics
        decision_node.backup_to_root(total_reward)

    def get_plan(self):
        """Only return the first action, the rest is conditioned on observations"""
        return self.root.selection_rule()

class DecisionNode(MCTSNode):
    K = 1.0
    """ The value function first-order filter gain"""

    def __init__(self, parent, planner):
        super().__init__(parent, planner)
        self.value = 0
        self.k_action = self.planner.config["k_action"]
        self.alpha_action = self.planner.config["alpha_action"]

    def unexplored_actions(self, state):
        if state is None:
            raise Exception("The state should be set before expanding a node")
        try:
            actions = state.get_available_actions()
        except AttributeError:
            actions = range(state.action_space.n)
        return set(self.children.keys()).symmetric_difference(actions)

    def expand(self, state):
        action = self.planner.np_random.choice(list(self.unexplored_actions(state)))
        self.children[action] = ChanceNode(self, self.planner)
        return self.children[action], action

    def get_child(self, state, temperature=None):
        if len(self.children) == len(state.get_available_actions()) \
                or self.k_action*self.count**self.alpha_action < len(self.children):
            # select one of previously expanded actions
            return self.selection_strategy(temperature)
        else:
            # insert a new aciton
            return self.expand(state)

    def backup_to_root(self, total_reward):
        """
            Update the whole branch from this node to the root with the total reward of the corresponding trajectory.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.update(total_reward)
        if self.parent:
            self.parent.backup_to_root(total_reward)

    def selection_strategy(self, temperature):
        """
            Select an action according to UCB.

        :param temperature: the exploration parameter, positive or zero.
        :return: the selected action with maximum value and exploration bonus.
        """

        actions = list(self.children.keys())
        indexes = []
        for a in actions:
            ucb_val = self.children[a].value + temperature * np.sqrt(np.log(self.count / (self.children[a].count)))
            indexes.append(ucb_val)

        action = actions[self.random_argmax(indexes)]
        return self.children[action], action


class ChanceNode(MCTSNode):
    K = 1.0
    """ The value function first-order filter gain"""
    def __init__(self, parent, planner):
        assert parent is not None
        super().__init__(parent, planner)
        # state progressive widenning parameters
        self.k_state = self.planner.config["k_state"]
        self.alpha_state = self.planner.config["alpha_state"]
        self.value = 0

    def expand(self, obs_id):
        self.children[obs_id] = DecisionNode(self, self.planner)

    def get_child(self, observation):
        import hashlib
        obs_id = hashlib.sha1(str(observation).encode("UTF-8")).hexdigest()[:5]
        if obs_id not in self.children:
            if self.k_state*self.count**self.alpha_state < len(self.children):
                obs_id = self.planner.np_random.choice(list(self.children))
                return self.children[obs_id]
            else:
                # Add observation to the children set
                self.expand(obs_id)

        return self.children[obs_id]

    def backup_to_root(self, total_reward):
        """
            Update the whole branch from this node to the root with the total reward of the corresponding trajectory.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        assert self.children
        assert self.parent
        self.update(total_reward)
        self.parent.backup_to_root(total_reward)

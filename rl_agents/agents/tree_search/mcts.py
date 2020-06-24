import logging
import numpy as np
from functools import partial

from rl_agents.agents.common.factory import safe_deepcopy_env
from rl_agents.agents.tree_search.abstract import Node, AbstractTreeSearchAgent, AbstractPlanner
from rl_agents.agents.tree_search.olop import OLOP

logger = logging.getLogger(__name__)


class MCTSAgent(AbstractTreeSearchAgent):
    """
        An agent that uses Monte Carlo Tree Search to plan a sequence of action in an MDP.
    """
    def make_planner(self):
        prior_policy = MCTSAgent.policy_factory(self.config["prior_policy"])
        rollout_policy = MCTSAgent.policy_factory(self.config["rollout_policy"])
        return MCTS(self.env, prior_policy, rollout_policy, self.config)

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "budget": 100,
            "horizon": None,
            "prior_policy": {"type": "random_available"},
            "rollout_policy": {"type": "random_available"},
            "env_preprocessors": []
         })
        return config

    @staticmethod
    def policy_factory(policy_config):
        if policy_config["type"] == "random":
            return MCTSAgent.random_policy
        elif policy_config["type"] == "random_available":
            return MCTSAgent.random_available_policy
        elif policy_config["type"] == "preference":
            return partial(MCTSAgent.preference_policy,
                           action_index=policy_config["action"],
                           ratio=policy_config["ratio"])
        else:
            raise ValueError("Unknown policy type")

    @staticmethod
    def random_policy(state, observation):
        """
            Choose actions from a uniform distribution.

        :param state: the environment state
        :param observation: the corresponding observation
        :return: a tuple containing the actions and their probabilities
        """
        actions = np.arange(state.action_space.n)
        probabilities = np.ones((len(actions))) / len(actions)
        return actions, probabilities

    @staticmethod
    def random_available_policy(state, observation):
        """
            Choose actions from a uniform distribution over currently available actions only.

        :param state: the environment state
        :param observation: the corresponding observation
        :return: a tuple containing the actions and their probabilities
        """
        if hasattr(state, 'get_available_actions'):
            available_actions = state.get_available_actions()
        else:
            available_actions = np.arange(state.action_space.n)
        probabilities = np.ones((len(available_actions))) / len(available_actions)
        return available_actions, probabilities

    @staticmethod
    def preference_policy(state, observation, action_index, ratio=2):
        """
            Choose actions with a distribution over currently available actions that favors a preferred action.

            The preferred action probability is higher than others with a given ratio, and the distribution is uniform
            over the non-preferred available actions.
        :param state: the environment state
        :param observation: the corresponding observation
        :param action_index: the label of the preferred action
        :param ratio: the ratio between the preferred action probability and the other available actions probabilities
        :return: a tuple containing the actions and their probabilities
        """
        if hasattr(state, 'get_available_actions'):
            available_actions = state.get_available_actions()
        else:
            available_actions = np.arange(state.action_space.n)
        for i in range(len(available_actions)):
            if available_actions[i] == action_index:
                probabilities = np.ones((len(available_actions))) / (len(available_actions) - 1 + ratio)
                probabilities[i] *= ratio
                return available_actions, probabilities
        return MCTSAgent.random_available_policy(state, observation)


class MCTS(AbstractPlanner):
    """
       An implementation of Monte-Carlo Tree Search, with Upper Confidence Tree exploration.
    """
    def __init__(self, env, prior_policy, rollout_policy, config=None):
        """
            New MCTS instance.

        :param config: the mcts configuration. Use default if None.
        :param prior_policy: the prior policy used when expanding and selecting nodes
        :param rollout_policy: the rollout policy used to estimate the value of a leaf node
        """
        super().__init__(config)
        self.env = env
        self.prior_policy = prior_policy
        self.rollout_policy = rollout_policy
        if not self.config["horizon"]:
            self.config["episodes"], self.config["horizon"] = \
                OLOP.allocation(self.config["budget"], self.config["gamma"])

    @classmethod
    def default_config(cls):
        cfg = super(MCTS, cls).default_config()
        cfg.update({
            "temperature": 2 / (1 - cfg["gamma"]),
            "closed_loop": False
        })
        return cfg

    def reset(self):
        self.root = MCTSNode(parent=None, planner=self)

    def run(self, state, observation):
        """
            Run an iteration of Monte-Carlo Tree Search, starting from a given state

        :param state: the initial environment state
        :param observation: the corresponding observation
        """
        node = self.root
        total_reward = 0
        depth = 0
        terminal = False
        state.seed(self.np_random.randint(2**30))
        while depth < self.config['horizon'] and node.children and not terminal:
            action = node.sampling_rule(temperature=self.config['temperature'])
            observation, reward, terminal, _ = self.step(state, action)
            total_reward += self.config["gamma"] ** depth * reward
            node_observation = observation if self.config["closed_loop"] else None
            node = node.get_child(action, observation=node_observation)
            depth += 1

        if not node.children \
                and depth < self.config['horizon'] \
                and (not terminal or node == self.root):
            node.expand(self.prior_policy(state, observation))

        if not terminal:
            total_reward = self.evaluate(state, observation, total_reward, depth=depth)
        node.update_branch(total_reward)

    def evaluate(self, state, observation, total_reward=0, depth=0):
        """
            Run the rollout policy to yield a sample of the value of being in a given state.

        :param state: the leaf state.
        :param observation: the corresponding observation.
        :param total_reward: the initial total reward accumulated until now
        :param depth: the initial simulation depth
        :return: the total reward of the rollout trajectory
        """
        for h in range(depth, self.config["horizon"]):
            actions, probabilities = self.rollout_policy(state, observation)
            action = self.np_random.choice(actions, 1, p=np.array(probabilities))[0]
            observation, reward, terminal, _ = self.step(state, action)
            total_reward += self.config["gamma"] ** h * reward
            if np.all(terminal):
                break
        return total_reward

    def plan(self, state, observation):
        for i in range(self.config['episodes']):
            if (i+1) % 10 == 0:
                logger.debug('{} / {}'.format(i+1, self.config['episodes']))
            self.run(safe_deepcopy_env(state), observation)
        return self.get_plan()

    def step_planner(self, action):
        if self.config["step_strategy"] == "prior":
            self.step_by_prior(action)
        else:
            super().step_planner(action)

    def step_by_prior(self, action):
        """
            Replace the MCTS tree by its subtree corresponding to the chosen action, but also convert the visit counts
            to prior probabilities and before resetting them.

        :param action: a chosen action from the root node
        """
        self.step_by_subtree(action)
        self.root.convert_visits_to_prior_in_branch()


class MCTSNode(Node):
    K = 1.0
    """ The value function first-order filter gain"""

    def __init__(self, parent, planner, prior=1):
        super(MCTSNode, self).__init__(parent, planner)
        self.value = 0
        self.prior = prior

    def selection_rule(self):
        if not self.children:
            return None
        # Tie best counts by best value
        actions = list(self.children.keys())
        counts = Node.all_argmax([self.children[a].count for a in actions])
        return actions[max(counts, key=(lambda i: self.children[actions[i]].get_value()))]

    def sampling_rule(self, temperature=None):
        """
            Select an action from the node.
            - if exploration is wanted with some temperature, follow the selection strategy.
            - else, select the action with maximum visit count

        :param temperature: the exploration parameter, positive or zero
        :return: the selected action
        """
        if self.children:
            actions = list(self.children.keys())
            # Randomly tie best candidates with respect to selection strategy
            indexes = [self.children[a].selection_strategy(temperature) for a in actions]
            return actions[self.random_argmax(indexes)]
        else:
            return None

    def expand(self, actions_distribution):
        """
            Expand a leaf node by creating a new child for each available action.

        :param actions_distribution: the list of available actions and their prior probabilities
        """
        actions, probabilities = actions_distribution
        for i in range(len(actions)):
            if actions[i] not in self.children:
                self.children[actions[i]] = type(self)(self, self.planner, probabilities[i])

    def update(self, total_reward):
        """
            Update the visit count and value of this node, given a sample of total reward.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.count += 1
        self.value += self.K / self.count * (total_reward - self.value)

    def update_branch(self, total_reward):
        """
            Update the whole branch from this node to the root with the total reward of the corresponding trajectory.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.update(total_reward)
        if self.parent:
            self.parent.update_branch(total_reward)

    def get_child(self, action, observation=None):
        child = self.children[action]
        if observation is not None:
            if str(observation) not in child.children:
                child.children[str(observation)] = MCTSNode(parent=child, planner=self.planner, prior=0)
            child = child.children[str(observation)]
        return child

    def selection_strategy(self, temperature):
        """
            Select an action according to its value, prior probability and visit count.

        :param temperature: the exploration parameter, positive or zero.
        :return: the selected action with maximum value and exploration bonus.
        """
        if not self.parent:
            return self.get_value()

        # return self.value + temperature * self.prior * np.sqrt(np.log(self.parent.count) / self.count)
        return self.get_value() + temperature * len(self.parent.children) * self.prior/(self.count+1)

    def convert_visits_to_prior_in_branch(self, regularization=0.5):
        """
            For any node in the subtree, convert the distribution of all children visit counts to prior
            probabilities, and reset the visit counts.

        :param regularization: in [0, 1], used to add some probability mass to all children.
                               when 0, the prior is a Boltzmann distribution of visit counts
                               when 1, the prior is a uniform distribution
        """
        self.count = 0
        total_count = sum([(child.count+1) for child in self.children.values()])
        for child in self.children.values():
            child.prior = regularization*(child.count+1)/total_count + regularization/len(self.children)
            child.convert_visits_to_prior_in_branch()

    def get_value(self):
        return self.value


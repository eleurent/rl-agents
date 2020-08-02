import logging
import numpy as np
from functools import partial

from rl_agents.agents.common.factory import safe_deepcopy_env
from rl_agents.agents.tree_search.abstract import Node, AbstractTreeSearchAgent, AbstractPlanner

logger = logging.getLogger(__name__)

class MCTSDPWAgent(AbstractTreeSearchAgent):
    """
        An MCTS_DPW agent that uses Double Progressive Widenning for handling continuous
        stochastic MDPs.
    """
    def make_planner(self):
        rollout_policy = MCTSDPWAgent.policy_factory(self.config["rollout_policy"])
        return MCTSDPW(self.env,rollout_policy, self.config)

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "budget": 100,
            "horizon": None,
            "gamma": 0.95,
            "rollout_policy": {"type": "random_available"},
            "env_preprocessors": []
         })
        return config

    @staticmethod
    def policy_factory(policy_config):
        if policy_config["type"] == "random":
            return MCTSDPWAgent.random_policy
        elif policy_config["type"] == "random_available":
            return MCTSDPWAgent.random_available_policy
        elif policy_config["type"] == "preference":
            return partial(MCTSDPWAgent.preference_policy,
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
        return MCTSDPWAgent.random_available_policy(state, observation)


class MCTSDPW(AbstractPlanner):
    """
       An implementation of Monte-Carlo Tree Search with Upper Confidence Tree exploration
       and Double Progressive Widenning.
    """
    def __init__(self, env, rollout_policy, config=None):
        """
            New MCTSDPW instance.

        :param config: the mcts configuration. Use default if None.
        :param rollout_policy: the rollout policy used to estimate the value of a leaf node
        """
        super().__init__(config)
        self.env = env
        self.rollout_policy = rollout_policy
        if not self.config["horizon"]:
            self.config["episodes"], self.config["horizon"] = \
                self.allocation(self.config["budget"], self.config["gamma"])

    @staticmethod
    def horizon(episodes, gamma):
        return max(int(np.ceil(np.log(episodes) / (2 * np.log(1 / gamma)))), 1)

    @staticmethod
    def allocation(budget, gamma):
        """
            Allocate the computational budget into M episodes of fixed horizon L.
        """
        for episodes in range(1, int(budget)):
            if episodes * MCTSDPW.horizon(episodes, gamma) > budget:
                episodes = max(episodes - 1, 1)
                horizon = MCTSDPW.horizon(episodes, gamma)
                break
        else:
            raise ValueError("Could not split budget {} with gamma {}".format(budget, gamma))
        return episodes, horizon

    @classmethod
    def default_config(cls):
        cfg = super(MCTSDPW, cls).default_config()
        cfg.update({
            "temperature": 1,
            "closed_loop": False

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

        while depth < self.config['horizon'] and not terminal and decision_node.count != 0:


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

    def get_plan(self):
        """Only return the first action, the rest is conditioned on observations"""
        return self.root.selection_rule()

    def plan(self, state, observation):
        for i in range(self.config['episodes']):
            if (i+1) % 10 == 0:
                logger.debug('{} / {}'.format(i+1, self.config['episodes']))
            self.run(safe_deepcopy_env(state), observation)
        return self.get_plan()

    def step_planner(self, action):
        super().step_planner(action)

class DecisionNode(Node):
    k_action = 4 # pw parameters
    alpha_action = 0.3 # pw parameters
    K = 1.0
    """ The value function first-order filter gain"""

    def __init__(self, parent, planner):
        super(DecisionNode, self).__init__(parent, planner)
        self.value = 0
        self.depth = 0 if parent is None else parent.depth + 1

    def selection_rule(self):
        if not self.children:
            return None
        # Tie best counts by best value
        actions = list(self.children.keys())
        counts = Node.all_argmax([self.children[a].count for a in actions])
        return actions[max(counts, key=(lambda i: self.children[actions[i]].get_value()))]

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

    def update(self, total_reward):
        """
            Update the visit count and value of this node, given a sample of total reward.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.count += 1
        self.value += self.K / self.count * (total_reward - self.value)

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
            ucb_val = self.children[a].get_value() +  0.5 * np.sqrt(np.log(self.count / (self.children[a].count)))
            indexes.append(ucb_val)

        action = actions[self.random_argmax(indexes)]
        return self.children[action], action

    def get_value(self):
        return self.value

class ChanceNode(Node):
    K = 1.0
    # state progressive widenning parameters
    k_state = 1
    alpha_state = 0.3

    def __init__(self, parent, planner):
        assert parent is not None
        super(ChanceNode, self).__init__(parent, planner)
        self.value = 0
        self.depth = parent.depth

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

    def update(self, total_reward):
        """
            Update the visit count and value of this node, given a sample of total reward.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.count += 1
        self.value += self.K / self.count * (total_reward - self.value)

    def backup_to_root(self, total_reward):
        """
            Update the whole branch from this node to the root with the total reward of the corresponding trajectory.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        assert self.children
        assert self.parent
        self.update(total_reward)
        self.parent.backup_to_root(total_reward)

    def get_value(self):
        return self.value

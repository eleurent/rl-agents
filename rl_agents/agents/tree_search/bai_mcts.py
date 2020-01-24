import logging
import numpy as np

from rl_agents.agents.tree_search.ugape_mcts import UGapEMCTSNode, UgapEMCTS, UgapEMCTSAgent
from rl_agents.utils import max_expectation_under_constraint

logger = logging.getLogger(__name__)


class BaiMCTSAgent(UgapEMCTSAgent):
    """
        An agent that uses BAI to plan a sequence of actions in an MDP.
    """
    def make_planner(self):
        return BaiMCTS(self.env, self.config)

    def step(self, actions):
        """
            Handle receding horizon mechanism
        :return: whether a replanning is required
        """
        replanning_required = self.remaining_horizon == 0  # Cannot check remaining actions here
        if replanning_required:
            self.remaining_horizon = self.config["receding_horizon"] - 1
            self.planner.step_by_reset()
        else:
            self.remaining_horizon -= 1
            self.planner.step(actions)

            # Check for remaining children here instead
            if self.planner.root.children:
                self.previous_actions.extend(self.planner.get_plan())
            else:  # After stepping the transition in the tree, the subtree is empty
                replanning_required = True
                self.planner.step_by_reset()

        return replanning_required

    def record(self, state, action, reward, next_state, done, info):
        self.planner.next_observation = next_state


class BaiMCTS(UgapEMCTS):
    """
       Best-Arm Identification MCTS.
    """
    def __init__(self, env, config=None):
        super().__init__(env, config)
        self.next_observation = None

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update(
            {
                "upper_bound": {
                    "transition_threshold": "0.1*np.log(time)"
                },
                "max_next_states_count": 1
            }
        )
        return cfg

    def make_root(self):
        if "horizon" not in self.config:
            self.allocate_budget()
        root = BaiStateNode(parent=None, planner=self)
        return root

    def run(self, state):
        """
            Run a BAI MCTS episode.

        :param state: the initial environment state
        """
        # We need randomness
        state.seed(self.np_random.randint(10**10))
        best, challenger = None, None
        if self.root.children:
            logger.debug(" / ".join(["a{} ({}): [{:.3f}, {:.3f}]".format(k, n.count, n.value_lower, n.value)
                                     for k, n in self.root.children.items()]))

        # Follow selection policy, expand tree if needed, collect rewards and update confidence bounds.
        state_node = self.root
        for h in range(self.config["horizon"]):
            # Select action
            if not state_node.children:  # Break ties at leaves
                action = self.np_random.randint(state.action_space.n) \
                    if self.config["continuation_type"] == "uniform" else 0
            elif state_node == self.root:  # Run BAI at the root
                selected_child, best, challenger = self.root.best_arm_identification_selection()
                action = next(selected_child.path())
            else:  # Run UCB elsewhere
                action, _ = max([child for child in state_node.children.items()], key=lambda c: c[1].value)

            # Perform transition
            action_node, action = state_node.get_child(action, state)
            observation, reward, done, _ = state.step(action)
            state_node = action_node.get_child(observation)

            # Update local statistics
            action_node.update(np.nan, False)
            state_node.update(reward, done)

        # Backup global statistics
        state_node.backup_to_root()
        return best, challenger

    def step(self, actions):
        """
            Update the planner tree when the agent performs an action and observes the next state
        :param actions: a sequence of actions to follow from the root node
        """
        if self.config["step_strategy"] == "reset":
            self.step_by_reset()
        elif self.config["step_strategy"] == "subtree":
            if actions:
                self.step_by_subtree(actions[0])
                self.step_by_subtree(str(self.next_observation))  # Step to the observed next state
            else:
                self.step_by_reset()
        else:
            logger.warning("Unknown step strategy: {}".format(self.config["step_strategy"]))
            self.step_by_reset()

    def get_plan(self):
        """Only return the first action, the rest is conditioned on observations"""
        return [self.root.selection_rule()]


class BaiStateNode(UGapEMCTSNode):
    def __init__(self, parent, planner):
        super().__init__(parent, planner)

    def expand(self, state):
        if state is None:
            raise Exception("The state should be set before expanding a node")
        try:
            actions = state.get_available_actions()
        except AttributeError:
            actions = range(state.action_space.n)
        for action in actions:
            self.children[action] = BaiActionNode(self, self.planner)

    def get_child(self, action, state):
        if not self.children:
            self.expand(state)
        if action not in self.children:  # Default action may not be available
            action = list(self.children.keys())[0]  # Pick first available action instead
        return self.children[action], action

    def backup_to_root(self):
        """
            Bellman V(s) = max_a Q(s,a)
        """
        if self.children:
            self.value = np.amax([child.value for child in self.children.values()])
            self.value_lower = np.amax([child.value_lower for child in self.children.values()])
        else:
            assert self.depth == self.planner.config["horizon"]
            self.value = 0  # Maybe count bound over r(H..inf) ?
            self.value_lower = 0  # Maybe count bound over r(H..inf) ?
        if self.parent:
            self.parent.backup_to_root()


class BaiActionNode(UGapEMCTSNode):
    def __init__(self, parent, planner):
        assert parent is not None
        super().__init__(parent, planner)
        self.depth = parent.depth
        gamma = self.planner.config["gamma"]
        self.value = (1 - gamma ** (self.planner.config["horizon"] - self.depth)) / (1 - gamma)
        self.p_hat, self.p_plus, self.p_minus = None, None, None
        delattr(self, 'cumulative_reward')
        delattr(self, 'mu_ucb')
        delattr(self, 'mu_lcb')

    def update(self, reward, done):
        self.count += 1

    def expand(self, state):
        # Generate placeholder nodes
        for i in range(self.planner.config["max_next_states_count"]):
            self.children["placeholder_{}".format(i)] = BaiStateNode(self, self.planner)

    def get_child(self, observation, hash=False):
        if not self.children:
            self.expand(None)
        import hashlib
        state_id = hashlib.sha1(str(observation).encode("UTF-8")).hexdigest()[:5] if hash else str(observation)
        if state_id not in self.children:
            # Assign the first available placeholder to the observation
            for i in range(self.planner.config["max_next_states_count"]):
                if "placeholder_{}".format(i) in self.children:
                    self.children[state_id] = self.children.pop("placeholder_{}".format(i))
                    break
            else:
                raise ValueError("No more placeholder nodes available, we observed more next states than "
                                 "the 'max_next_states_count' config")
        return self.children[state_id]

    def backup_to_root(self):
        """
            Bellman Q(s,a) = r(s,a) + gamma E_s' V(s')
        """
        assert self.children
        assert self.parent
        gamma = self.planner.config["gamma"]
        children = list(self.children.values())
        u_next = np.array([c.mu_ucb + gamma * c.value for c in children])
        l_next = np.array([c.mu_lcb + gamma * c.value_lower for c in children])
        self.p_hat = np.array([child.count for child in children]) / self.count
        threshold = self.transition_threshold() / self.count

        self.p_plus = max_expectation_under_constraint(u_next, self.p_hat, threshold)
        self.p_minus = max_expectation_under_constraint(-l_next, self.p_hat, threshold)
        self.value = self.p_plus @ u_next
        self.value_lower = self.p_minus @ l_next
        self.parent.backup_to_root()

    def transition_threshold(self):
        horizon = self.planner.config["horizon"]
        actions = self.planner.env.action_space.n
        confidence = self.planner.config["confidence"]
        count = self.count
        time = self.planner.config["episodes"]
        return eval(self.planner.config["upper_bound"]["transition_threshold"])

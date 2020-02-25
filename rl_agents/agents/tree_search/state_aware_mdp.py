import numpy as np
import logging
from rl_agents.agents.common.factory import safe_deepcopy_env
from rl_agents.agents.tree_search.abstract import Node, AbstractTreeSearchAgent, AbstractPlanner

logger = logging.getLogger(__name__)


class StateAwareMDPAgent(AbstractTreeSearchAgent):
    def make_planner(self):
        return StateAwareMDPAgent(self.env, self.config)


class StateAwareMDPPlanner(AbstractPlanner):
    def __init__(self, env, config=None):
        super().__init__(config)
        self.env = env
        self.nodes = {}
        self.sinks = {}

    def run(self, observation):
        while str(observation) not in self.sinks:
            actions, action_values = self.nodes[str(observation)].action_values()
            reward, observation = actions[np.argmax(action_values)]
        self.expand(observation)
        self.value_iteration()

    def expand(self, observation):
        node = self.nodes[str(observation)]
        try:
            actions = node.state.get_available_actions()
        except AttributeError:
            actions = range(node.state.action_space.n)
        for action in actions:
            next_state = safe_deepcopy_env(node.state)
            next_observation, reward, done, _ = next_state.step(action)
            # Add new state node
            if str(next_observation) not in self.nodes:
                self.nodes[str(next_observation)] = StateNode(self, next_state)
            node.transitions[action] = reward, next_observation

    def value_iteration(self, eps=1e-2):
        for _ in range(int(3 / np.log(1 / self.config["gamma"]))):
            delta = 0
            for node in self.nodes.values():
                _, action_values = node.action_values()
                new_bound = np.amax(action_values)
                delta = max(delta, node.value_upper_bound - new_bound)
                node.value_upper_bound = new_bound
            if delta < eps:
                break

    def plan(self, state, observation):
        if str(observation) not in self.nodes:
            self.sinks[str(observation)] = StateNode(self, state)
        for epoch in np.arange(self.config["budget"] // state.action_space.n):
            logger.debug("Expansion {}/{}".format(epoch + 1, self.config["budget"] // state.action_space.n))
            self.run(observation)

        return self.get_plan()


class StateNode(Node):
    def __init__(self, planner, state):
        super().__init__(parent=None, planner=planner)
        self.state = state
        self.value_upper_bound = 1/(1 - self.planner.config["gamma"])
        self.transitions = {}

    def selection_rule(self):
        pass

    def action_values(self):
        gamma = self.planner.config["gamma"]
        actions, transitions = self.transitions.items()
        action_values = [reward + gamma * self.planner.nodes[obs].value_upper_bound for reward, obs in transitions]
        return actions, action_values

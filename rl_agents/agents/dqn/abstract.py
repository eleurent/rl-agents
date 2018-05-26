from abc import ABC, abstractmethod

import numpy as np

from rl_agents.agents.abstract import AbstractStochasticAgent
from rl_agents.agents.exploration.epsilon_greedy import EpsilonGreedy
from rl_agents.agents.utils import ReplayMemory
from rl_agents.configuration import Config, Configurable


class DQNAgent(AbstractStochasticAgent, ABC, Configurable):
    def __init__(self, env, config=None):
        self.env = env
        self.config = self.default_config()
        if config:
            self.config.update(config)
        self.config.num_states = env.observation_space.shape[0]
        self.config.num_actions = env.action_space.n
        self.config.all_layers = [self.config.num_states] + self.config.layers + [self.config.num_actions]
        self.memory = ReplayMemory(self.config)
        self.exploration_policy = self.config.exploration.method(self.config.exploration, self.env.action_space)
        self.previous_state = None

    @classmethod
    def default_config(cls):
        return Config(layers=[100, 100],
                      memory_capacity=5000,
                      batch_size=32,
                      gamma=0.99,
                      exploration=Config(method=EpsilonGreedy,
                                         epsilon=[1.0, 0.01],
                                         epsilon_tau=5000),
                      target_update=1)

    def action_distribution(self, state):
        self.previous_state = state
        values = self.get_state_action_values(state)
        self.exploration_policy.update(values)
        return self.exploration_policy.get_distribution()

    def act(self, state):
        self.previous_state = state
        values = self.get_state_action_values(state)
        self.exploration_policy.update(values)
        return self.exploration_policy.sample()

    @abstractmethod
    def get_batch_state_values(self, states):
        """
        Get the state values of several states
        :param states: [s1; ...; sN] an array of states
        :return: values, actions:
                 - [V1; ...; VN] the array of the state values for each state
                 - [a1*; ...; aN*] the array of corresponding optimal action indexes for each state
        """
        raise NotImplementedError()

    @abstractmethod
    def get_batch_state_action_values(self, states):
        """
        Get the state-action values of several states
        :param states: [s1; ...; sN] an array of states
        :return: values:[[Q11, ..., Q1n]; ...] the array of all action values for each state
        """
        raise NotImplementedError()

    def get_state_value(self, state):
        """
        :param state: s, an environment state
        :return: V, its state-value
        """
        values, actions = self.get_batch_state_values([state])
        return values[0], actions[0]

    def get_state_action_values(self, state):
        """
        :param state: s, an environment state
        :return: [Q(a1,s), ..., Q(an,s)] the array of its action-values for each actions
        """
        return self.get_batch_state_action_values([state])[0]

    def seed(self, seed=None):
        return self.exploration_policy.seed(seed)

    def reset(self):
        pass

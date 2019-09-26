import abc
import copy
import torch
import numpy as np

from rl_agents.agents.budgeted_ftq.greedy_policy import optimal_mixture, pareto_frontier_at
from rl_agents.agents.common.exploration.epsilon_greedy import EpsilonGreedy
from rl_agents.agents.common.utils import sample_simplex


class BudgetedPolicy:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def execute(self, state, beta):
        pass


class EpsilonGreedyBudgetedPolicy(BudgetedPolicy):
    def __init__(self, pi_greedy, pi_random, config, np_random=np.random):
        super().__init__()
        self.pi_greedy = pi_greedy
        self.pi_random = pi_random
        self.config = config
        self.np_random = np_random
        self.time = 0

    def execute(self, state, beta):
        epsilon = self.config['final_temperature'] + (self.config['temperature'] - self.config['final_temperature']) * \
                       np.exp(- self.time / self.config['tau'])
        self.time += 1

        if self.np_random.random_sample() > epsilon:
            return self.pi_greedy.execute(state, beta)
        else:
            return self.pi_random.execute(state, beta)

    def set_time(self, time):
        self.time = time


class RandomBudgetedPolicy(BudgetedPolicy):
    def __init__(self, n_actions, np_random=np.random):
        self.n_actions = n_actions
        self.np_random = np_random

    def execute(self, state, beta):
        action_probs = self.np_random.rand(self.n_actions)
        action_probs /= np.sum(action_probs)
        budget_probs = sample_simplex(coeff=action_probs, bias=beta, min_x=0, max_x=1, np_random=self.np_random)
        action = self.np_random.choice(a=range(self.n_actions), p=action_probs)
        beta = budget_probs[action]
        return action, beta


class PytorchBudgetedFittedPolicy(BudgetedPolicy):
    def __init__(self, network, betas_for_discretisation, device, hull_options, clamp_qc=None, np_random=np.random):
        self.betas_for_discretisation = betas_for_discretisation
        self.device = device
        self.network = None
        self.hull_options = hull_options
        self.clamp_qc = clamp_qc
        self.np_random = np_random
        self.network = network

    def load_network(self, network_path):
        self.network = torch.load(network_path, map_location=self.device)

    def set_network(self, network):
        self.network = copy.deepcopy(network)

    def execute(self, state, beta):
        mixture, _ = self.greedy_policy(state, beta)
        choice = mixture.sup if self.np_random.rand() < mixture.probability_sup else mixture.inf
        return choice.action, choice.budget

    def greedy_policy(self, state, beta):
        with torch.no_grad():
            hull = pareto_frontier_at(
                state=torch.tensor([state], device=self.device, dtype=torch.float32),
                value_network=self.network,
                betas=self.betas_for_discretisation,
                device=self.device,
                hull_options=self.hull_options,
                clamp_qc=self.clamp_qc)
        mixture = optimal_mixture(hull[0], beta)
        return mixture, hull

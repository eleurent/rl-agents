import abc
import copy
import importlib
import torch
import numpy as np

from rl_agents.agents.budgeted_ftq.budgeted_utils import optimal_mixture, compute_convex_hull
from rl_agents.agents.utils import sample_simplex


class Policy:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def execute(self, state, beta):
        pass

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class EpsilonGreedyPolicy(Policy):
    def __init__(self, pi_greedy, pi_random, epsilon, np_random=np.random, **kwargs):
        self.pi_greedy = pi_greedy
        self.pi_random = pi_random
        self.epsilon = epsilon
        self.np_random = np_random

    def execute(self, state, beta):
        if self.np_random.random_sample() > self.epsilon:
            return self.pi_greedy.execute(state, beta)
        else:
            return self.pi_random.execute(state, beta)

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["pi_greedy"] = policy_factory(config["pi_greedy"])
        if config["pi_random"]:
            config["pi_random"] = policy_factory(config["pi_random"])
        return super(EpsilonGreedyPolicy, cls).from_config(config)


class RandomBudgetedPolicy(Policy):
    def __init__(self, n_actions, np_random=np.random, **kwargs):
        self.n_actions = n_actions
        self.np_random = np_random

    def execute(self, state, beta):
        action_probs = self.np_random.rand(self.n_actions)
        action_probs /= np.sum(action_probs)
        budget_probs = sample_simplex(coeff=action_probs, bias=beta, min_x=0, max_x=1, np_random=self.np_random)
        action = self.np_random.choice(a=range(self.n_actions), p=action_probs)
        beta = budget_probs[action]
        return action, beta


class PytorchBudgetedFittedPolicy(Policy):
    def __init__(self, network, betas_for_discretisation, device, hull_options, clamp_qc=None,
                 np_random=np.random, **kwargs):
        self.betas_for_discretisation = betas_for_discretisation
        self.device = device
        self.network = None
        self.hull_options = hull_options
        self.clamp_qc = clamp_qc
        self.np_random = np_random
        self.network = network
        if isinstance(network, str):
            self.load_network(network)

    def load_network(self, network_path):
        self.network = torch.load(network_path, map_location=self.device)

    def set_network(self, network):
        self.network = copy.deepcopy(network)

    def execute(self, state, beta):
        with torch.no_grad():
            hull, _, _, _ = compute_convex_hull(
                state=torch.tensor([state], device=self.device, dtype=torch.float32),
                value_network=self.network,
                betas=self.betas_for_discretisation,
                device=self.device,
                hull_options=self.hull_options,
                clamp_qc=self.clamp_qc)
            mixture = optimal_mixture(hull, beta)
            choice = mixture.sup if self.np_random.rand() < mixture.probability_sup else mixture.inf
            return choice.action, choice.budget


def policy_factory(config, dynamically=False):
    if "__class__" in config:
        if dynamically:
            path = config['__class__'].split("'")[1]
            module_name, class_name = path.rsplit(".", 1)
            policy_class = getattr(importlib.import_module(module_name), class_name)
            return policy_class.from_config(config)
        elif config['__class__'] == repr(RandomBudgetedPolicy):
            return RandomBudgetedPolicy.from_config(config)
        elif config['__class__'] == repr(PytorchBudgetedFittedPolicy):
            return PytorchBudgetedFittedPolicy.from_config(config)
    else:
        raise ValueError("The configuration should specify the policy __class__")

import abc
import copy
import importlib
import torch
import numpy as np

from rl_agents.agents.budgeted_ftq.budgeted_utils import optimal_mixture, convex_hull


class Policy:
    __metaclass__ = abc.ABCMeta

    # must return Q function (s,a) -> double
    @abc.abstractmethod
    def execute(self, s, info):
        pass

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class EpsilonGreedyPolicy(Policy):
    def __init__(self, pi_greedy, pi_random, epsilon, **kwargs):
        self.pi_greedy = pi_greedy
        self.pi_random = pi_random
        self.epsilon = epsilon

    def execute(self, s, info):
        if np.random.random_sample() > self.epsilon:
            return self.pi_greedy.execute(s, info)
        else:
            return self.pi_random.execute(s, info)

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config["pi_greedy"] = policy_factory(config["pi_greedy"])
        if config["pi_random"]:
            config["pi_random"] = policy_factory(config["pi_random"])
        return super(EpsilonGreedyPolicy, cls).from_config(config)


class RandomBudgetedPolicy(Policy):
    def __init__(self, **kwargs):
        pass

    def execute(self, s, action_mask, info):
        beta = info["beta"]
        actions = []
        for i in range(len(action_mask)):
            if action_mask[i] == 0:
                actions.append(i)
        action_repartition = np.random.random(len(actions))
        action_repartition /= np.sum(action_repartition)
        budget_repartion = generate_random_point_on_simplex_not_uniform(
            coeff=action_repartition,
            bias=beta,
            min_x=0,
            max_x=1)
        index = np.random.choice(a=range(len(actions)),
                                 p=action_repartition)
        a = actions[index]
        beta_ = budget_repartion[index]
        info["beta"] = beta_
        return a, info


class PytorchBudgetedFittedPolicy(Policy):
    def __init__(self, network_path, betas_for_discretisation, device, hull_options, clamp_qc=None,
                 **kwargs):
        self.betas_for_discretisation = betas_for_discretisation
        self.device = device
        self.network = None
        self.hull_options = hull_options
        self.clamp_Qc = clamp_qc
        if network_path:
            self.load_network(network_path)

    def execute(self, s, info):
        a, beta = self.pi(s, info["beta"])
        info["beta"] = beta
        return a, info

    def load_network(self, network_path):
        self.network = torch.load(network_path, map_location=self.device)

    def set_network(self, network):
        self.network = copy.deepcopy(network)

    def pi(self, state, beta):
        with torch.no_grad():
            hull = convex_hull(s=torch.tensor([state], device=self.device, dtype=torch.float32),
                               Q=self.network,
                               id="run_" + str(state), disp=False,
                               betas=self.betas_for_discretisation,
                               device=self.device,
                               hull_options=self.hull_options,
                               clamp_Qc=self.clamp_Qc)
            opt, _ = optimal_mixture(hull, beta)
            rand = np.random.random()
            a = opt.action_inf if rand < opt.proba_inf else opt.action_sup
            b = opt.budget_inf if rand < opt.proba_inf else opt.budget_sup
            return a, b


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

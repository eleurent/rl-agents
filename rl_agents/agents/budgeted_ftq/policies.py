import abc
import copy
import importlib
import torch
import numpy as np

from rl_agents.agents.budgeted_ftq.budgeted_utils import optimal_pia_pib, convex_hull


class Policy:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def reset(self):
        pass

    # must return Q function (s,a) -> double
    @abc.abstractmethod
    def execute(self, s, action_mask, info):
        pass

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class RandomBudgetedPolicy(Policy):
    def __init__(self, **kwargs):
        pass

    def reset(self):
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
        return a, False, info


class PytorchBudgetedFittedPolicy(Policy):
    def __init__(self, env, feature_str, network_path, betas_for_discretisation, device, hull_options, clamp_Qc=False,
                 **kwargs):
        self.env = env
        self.betas_for_discretisation = betas_for_discretisation
        self.device = device
        self.network = None
        self.hull_options = hull_options
        self.clamp_Qc = clamp_Qc
        if network_path:
            self.load_network(network_path)

    def reset(self):
        pass

    def execute(self, s, action_mask, info):
        a, beta = self.pi(s, info["beta"], action_mask)
        info["beta"] = beta
        return a, False, info

    def load_network(self, network_path):
        self.network = torch.load(network_path, map_location=self.device)

    def set_network(self, network):
        self.network = copy.deepcopy(network)

    def pi(self, state, beta, action_mask):
        with torch.no_grad():
            if not type(action_mask) == type(np.zeros(1)):
                action_mask = np.asarray(action_mask)
            hull = convex_hull(s=torch.tensor([state], device=self.device, dtype=torch.float32),
                               Q=self.network,
                               action_mask=action_mask,
                               id="run_" + str(state), disp=False,
                               betas=self.betas_for_discretisation,
                               device=self.device,
                               hull_options=self.hull_options,
                               clamp_Qc=self.clamp_Qc)
            opt, _ = optimal_pia_pib(beta=beta, hull=hull, statistic={})
            rand = np.random.random()
            a = opt.id_action_inf if rand < opt.proba_inf else opt.id_action_sup
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

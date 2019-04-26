# coding=utf-8
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt
import numpy as np
import logging
import os

from rl_agents.agents.abstract import AbstractAgent
from rl_agents.agents.budgeted_ftq.bftq import BudgetedFittedQ
from rl_agents.agents.budgeted_ftq.models import NetBFTQ
from rl_agents.agents.budgeted_ftq.policies import PytorchBudgetedFittedPolicy, RandomBudgetedPolicy, \
    EpsilonGreedyPolicy


class BFTQAgent(AbstractAgent):
    def __init__(self, env, config=None):
        super(BFTQAgent, self).__init__(config)

        # Network
        net = NetBFTQ(size_state=env.observation_space.shape, n_actions=env.action_space.n,
                **bftq_net_params)
        # Budgeted policy
        self.bftq = BudgetedFittedQ(
            policy_network=net,
            config=config)

        self.exploration_policy = {
            "__class__": repr(EpsilonGreedyPolicy),
            "pi_greedy": {"__class__": repr(RandomBudgetedPolicy)},
            "pi_random": {"__class__": repr(RandomBudgetedPolicy)},
            "epsilon": self.config["epsilon"],
            "hull_options": self.config["hull_options"],
            "clamp_Qc": self.config["clamp_Qc"]
        }

    @classmethod
    def default_config(cls):
        return {
            "gamma": 0.9,
            "gamma_c": 0.9,
            "betas_for_duplication": np.arange(0, 1, 0.1),
            "betas_for_discretisation": np.arange(0, 1, 0.1),
            "optimizer": None,
            "loss_function": "l2",
            "loss_function_c": "l2",
            "max_ftq_epoch": np.inf,
            "max_nn_epoch": 1000,
            "learning_rate": 0.001,
            "weight_decay": 0.001,
            "delta_stop": 0.,
            "clamp_Qc": None,
            "nn_loss_stop_condition": 0.0,
            "weights_losses": [1., 1.],
            "split_batches": 4,
            "cpu_processes": 4,
            "batch_size_experience_replay": 50,
            "device": None,
            "hull_options": None,
            "reset_policy_each_ftq_epoch": True,
        }

    def act(self):
        action = self.explore()
        batch_complete = False
        if batch_complete:
            self.fit()
        return action

    def explore(self):
        # Execute eps greedy policy
        return 0

    def record(self, state, action, reward, next_state, done, info):
        self.bftq.push(state, action, reward, next_state, done, info["constraint"])

    def fit(self):
            # Fit model
            self.bftq.reset()
            self.bftq.fit()
            network_path = self.bftq.save_policy()

            # Update greedy policy
            self.exploration_policy["pi_greedy"] = {
                "__class__": repr(PytorchBudgetedFittedPolicy),
                "network_path": network_path,
                "betas_for_discretisation": self.bftq.betas_for_discretisation,
                "device": self.bftq.device,
                "hull_options": self.config["hull_options"],
                "clamp_Qc": self.config["clamp_Qc"]
            }

    def reset(self):
        pass

    def seed(self, seed=None):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

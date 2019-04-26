# coding=utf-8
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt
import numpy as np
import logging
import os

from rl_agents.agents.abstract import AbstractAgent
from rl_agents.agents.budgeted_ftq.bftq import BudgetedFittedQ


class BFTQAgent(AbstractAgent):
    def __init__(self, env, config=None):
        super(BFTQAgent, self).__init__(config)


        # Network
        net = NetBFTQ(size_state=env.observation_space.shape, n_actions=env.action_space.n,
                **bftq_net_params)
        # Budgeted policy
        self.bftq = BudgetedFittedQ(
            gamma=config["gamma"],
            gamma_c=config["gamma_c"],
            policy_network=net,
            device=config["device"],
            cpu_processes=config["cpu_processes"],
            split_batches=config["split_batches"],
            hull_options=config["hull_options"],
            **config["bftq_params"])

        exploration_policy = {
            "__class__": repr(EpsilonGreedyPolicy),
            "pi_greedy": {"__class__": repr(RandomBudgetedPolicy)},
            "pi_random": {"__class__": repr(RandomBudgetedPolicy)},
            "epsilon": decays[0],
            "hull_options": general["hull_options"],
            "clamp_Qc": bftq_params["clamp_Qc"]
        }

    @classmethod
    def default_config(cls):
        return dict(
            device=None,
            gamma=0.9,
            gamma_c=0.9,
            hull_options=None,
            bftq_params=None,
            split_batches=4
        )

    def act(self):
        action = self.explore()
        if batch_complete:
            self.fit()
        return action

    def explore(self):
        # Execute eps greedy policy

    def collect(self, state, action, next_state, reward, info, done):
            trajs = []
            # Fill memory
            transition = ()

    def fit(self):
            # Fit model
            logging.info("[BATCH={}][learning bftq pi greedy] #samples={} #traj={}"
                        .format(batch, len(transition_bftq), i_traj))

            bftq.reset(True)
            bftq.workspace = workspace / "batch={}".format(batch)
            makedirs(bftq.workspace)
            q = bftq.fit(transition_bftq)

            # Save policy
            network_path = bftq.save_policy()

            # Update greedy policy
            pi_epsilon_greedy_config["pi_greedy"] = {
                "__class__": repr(PytorchBudgetedFittedPolicy),
                "feature_str": feature_str,
                "network_path": network_path,
                "betas_for_discretisation": bftq.betas_for_discretisation,
                "device": bftq.device,
                "hull_options": general["hull_options"],
                "clamp_Qc": bftq_params["clamp_Qc"]
            }

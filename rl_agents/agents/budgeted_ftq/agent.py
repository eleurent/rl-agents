import numpy as np
import torch
from gym import logger
from gym.utils import seeding

from rl_agents.agents.common.abstract import AbstractAgent
from rl_agents.agents.budgeted_ftq.bftq import BudgetedFittedQ
from rl_agents.agents.budgeted_ftq.models import NetBFTQ
from rl_agents.agents.budgeted_ftq.policies import PytorchBudgetedFittedPolicy, RandomBudgetedPolicy, \
    EpsilonGreedyBudgetedPolicy


class BFTQAgent(AbstractAgent):
    def __init__(self, env, config=None):
        super(BFTQAgent, self).__init__(config)
        if not self.config["epochs"]:
            self.config["epochs"] = int(1 / np.log(1 / self.config["gamma"]))
        self.env = env
        self.np_random = None
        self.bftq = None
        self.exploration_policy = None
        self.beta = 0

    @classmethod
    def default_config(cls):
        return {
            "gamma": 0.9,
            "gamma_c": 0.9,
            "epochs": None,
            "delta_stop": 0.,
            "memory_capacity": 10000,
            "betas_for_duplication": "np.arange(0, 1, 0.1)",
            "betas_for_discretisation": "np.arange(0, 1, 0.1)",
            "exploration": {
                "temperature": 1.0,
                "final_temperature": 0.1,
                "tau": 5000
            },
            "optimizer": {
                "type": "ADAM",
                "learning_rate": 1e-3,
                "weight_decay": 1e-3
            },
            "loss_function": "l2",
            "loss_function_c": "l2",
            "regression_epochs": 500,
            "clamp_qc": None,
            "nn_loss_stop_condition": 0.0,
            "weights_losses": [1., 1.],
            "split_batches": 1,
            "cpu_processes": 1,
            "samples_per_batch": 500,
            "device": "cpu",
            "hull_options": {
                "decimals": None,
                "qhull_options": "",
                "remove_duplicated_points": True,
                "library": "scipy"
            },
            "reset_network_each_epoch": True,
            "network": {
                "beta_encoder_type": "LINEAR",
                "size_beta_encoder": 10,
                "activation_type": "RELU",
                "reset_type": "XAVIER",
                "layers": [
                    64,
                    64
                ]
            }
        }

    def act(self, state):
        action = self.explore(state)
        minibatch_complete = self.bftq.memory and len(self.bftq.memory) % self.config["samples_per_batch"] == 0
        if minibatch_complete:
            self.fit()
        return action

    def explore(self, state):
        """
            Run the exploration policy to pick actions and budgets
        """
        action, self.beta = self.exploration_policy.execute(state, self.beta)
        return action

    def record(self, state, action, reward, next_state, done, info):
        """
            Record a transition to update the BFTQ policy
        """
        self.bftq.push(state, action, reward, next_state, done, info["cost"])

        # At the end of an episode, pick a budget for the next one
        if done:
            self.beta = self.np_random.rand()

    def fit(self):
        logger.info("Run BFTQ on current batch")

        # Fit model
        self.bftq.reset()
        network = self.bftq.run()
        # Update greedy policy
        self.exploration_policy.pi_greedy.set_network(network)

    def reset(self):
        if not self.bftq:  # Do not reset the bftq replay memory at each episode
            if not self.np_random:
                raise Exception("Seed the agent before reset.")
            network = NetBFTQ(size_state=np.prod(self.env.observation_space.shape),
                              n_actions=self.env.action_space.n,
                              **self.config["network"])
            self.bftq = BudgetedFittedQ(policy_network=network, config=self.config)
            self.exploration_policy = EpsilonGreedyBudgetedPolicy(
                pi_greedy=PytorchBudgetedFittedPolicy(
                    network,
                    self.bftq.betas_for_discretisation,
                    self.bftq.device,
                    self.config["hull_options"],
                    self.config["clamp_qc"],
                    np_random=self.np_random
                ),
                pi_random=RandomBudgetedPolicy(n_actions=self.env.action_space.n, np_random=self.np_random),
                config=self.config["exploration"],
                np_random=self.np_random
            )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        torch.manual_seed(seed & ((1 << 63) - 1))  # torch seeds are int64
        return [seed]

    def save(self, filename):
        self.bftq.save_network(filename)

    def load(self, filename):
        self.bftq.load_network(filename)

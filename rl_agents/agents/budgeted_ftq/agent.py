import logging
import numpy as np
import torch
from gym.utils import seeding

from rl_agents.agents.common.abstract import AbstractAgent
from rl_agents.agents.budgeted_ftq.bftq import BudgetedFittedQ
from rl_agents.agents.budgeted_ftq.models import BudgetedMLP
from rl_agents.agents.budgeted_ftq.policies import PytorchBudgetedFittedPolicy, RandomBudgetedPolicy, \
    EpsilonGreedyBudgetedPolicy
from rl_agents.agents.common.utils import load_pytorch

logger = logging.getLogger(__name__)


class BFTQAgent(AbstractAgent):
    def __init__(self, env, config=None):
        super(BFTQAgent, self).__init__(config)
        self.batched = True
        if not self.config["epochs"]:
            self.config["epochs"] = int(1 / np.log(1 / self.config["gamma"]))
        self.env = env
        self.np_random = None
        self.bftq = None
        self.exploration_policy = None
        self.beta = self.previous_beta = 0
        self.training = True
        self.previous_state = None

        load_pytorch()

    @classmethod
    def default_config(cls):
        return {
            "gamma": 0.9,
            "gamma_c": 0.9,
            "epochs": None,
            "delta_stop": 0.,
            "memory_capacity": 10000,
            "beta": 0,
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
            "processes": 1,
            "samples_per_batch": 500,
            "device": "cuda:best",
            "hull_options": {
                "decimals": None,
                "qhull_options": "",
                "remove_duplicates": False,
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
        """
            Run the exploration policy to pick actions and budgets
        """
        # TODO: Choose the initial budget for the next episode and not at each step
        self.beta = self.np_random.rand() if self.training else self.config["beta"]

        state = state.flatten()
        self.previous_state, self.previous_beta = state, self.beta
        action, self.beta = self.exploration_policy.execute(state, self.beta)
        return action

    def record(self, state, action, reward, next_state, done, info):
        """
            Record a transition to update the BFTQ policy

            When enough experience is collected, fit the model to the batch.
        """
        if not self.training:
            return
        # Store transition to memory
        self.bftq.push(state.flatten(), action, reward, next_state.flatten(), done, info["cost"])

    def update(self):
        """
            Fit a budgeted policy on the batch by running the BFTQ algorithm.
        """
        # Fit model
        self.bftq.reset()
        network = self.bftq.run()
        # Update greedy policy
        self.exploration_policy.pi_greedy.set_network(network)

    def reset(self):
        if not self.np_random:
            self.seed()
        network = BudgetedMLP(size_state=np.prod(self.env.observation_space.shape),
                              n_actions=self.env.action_space.n,
                              **self.config["network"])
        self.bftq = BudgetedFittedQ(value_network=network, config=self.config, writer=self.writer)
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

    def set_time(self, time):
        self.exploration_policy.set_time(time)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        torch.manual_seed(seed & ((1 << 63) - 1))  # torch seeds are int64
        return [seed]

    def save(self, filename):
        self.bftq.save_network(filename)
        return filename

    def load(self, filename):
        if not self.bftq:
            self.reset()
        network = self.bftq.load_network(filename)
        self.exploration_policy.pi_greedy.set_network(network)
        return filename

    def eval(self):
        self.training = False
        self.config['exploration']['temperature'] = 0
        self.config['exploration']['final_temperature'] = 0
        self.exploration_policy.config = self.config["exploration"]

    @property
    def memory(self):
        return self.bftq.memory


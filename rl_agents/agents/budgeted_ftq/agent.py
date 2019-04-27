import numpy as np
from gym import logger
from gym.utils import seeding

from rl_agents.agents.abstract import AbstractAgent
from rl_agents.agents.budgeted_ftq.bftq import BudgetedFittedQ
from rl_agents.agents.budgeted_ftq.models import NetBFTQ
from rl_agents.agents.budgeted_ftq.policies import PytorchBudgetedFittedPolicy, RandomBudgetedPolicy, \
    EpsilonGreedyPolicy


class BFTQAgent(AbstractAgent):
    def __init__(self, env, config=None):
        super(BFTQAgent, self).__init__(config)
        self.env = env
        self.seed = None
        self.bftq = None
        self.exploration_policy = None
        self.reset()

    @classmethod
    def default_config(cls):
        return {
            "gamma": 0.9,
            "gamma_c": 0.9,
            "max_epochs": np.inf,
            "betas_for_duplication": np.arange(0, 1, 0.1),
            "betas_for_discretisation": np.arange(0, 1, 0.1),
            "optimizer": None,
            "loss_function": "l2",
            "loss_function_c": "l2",
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
            "reset_network_each_epoch": True,
            "network": {
                "beta_encoder_type": "LINEAR",
                "size_beta_encoder": 50,
                "activation_type": "RELU",
                "reset_type": "XAVIER",
                "layers": [
                    256,
                    128,
                    64
                ]
            }
        }

    def act(self, state):
        action = self.explore(state)
        minibatch_complete = False
        if minibatch_complete:
            self.fit()
        return action

    def explore(self, state):
        """

        :param state:
        :return: the selected action
        """
        beta = self.np_random.rand()
        action, beta = self.exploration_policy.execute(state, beta)
        return action


    def record(self, state, action, reward, next_state, done, info):
        self.bftq.push(state, action, reward, next_state, done, info["cost"])

    def fit(self):
        logger.info("Run BFTQ on current batch")

        # Fit model
        self.bftq.reset()
        self.bftq.run()
        network_path = self.bftq.save_policy()

        # Update greedy policy
        self.exploration_policy.pi_greedy = PytorchBudgetedFittedPolicy(
            network_path,
            self.bftq.betas_for_discretisation,
            self.bftq.device,
            self.config["hull_options"],
            self.config["clamp_qc"]
        )

    def reset(self):
        network = NetBFTQ(size_state=self.env.observation_space.shape,
                          n_actions=self.env.action_space.n,
                          *self.config["network"])
        self.bftq = BudgetedFittedQ(policy_network=network, config=self.config)
        self.exploration_policy = EpsilonGreedyPolicy(
            pi_greedy=RandomBudgetedPolicy(),
            pi_random=RandomBudgetedPolicy(),
            epsilon=self.config["epsilon"]
        )

    def seed(self, seed=None):
        """
            Seed the policy randomness source
        :param seed: the seed to be used
        :return: the used seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def save(self, filename):
        pass

    def load(self, filename):
        pass

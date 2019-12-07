import torch
import torch.nn as nn
import numpy as np

from rl_agents.agents.common.models import BaseModule, model_factory
from rl_agents.agents.common.models import model_factory as common_model_factory


class BudgetedNetwork(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.config["state_encoder"]["in"] = self.config["in"]
        self.config["head"]["in"] = self.config["state_encoder"]["out"] + self.config["size_beta_encoder"]
        self.config["head"]["out"] = self.config["out"]
        self.state_encoder = model_factory(self.config["state_encoder"])
        self.beta_encoder = torch.nn.Linear(1, self.config["size_beta_encoder"])
        self.head = model_factory(self.config["head"])

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "in": None,
            "out": None,
            "size_beta_encoder": 10,
            "state_encoder": {
                "type": "MultiLayerPerceptron",
                "out": 64
            },
            "head": {
                "type": "MultiLayerPerceptron"
            },
        })
        return config

    def forward(self, states, budgets):
        # Since we use the "out" parameter in the state encoder to specify output size,
        # the output is linear and needs an additional activation.
        states = self.activation(self.state_encoder(states))
        budgets = self.beta_encoder(budgets)
        x = torch.cat((states, budgets), dim=1)
        x = self.head(x)
        return x


def size_model_config(env, model_config):
    """
        Update the configuration of a model depending on the environment observation/action spaces

        Typically, the input/output sizes.

    :param env: an environment
    :param model_config: a model configuration
    """
    model_config["in"] = int(np.prod(env.observation_space.shape))
    model_config["out"] = 2 * env.action_space.n


def model_factory(config: dict) -> nn.Module:
    if config["type"] == "BudgetedNetwork":
        return BudgetedNetwork(config)
    else:
        return common_model_factory(config)

import torch
import torch.nn as nn
import numpy as np

from rl_agents.agents.common.models import BaseModule, model_factory
from rl_agents.configuration import Configurable
from rl_agents.agents.common.models import model_factory as common_model_factory


class BudgetedMLP(BaseModule, Configurable):
    def __init__(self, config):
        super().__init__(config)
        if self.config["beta_encoder_type"] == "LINEAR":
            self.beta_encoder = torch.nn.Linear(1, self.config["size_beta_encoder"])
        self.config["model"]["in"] = self.config["in"] + self.config["size_beta_encoder"]
        self.config["model"]["out"] = self.config["out"]
        self.model = model_factory(self.config["model"])

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "in": None,
            "out": None,
            "model": {
                "type": "MultiLayerPerceptron"
            },
            "size_beta_encoder": 10,
        })
        return config

    def forward(self, x):
        if self.config["normalize"]:
            x = (x - self.mean) / self.std

        beta = x[:, :, -1]
        beta = self.beta_encoder(beta)
        state = x[:, :, 0:-1][:, 0]
        x = torch.cat((state, beta), dim=1)
        x = self.model(x)
        return x.view(x.size(0), -1)


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
    if config["type"] == "BudgetedMLP":
        return BudgetedMLP(config)
    else:
        return common_model_factory(config)

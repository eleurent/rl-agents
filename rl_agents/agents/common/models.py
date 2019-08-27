import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from rl_agents.configuration import Configurable


class MultiLayerPerceptron(nn.Module, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        sizes = [self.config["in"]] + self.config["layers"]
        self.activation = activation_factory(self.config["activation"])
        layers_list = [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.layers = nn.ModuleList(layers_list)
        if self.config.get("out", None):
            self.predict = nn.Linear(sizes[-1], self.config["out"])

    @classmethod
    def default_config(cls):
        return {"in": None,
                "layers": [64, 64],
                "activation": "RELU",
                "reshape": "True",
                "out": None}

    def forward(self, x):
        if self.config["reshape"]:
            x = x.reshape(x.shape[0], -1)  # We expect a batch of vectors
        for layer in self.layers:
            x = self.activation(layer(x))
        if self.config.get("out", None):
            x = self.predict(x)
        return x


class DuelingNetwork(nn.Module, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.config["base_module"]["in"] = self.config["in"]
        self.base_module = model_factory(self.config["base_module"])
        self.advantage = nn.Linear(self.config["base_module"]["layers"][-1], self.config["out"])
        self.value = nn.Linear(self.config["base_module"]["layers"][-1], 1)

    @classmethod
    def default_config(cls):
        return {"in": None,
                "base_module": {"type": "MultiLayerPerceptron", "out": None},
                "out": None}

    def forward(self, x):
        x = self.base_module(x)
        advantage = self.advantage(x)
        value = self.value(x).expand(-1,  self.config["out"])
        return value + advantage - advantage.mean(1).unsqueeze(1).expand(-1,  self.config["out"])


class EgoAttention(nn.Module, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.features_per_head = int(self.config["feature_size"] / self.config["heads"])

        self.value_all = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.key_all = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.query_ego = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.attention_combine = nn.Linear(self.feature_size, self.feature_size, bias=False)

    @classmethod
    def default_config(cls):
        return {
            "feature_size": 64,
            "heads": 4,
            "dropout_factor": 0,
        }

    def forward(self, ego, others, mask=None):
        batch_size = others.shape[0]
        n_others = others.shape[1]
        input_all = torch.cat((ego.view(batch_size, 1, self.config["feature_size"]), others), dim=1)
        # Dimensions: Batch, item, head, feature
        key_all = self.key_all(input_all).view(batch_size, n_others + 1, self.config["heads"], self.features_per_head)
        value_all = self.value_all(input_all).view(batch_size, n_others + 1, self.config["heads"], self.features_per_head)
        query_ego = self.query_ego(ego).view(batch_size, 1, self.config["heads"], self.features_per_head)
        if mask is not None:
            mask = mask.unsqueeze(2).repeat((1, 1, self.config["heads"], 1))

        key_all = key_all.permute(0, 2, 1, 3)
        value_all = value_all.permute(0, 2, 1, 3)
        query_ego = query_ego.permute(0, 2, 1, 3)
        output = attention(query_ego, key_all, value_all, mask,
                           nn.Dropout(self.config["dropout_factor"])).reshape(batch_size, self.config["feature_size"])
        output = (self.attention_combine(output) + ego)/2
        return output

    def get_attention_matrix(self, ego, others, mask=None):
        batch_size = others.shape[0]
        n_others = others.shape[1]
        input_all = torch.cat((ego.view(batch_size, 1, self.config["feature_size"]), others), dim=1)
        key_all = self.key_others(input_all).view(batch_size, n_others, self.config["heads"], self.features_per_head)
        query_ego = self.query_ego(ego).view(batch_size, 1, self.config["heads"], self.features_per_head)
        mask = mask.unsqueeze(2).repeat((1, 1, self.config["heads"], 1))
        d_k = query_ego.size(-1)
        scores = torch.matmul(query_ego, key_all.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask_veh = (1 - torch.prod(1 - mask, 0)).unsqueeze(2)
            scores[:, :, :, :mask_veh.shape[-1]] = scores[:, :, :, :mask_veh.shape[-1]].masked_fill(mask_veh == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        return p_attn


class EgoAttentionAction(nn.Module, Configurable):
    def __init__(self, config):
        super(EgoAttentionAction, self).__init__(config)
        self.config = config
        self.config["embedding_layer"]["in"] = self.config["in"]
        self.config["output_layer"]["in"] = self.config["attention_layer"]["feature_size"]
        self.config["output_layer"]["out"] = self.config["out"]

        self.ego_embedding = model_factory(self.config["embedding_layer"])
        self.others_embedding = self.ego_embedding
        self.attention = EgoAttention(self.config["attention_layer"])
        self.output_layer = model_factory(self.config["output_layer"])

    @classmethod
    def default_config(cls):
        return {
            "in": None,
            "out": None,
            "n_head": 4,
            "embedding_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128]
            },
            "attention_layer": {
                "type": "EgoAttention",
                "feature_size": 128,
                "heads": 4
            },
            "output_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128]
            },
        }

    def forward(self, ego, others):
        ego_embedded_att = self.attention(self.ego_embedding(ego), self.others_embedding(others))
        return self.output_layer(ego_embedded_att)

    def get_attention_matrices(self, ego, others):
        return self.attention.get_attention_matrix(self.ego_embedding(ego), self.others_embedding(others))


def attention(query, key, value, mask=None, dropout=None):
    """ Compute a Scaled Dot Product Attention """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask_veh = (1 - torch.prod(1-mask, 0)).unsqueeze(2)
        scores[:, :, :, :mask_veh.shape[-1]] = scores[:, :, :, :mask_veh.shape[-1]].masked_fill(mask_veh == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output


def activation_factory(activation_type):
    if activation_type == "RELU":
        return F.relu
    elif activation_type == "TANH":
        return torch.tanh
    else:
        raise ValueError("Unknown activation_type: {}".format(activation_type))


def loss_function_factory(loss_function):
    if loss_function == "l2":
        return F.mse_loss
    elif loss_function == "l1":
        return  F.l1_loss
    elif loss_function == "bce":
        return  F.binary_cross_entropy
    else:
        raise ValueError("Unknown loss function : {}".format(loss_function))


def optimizer_factory(optimizer_type, params, lr=None, weight_decay=None):
    if optimizer_type == "ADAM":
        return torch.optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "RMS_PROP":
        return torch.optim.RMSprop(params=params, weight_decay=weight_decay)
    else:
        raise ValueError("Unknown optimizer type: {}".format(optimizer_type))


def model_factory(config: dict) -> nn.Module:
    if config["type"] == "MultiLayerPerceptron":
        return MultiLayerPerceptron(config)
    elif config["type"] == "DuelingNetwork":
        return DuelingNetwork(config)
    else:
        raise ValueError("Unknown model type")

import torch
import torch.nn.functional as F


class BaseModule(torch.nn.Module):
    """
        Base torch.nn.Module implementing basic features:
            - initialization factory
            - activation factory
            - normalization parameters
    """
    def __init__(self, activation_type="RELU", reset_type="XAVIER", normalize=None, **kwargs):
        super(BaseModule, self).__init__()
        self.activation = BaseModule.activation_factory(activation_type)
        self.reset_type = reset_type
        self.normalize = normalize
        self.mean = None
        self.std = None

    def _init_weights(self, m):
        if hasattr(m, 'weight'):
            if self.reset_type == "XAVIER":
                torch.nn.init.xavier_uniform_(m.weight.data)
            elif self.reset_type == "ZEROS":
                torch.nn.init.constant_(m.weight.data, 0.)
            else:
                raise ValueError("Unknown reset type")
        if hasattr(m, 'bias'):
            torch.nn.init.constant_(m.bias.data, 0.)

    @staticmethod
    def activation_factory(activation_type):
        if activation_type == "RELU":
            return F.relu
        elif activation_type == "TANH":
            return torch.tanh
        else:
            raise Exception("Unknown activation_type: {}".format(activation_type))

    def set_normalization_params(self, mean, std):
        if self.normalize:
            std[std == 0.] = 1.
        self.std = std
        self.mean = mean

    def reset(self):
        self.apply(self._init_weights)

    def forward(self, *x):
        return NotImplementedError


class MultiLayerPerceptron(BaseModule):
    def __init__(self, config):
        super().__init__(**config)
        self.config = config
        self.layer_sizes = [config["in"]] + config["layers"] + [config["out"]]
        self.layers = []
        for i in range(len(self.layer_sizes) - 2):
            module = torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1])
            self.layers.append(module)
            self.add_module("h_" + str(i), module)
        self.predict = torch.nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1])

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)  # We expect a batch of vectors
        if self.normalize:
            x = (x.float() - self.mean.float()) / self.std.float()
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.predict(x)
        return x


class DuelingNetwork(BaseModule):
    def __init__(self, config):
        super(DuelingNetwork, self).__init__(**config)
        self.config = config
        self.base_module = model_factory(config["base_module"])
        self.advantage = torch.nn.Linear(config["base_module"]["out"], config["out"])
        self.value = torch.nn.Linear(config["base_module"]["out"], 1)

    def forward(self, x):
        x = self.base_module(x)
        advantage = self.advantage(x)
        value = self.value(x).expand(-1,  self.config["out"])
        return value + advantage - advantage.mean(1).unsqueeze(1).expand(-1,  self.config["layers"][-1])


def model_factory(config):
    if config["type"] == "MultiLayerPerceptron":
        return MultiLayerPerceptron(**config)
    elif config["type"] == "DuelingNetwork":
        return DuelingNetwork(**config)
    else:
        raise ValueError("Unknown model type")

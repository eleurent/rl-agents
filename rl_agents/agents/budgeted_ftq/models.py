import torch
import torch.nn.functional as F


class BaseModule(torch.nn.Module):
    """
        Base torch.nn.Module implementing basic features:
            - initialization factory
            - activation factory
            - normalization parameters
    """
    def __init__(self, activation_type="RELU", reset_type="XAVIER", normalize=None):
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

    def forward(self, *input):
        return NotImplementedError


class NetBFTQ(BaseModule):
    def __init__(self, size_state, size_beta_encoder, intra_layers, n_actions,
                 activation_type="RELU",
                 reset_type="XAVIER",
                 normalize=False,
                 beta_encoder_type="LINEAR",
                 **kwargs):
        super(NetBFTQ, self).__init__(activation_type, reset_type, normalize)
        sizes = intra_layers + [2 * n_actions]
        self.beta_encoder_type = beta_encoder_type
        self.size_state = size_state
        self.size_beta_encoder = size_beta_encoder
        self.size_action = sizes[-1] / 2
        intra_layers = []
        if size_beta_encoder > 1:
            if self.beta_encoder_type == "LINEAR":
                self.beta_encoder = torch.nn.Linear(1, size_beta_encoder)
            self.concat_layer = torch.nn.Linear(size_state + size_beta_encoder, sizes[0])
        else:
            module = torch.nn.Linear(size_state + size_beta_encoder, sizes[0])
            intra_layers.append(module)
        for i in range(0, len(sizes) - 2):
            module = torch.nn.Linear(sizes[i], sizes[i + 1])
            intra_layers.append(module)
        self.linears = torch.nn.ModuleList(intra_layers)
        self.predict = torch.nn.Linear(sizes[-2], sizes[-1])

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std

        if self.size_beta_encoder > 1:
            beta = x[:, :, -1]
            if self.beta_encoder_type == "REPEAT":
                beta = beta.repeat(1, self.size_beta_encoder)
            elif self.beta_encoder_type == "LINEAR":
                beta = self.beta_encoder(beta)
            else:
                raise "Unknown encoder type : {}".format(self.beta_encoder_type)
            state = x[:, :, 0:-1][:, 0]
            x = torch.cat((state, beta), dim=1)
            x = self.concat_layer(x)
        elif self.size_beta_encoder == 1:
            pass
        else:
            x = x[:, :, 0:-1]

        for i, layer in enumerate(self.linears):
            x = self.activation(layer(x))
        x = self.predict(x)

        return x.view(x.size(0), -1)


def loss_fonction_factory(loss_function):
    if loss_function == "l2":
        return F.mse_loss
    elif loss_function == "l1":
        return  F.l1_loss
    elif loss_function == "bce":
        return  F.binary_cross_entropy
    else:
        raise Exception("Unknown loss function : {}".format(loss_function))


def optimizer_factory(optimizer_type, params, lr=None, weight_decay=None):
    if optimizer_type == "ADAM":
        return torch.optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "RMS_PROP":
        return torch.optim.RMSprop(params=params, weight_decay=weight_decay)
    else:
        raise ValueError("Unknown optimizer type: {}".format(optimizer_type))

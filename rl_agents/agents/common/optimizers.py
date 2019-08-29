import math
import torch
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
import itertools as it


class Ranger(Optimizer):
    """
        Credit Ranger code by https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
        # credit1 - Lookahead implementation from LonePatient - https://github.com/lonePatient/lookahead_pytorch/blob/master/optimizer.py
        # credit2 - RAdam code by https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py
        Credit Ralamb code by https://gist.github.com/redknightlois/c4023d393eb8f92bb44b2ab582d7ec20
    """
    def __init__(self, params, lr=1e-3, alpha=0.5, k=5, betas=(.9, 0.999), eps=1e-8, weight_decay=0):
        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('Invalid slow update rate: ', alpha)
        if not 1 <= k:
            raise ValueError('Invalid lookahead steps: ', k)
        if not lr > 0:
            raise ValueError('Invalid Learning Rate: ', lr)
        if not eps > 0:
            raise ValueError('Invalid eps: ', eps)

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # now we can get to work...
        for group in self.param_groups:
            group["step_counter"] = 0
            # print("group step counter init")

        # look ahead params
        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # # lookahead weights
        # self.slow_weights = [[p.clone().detach() for p in group['params']]
        #                      for group in self.param_groups]
        #
        # # don't use grad for lookahead weights
        # for w in it.chain(*self.slow_weights):
        #     w.requires_grad = False

    def __setstate__(self, state):
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        # note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.
        # Uncomment if you need to use the actual closure...

        # if closure is not None:
        # loss = closure()

        # ------------ radam
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.radam_buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > 5:
                        step_size = group['lr'] * math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
                radam_norm = p_data_fp32.pow(2).sum().sqrt()
                if weight_norm == 0 or radam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / radam_norm

                state['weight_norm'] = weight_norm
                state['adam_norm'] = radam_norm
                state['trust_ratio'] = trust_ratio

                if N_sma > 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * trust_ratio, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * trust_ratio, exp_avg)

                p.data.copy_(p_data_fp32)

        # ---------------- end radam step

        # # look ahead tracking and updating if latest batch = k
        # for group, slow_weights in zip(self.param_groups, self.slow_weights):
        #     group['step_counter'] += 1
        #     if group['step_counter'] % self.k != 0:
        #         continue
        #     for p, q in zip(group['params'], slow_weights):
        #         if p.grad is None:
        #             continue
        #         q.data.add_(self.alpha, p.data - q.data)
        #         p.data.copy_(q.data)

        return loss


def loss_function_factory(loss_function):
    if loss_function == "l2":
        return F.mse_loss
    elif loss_function == "l1":
        return F.l1_loss
    elif loss_function == "smooth_l1":
        return F.smooth_l1_loss
    elif loss_function == "bce":
        return F.binary_cross_entropy
    else:
        raise ValueError("Unknown loss function : {}".format(loss_function))


def optimizer_factory(optimizer_type, params, lr=None, weight_decay=None, k=None, **kwargs):
    if optimizer_type == "ADAM":
        return torch.optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "RMS_PROP":
        return torch.optim.RMSprop(params=params, weight_decay=weight_decay)
    elif optimizer_type == "RANGER":
        return Ranger(params=params, lr=lr, weight_decay=weight_decay, k=k)
    else:
        raise ValueError("Unknown optimizer type: {}".format(optimizer_type))

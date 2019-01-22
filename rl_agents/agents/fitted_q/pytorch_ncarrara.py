# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
import random
import os
import matplotlib.pyplot as plt
import tools.filled_step as fs
from agents.utils import ReplayMemory, Transition

from gym import logger

class Net(torch.nn.Module):
    def _init_weights(self, m):
        if hasattr(m, 'weight'):
            torch.nn.init.xavier_uniform_(m.weight.data)

    def __init__(self, sizes, activation=F.relu, normalize=None, clamp_Q=None):
        super(Net, self).__init__()
        self.activation = activation
        self.clamp_Q = clamp_Q
        if clamp_Q is not None:
            raise Exception("To be clamped")
        if normalize is not None:
            self.normalize = normalize
        self.layers = []
        for i in range(0, len(sizes) - 2):
            module = torch.nn.Linear(sizes[i], sizes[i + 1])
            self.layers.append(module)
            self.add_module("h_" + str(i), module)

        self.predict = torch.nn.Linear(sizes[-2], sizes[-1])

        self.std = None
        self.mean = None

    def set_normalization_params(self, mean, std):
        std[std == 0.] = 1.
        self.std = std
        self.mean = mean

    def forward(self, x):
        if self.normalize:
            x = (x.float() - self.mean.float()) / self.std.float()
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.predict(x)
        return x.view(x.size(0), -1)

    def reset(self):
        self.apply(self._init_weights)


class PytorchFittedQ:
    ALL_BATCH = "ALL_BATCH"
    ADAPTIVE = "ADAPTIVE"

    _GAMMA = None
    _MAX_FTQ_EPOCH = None
    _MAX_NN_EPOCH = None
    _id_ftq_epoch = None
    _non_final_mask = None
    _non_final_next_states = None
    _state_batch = None
    _action_batch = None
    _reward_batch = None
    _policy_network = None

    def __init__(self,
                 policy_network,
                 optimizer=None,
                 loss_function=None,
                 MAX_FTQ_EPOCH=np.inf,
                 MAX_NN_EPOCH=1000,
                 GAMMA=0.99,
                 RESET_POLICY_NETWORK_EACH_FTQ_EPOCH=True,
                 DELTA=0,
                 BATCH_SIZE_EXPERIENCE_REPLAY=50,
                 NN_LOSS_STOP_CONDITION=0.0,
                 CLIP_NEXT_REWARDS=False,
                 disp_states=[],
                 workspace="tmp",
                 action_str=None,
                 process_between_epoch=None
                 ):
        self.process_between_epoch=process_between_epoch
        self.action_str = action_str
        self.CLIP_NEXT_REWARDS = CLIP_NEXT_REWARDS
        self.workspace = workspace
        self.NN_LOSS_STOP_CONDITION = NN_LOSS_STOP_CONDITION
        self.BATCH_SIZE_EXPERIENCE_REPLAY = BATCH_SIZE_EXPERIENCE_REPLAY
        self.DELTA = DELTA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._policy_network = policy_network.to(self.device)
        self._policy_network.reset()
        self._MAX_FTQ_EPOCH = MAX_FTQ_EPOCH
        self._MAX_NN_EPOCH = MAX_NN_EPOCH
        self._GAMMA = GAMMA
        self.RESET_POLICY_NETWORK_EACH_FTQ_EPOCH = RESET_POLICY_NETWORK_EACH_FTQ_EPOCH
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = optim.RMSprop(self._policy_network.parameters())
        self.loss_function = loss_function
        if self.loss_function is None:
            self.loss_function = F.smooth_l1_loss
        self.memory = ReplayMemory(self.config[""])
        self.disp_states = disp_states
        self.path_historgrams = self.workspace + "/histograms"
        self.disp = True
        if self.disp and not os.path.exists(self.path_historgrams): os.mkdir(self.path_historgrams)
        self.statistiques=None

    def fit(self, transitions):
        self._construct_batch(transitions)
        self._policy_network.reset()
        self.delta = np.inf
        self._id_ftq_epoch = 0
        while self._id_ftq_epoch < self._MAX_FTQ_EPOCH and self.delta > self.DELTA:
            self._sample_batch()
            logger.debug("[epoch_ftq={}] #batch={}".format(self._id_ftq_epoch, len(self._state_batch)))
            losses = self._ftq_epoch()
            logger.debug("loss", losses[-1])
            self._id_ftq_epoch += 1
            logger.debug("[epoch_ftq={}] delta={}".format(self._id_ftq_epoch, self.delta))

    def _construct_batch(self, transitions):
        for t in transitions:
            # print t.state
            state = torch.tensor([[t.state]], device=self.device, dtype=torch.float)
            if t.next_state is not None:
                next_state = torch.tensor([[t.next_state]], device=self.device, dtype=torch.float)
            else:
                next_state = None
            action = torch.tensor([[t.action]], device=self.device, dtype=torch.long)
            reward = torch.tensor([t.reward], device=self.device)
            done = torch.tensor([t.done], device=self.device, dtype=torch.long)
            self.memory.push(state, action, next_state, reward, done)

        zipped = Transition(*zip(*self.memory.memory))
        state_batch = torch.cat(zipped.state)
        mean = torch.mean(state_batch, 0)
        std = torch.std(state_batch, 0)
        self._policy_network.set_normalization_params(mean, std)

    def _sample_batch(self):
        transitions = self.memory.sample(len(self.memory))
        self.batch_size = len(transitions)
        batch = Transition(*zip(*transitions))

        self._non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                            device=self.device,
                                            dtype=torch.uint8)
        self._non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        self._state_batch = torch.cat(batch.state)
        self._action_batch = torch.cat(batch.action)
        self._reward_batch = torch.cat(batch.reward)

    def _ftq_epoch(self):
        """
            Compute target value Qk+1 from model Qk and perform Bellman Residual Minimization
        :return: the evolution of losses
        """
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if self._id_ftq_epoch > 0:
            next_state_values[self._non_final_mask] = \
                self._policy_network(self._non_final_next_states).max(1)[0].detach()
        self.expected_state_action_values = (next_state_values * self._GAMMA) + self._reward_batch
        losses = self._optimize_model()
        return losses

    def _optimize_model(self):
        """
            Perform gradient descent on the Bellman Residual Loss
        :return: the evolution of losses
        """
        self.delta = self._compute_loss().item()
        if self.RESET_POLICY_NETWORK_EACH_FTQ_EPOCH:
            self._policy_network.reset()
        torch.set_grad_enabled(True)
        losses = []
        for _ in range(self._MAX_NN_EPOCH):
            losses.append(self._gradient_step())
        torch.set_grad_enabled(False)
        return losses

    def _compute_loss(self):
        """
            Evaluate the Bellman Residual Loss of the model from target Qk+1 value
        :return: the Bellman Residual Loss of the model over the training batch
        """
        state_action_values = self._policy_network(self._state_batch).gather(1, self._action_batch)
        Y_pred = state_action_values
        Y_target = self.expected_state_action_values.unsqueeze(1)
        loss = self.loss_function(Y_pred, Y_target)
        return loss

    def _gradient_step(self):
        """
            Perform one step of gradient descent over the loss.
            Use gradient clipping
        :return: the obtained loss
        """
        loss = self._compute_loss()
        self.optimizer.zero_grad()
        loss.backward()
        for param in self._policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def reset(self, reset_weight=True):
        self.memory.reset()
        if reset_weight:
            self._policy_network.reset()
        self._id_ftq_epoch = None
        self._non_final_mask = None
        self._non_final_next_states = None
        self._state_batch = None
        self._action_batch = None
        self._reward_batch = None

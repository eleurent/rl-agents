import numpy as np
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.autograd import Variable

from agents.abstract import AbstractAgent
from agents.utils import Transition, ReplayMemory, ExplorationPolicy, ValueFunctionViewer

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.lin1 = nn.Linear(config['layers'][0], config['layers'][1])
        self.d1 = nn.Dropout(p=0.2)
        self.lin2 = nn.Linear(config['layers'][1], config['layers'][2])
        self.d2 = nn.Dropout(p=0.2)
        self.lin3 = nn.Linear(config['layers'][2], config['layers'][3])

    def forward(self, x):
        x = functional.tanh(self.lin1(x))
        # x = self.d1(x)
        x = functional.tanh(self.lin2(x))
        # x = self.d2(x)
        return self.lin3(x)


class DqnPytorchAgent(AbstractAgent):
    BATCH_SIZE = 32
    GAMMA = 0.95
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 20*200
    TARGET_UPDATE = 10

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.config["num_states"] = env.observation_space.shape[0]
        self.config["num_actions"] = env.action_space.n
        self.config["layers"] = [self.config["num_states"]] + self.config["layers"] + [self.config["num_actions"]]
        self.policy_net = Network(config)
        self.target_net = Network(config)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        if use_cuda:
            self.policy_net.cuda()
            self.target_net.cuda()

        # self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-4)
        self.memory = ReplayMemory(config)
        self.exploration_policy = ExplorationPolicy(config)
        self.steps = 0

        self.load()

    def act(self, state):
        _, optimal_action = self.state_to_value(state)
        return self.exploration_policy.epsilon_greedy(optimal_action, self.env.action_space)

    def record(self, state, action, reward, next_state, done):
        # Store the transition in memory
        self.memory.push(Tensor([state]), action, reward, Tensor([next_state]), done)
        self.optimize_model()

    def optimize_model(self):
        if len(self.memory) < self.config['batch_size']:
            return
        transitions = self.memory.sample(self.config['batch_size'])
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = 1 - ByteTensor(batch.terminal)
        next_states_batch = Variable(torch.cat(batch.next_state))
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(LongTensor(batch.action))
        reward_batch = Variable(Tensor(batch.reward))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch)
        state_action_values = state_action_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(self.config['batch_size']).type(Tensor))
        _, best_actions = self.policy_net(next_states_batch).max(1)
        best_values = self.target_net(next_states_batch).gather(1, best_actions.unsqueeze(1)).squeeze(1)
        next_state_values[non_final_mask] = best_values[non_final_mask]

        # Compute the expected Q values
        expected_state_action_values = reward_batch + self.config['gamma'] * next_state_values
        # Undo volatility (which was used to prevent unnecessary gradients)
        expected_state_action_values = Variable(expected_state_action_values.data)

        # Compute loss
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        loss = functional.mse_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update the target network
        self.steps += 1
        if self.steps % self.config['target_update'] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def load(self):
        checkpoint = torch.load('tmp/latest.tar')
        self.policy_net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save(self, episode):
        filename = 'tmp/checkpoint-' + str(episode) + '.tar'
        state = {
                    'episode': episode + 1,
                    'state_dict': self.policy_net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
        torch.save(state, filename)
        torch.save(state, 'tmp/latest.tar')

    def state_to_value(self, state):
        value, action = self.policy_net(Variable(Tensor([state]))).max(1)
        return value.data.cpu().numpy()[0], action.data.cpu().numpy()[0]

    def states_to_values(self, states):
        values, actions = self.policy_net(Variable(Tensor(states))).max(1)
        return values.data.cpu().numpy(), actions.data.cpu().numpy()

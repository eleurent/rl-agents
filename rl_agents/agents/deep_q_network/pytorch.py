import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.autograd import Variable

from rl_agents.agents.deep_q_network.abstract import AbstractDQNAgent

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class DQNAgent(AbstractDQNAgent):
    def __init__(self, env, config=None):
        super(DQNAgent, self).__init__(env, config)
        self.policy_net = model_factory(self.config["model"])
        self.target_net = model_factory(self.config["model"])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        if use_cuda:
            self.policy_net.cuda()
            self.target_net.cuda()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config["optimizer"]["lr"])
        self.steps = 0

    def step_optimizer(self, loss):
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def compute_bellman_residual(self, batch, target_state_action_value=None):
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = 1 - ByteTensor(batch.terminal)
        next_states_batch = Variable(torch.cat(tuple(Tensor([batch.next_state]))))
        state_batch = Variable(torch.cat(tuple(Tensor([batch.state]))))
        action_batch = Variable(LongTensor(batch.action))
        reward_batch = Variable(Tensor(batch.reward))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch)
        state_action_values = state_action_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        if target_state_action_value is None:
            # Compute V(s_{t+1}) for all next states.
            next_state_values = Variable(torch.zeros(reward_batch.shape).type(Tensor))
            # Double Q-learning: pick best actions from policy network
            _, best_actions = self.policy_net(next_states_batch).max(1)
            # Double Q-learning: estimate action values from target network
            best_values = self.target_net(next_states_batch).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            next_state_values[non_final_mask] = best_values[non_final_mask]

            # Compute the expected Q values
            target_state_action_value = reward_batch + self.config["gamma"] * next_state_values
            # Undo volatility (to prevent unnecessary gradients)
            target_state_action_value = Variable(target_state_action_value.data)

        # Compute loss
        # loss = F.smooth_l1_loss(state_action_values, target_state_action_value)
        loss = functional.mse_loss(state_action_values, target_state_action_value)
        return loss, target_state_action_value

    def get_batch_state_values(self, states):
        values, actions = self.policy_net(Variable(Tensor(states))).max(1)
        return values.data.cpu().numpy(), actions.data.cpu().numpy()

    def get_batch_state_action_values(self, states):
        return self.policy_net(Variable(Tensor(states))).data.cpu().numpy()

    def save(self, filename):
        state = {'state_dict': self.policy_net.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['state_dict'])
        self.target_net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def initialize_model(self):
        self.policy_net.reset()


class FCNetwork(nn.Module):
    def __init__(self, config):
        super(FCNetwork, self).__init__()
        self.lin1 = nn.Linear(config["all_layers"][0], config["all_layers"][1])
        self.lin2 = nn.Linear(config["all_layers"][1], config["all_layers"][2])
        self.lin3 = nn.Linear(config["all_layers"][2], config["all_layers"][3])

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = torch.tanh(self.lin1(x))
        x = torch.tanh(self.lin2(x))
        return self.lin3(x)

    def reset(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if hasattr(m, 'weight'):
            torch.nn.init.xavier_uniform_(m.weight.data)


class DuelingNetwork(nn.Module):
    def __init__(self, config):
        super(DuelingNetwork, self).__init__()
        self.config = config
        self.lin1 = nn.Linear(config["all_layers"][0], config["all_layers"][1])
        self.lin2 = nn.Linear(config["all_layers"][1], config["all_layers"][2])
        self.advantage = nn.Linear(config["all_layers"][2], config["all_layers"][3])
        self.value = nn.Linear(config["all_layers"][2], 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = torch.tanh(self.lin1(x))
        x = torch.tanh(self.lin2(x))
        advantage = self.advantage(x)
        value = self.value(x).expand(-1,  self.config["all_layers"][3])
        return value + advantage - advantage.mean(1).unsqueeze(1).expand(-1,  self.config["all_layers"][3])


def model_factory(config):
    if config["type"] == "FCNetwork":
        return FCNetwork(config)
    elif config["type"] == "DuelingNetwork":
        return DuelingNetwork(config)
    else:
        raise ValueError("Unknown model type")

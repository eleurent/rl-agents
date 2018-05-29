import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.autograd import Variable

from rl_agents.agents.dqn.abstract import DQNAgent
from rl_agents.agents.utils import Transition

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class DQNPytorchAgent(DQNAgent):
    def __init__(self, env, config=None):
        super(DQNPytorchAgent, self).__init__(env, config)
        self.policy_net = model_factory(self.config["model"])
        self.target_net = model_factory(self.config["model"])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        if use_cuda:
            self.policy_net.cuda()
            self.target_net.cuda()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-4)
        self.steps = 0

    def record(self, state, action, reward, next_state, done):
        # Store the transition in memory
        self.memory.push(Tensor([state]), int(action), reward, Tensor([next_state]), done)
        self.optimize_model()

    def optimize_model(self):
        if len(self.memory) < self.config["batch_size"]:
            return
        transitions = self.memory.sample(self.config["batch_size"])
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
        next_state_values = Variable(torch.zeros(self.config["batch_size"]).type(Tensor))
        # Double Q-learning: pick best actions from policy network
        _, best_actions = self.policy_net(next_states_batch).max(1)
        # Double Q-learning: estimate action values from target network
        best_values = self.target_net(next_states_batch).gather(1, best_actions.unsqueeze(1)).squeeze(1)
        next_state_values[non_final_mask] = best_values[non_final_mask]

        # Compute the expected Q values
        expected_state_action_values = reward_batch + self.config["gamma"] * next_state_values
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
        if self.steps % self.config["target_update"] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

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
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class FCNetwork(nn.Module):
    def __init__(self, config):
        super(FCNetwork, self).__init__()
        self.lin1 = nn.Linear(config["all_layers"][0], config["all_layers"][1])
        self.d1 = nn.Dropout(p=0.2)
        self.lin2 = nn.Linear(config["all_layers"][1], config["all_layers"][2])
        self.d2 = nn.Dropout(p=0.2)
        self.lin3 = nn.Linear(config["all_layers"][2], config["all_layers"][3])

    def forward(self, x):
        x = functional.tanh(self.lin1(x))
        # x = self.d1(x)
        x = functional.tanh(self.lin2(x))
        # x = self.d2(x)
        return self.lin3(x)


class DuelingNetwork(nn.Module):
    def __init__(self, config):
        super(DuelingNetwork, self).__init__()
        self.config = config
        self.lin1 = nn.Linear(config["all_layers"][0], config["all_layers"][1])
        self.lin2 = nn.Linear(config["all_layers"][1], config["all_layers"][2])
        self.advantage = nn.Linear(config["all_layers"][2], config["all_layers"][3])
        self.value = nn.Linear(config["all_layers"][2], 1)

    def forward(self, x):
        x = functional.tanh(self.lin1(x))
        x = functional.tanh(self.lin2(x))
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

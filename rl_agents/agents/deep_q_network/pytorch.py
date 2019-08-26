import torch
import torch.optim as optim
import torch.nn.functional as functional
from torch.autograd import Variable

from rl_agents.agents.common.models import model_factory
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
        self.value_net = model_factory(self.config["model"])
        self.target_net = model_factory(self.config["model"])
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.target_net.eval()
        if use_cuda:
            self.value_net.cuda()
            self.target_net.cuda()

        self.optimizer = optim.Adam(self.value_net.parameters(), lr=self.config["optimizer"]["lr"])
        self.steps = 0

    def step_optimizer(self, loss):
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.value_net.parameters():
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
        state_action_values = self.value_net(state_batch)
        state_action_values = state_action_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        if target_state_action_value is None:
            # Compute V(s_{t+1}) for all next states.
            next_state_values = Variable(torch.zeros(reward_batch.shape).type(Tensor))
            # Double Q-learning: pick best actions from policy network
            _, best_actions = self.value_net(next_states_batch).max(1)
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
        values, actions = self.value_net(Variable(Tensor(states))).max(1)
        return values.data.cpu().numpy(), actions.data.cpu().numpy()

    def get_batch_state_action_values(self, states):
        return self.value_net(Variable(Tensor(states))).data.cpu().numpy()

    def save(self, filename):
        state = {'state_dict': self.value_net.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.value_net.load_state_dict(checkpoint['state_dict'])
        self.target_net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def initialize_model(self):
        self.value_net.reset()

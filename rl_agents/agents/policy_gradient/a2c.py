import torch
import torch.optim as optim
from torch.autograd import Variable

from rl_agents.agents.abstract import AbstractAgent
from rl_agents.agents.policy_gradient.models import ActorCritic

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class A2C(AbstractAgent):
    def __init__(self, env, config=None):
        super(A2C, self).__init__(env, config)
        self.policy_net = ActorCritic(env.obs_space, env.action_space)

        if use_cuda:
            self.policy_net.cuda()

        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                    lr=self.config["adam_lr"],
                                    eps=self.config["adam_eps"])
        self.steps = 0

    def update(self, minibatch):
        states, actions, returns = minibatch

        """update critic"""
        values_target = Variable(returns)
        values_pred = value_net(Variable(states))
        value_loss = (values_pred - values_target).pow(2).mean()
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

        """update policy"""
        log_probs = policy_net.get_log_prob(Variable(states), Variable(actions))
        policy_loss = -(log_probs * Variable(advantages)).mean()
        optimizer_policy.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)

    optimizer_policy.step()

    @classmethod
    def default_config(cls):
        return dict(lr=5e-4,
                    eps=1e-8)

    def record(self, state, action, reward, next_state, done):
        # Store the transition in memory
        self.memory.push(Tensor([state]), int(action), reward, Tensor([next_state]), done)
        self.optimize_model()


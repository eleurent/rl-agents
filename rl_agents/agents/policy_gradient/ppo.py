import torch
import torch.optim as optim

from rl_agents.agents.abstract import AbstractAgent
from rl_agents.agents.policy_gradient.models import ActorCritic

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class PPOAgent(AbstractAgent):
    def __init__(self, env, config=None):
        super(PPOAgent, self).__init__(env, config)
        self.policy_net = ActorCritic(env.obs_space, env.action_space)

        if use_cuda:
            self.policy_net.cuda()

        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                    lr=self.config["adam_lr"],
                                    eps=self.config["adam_eps"])
        self.steps = 0

    @classmethod
    def default_config(cls):
        return dict(lr=5e-4,
                    eps=1e-8)

    def record(self, state, action, reward, next_state, done):
        # Store the transition in memory
        self.memory.push(Tensor([state]), int(action), reward, Tensor([next_state]), done)
        self.optimize_model()


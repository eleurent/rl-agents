import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_space):
        super(ActorCritic, self).__init__()
        self.model = FCActorCritic(obs_shape[0])
        self.state_size = self.model.state_size

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states):
        actions, value = self.model(inputs, states)
        action = actions.sample()
        action_log_probs = actions.log_probs(action)
        dist_entropy = actions.entropy().mean()
        return action, action_log_probs, dist_entropy, value

    def get_value(self, inputs, states):
        value, _, _ = self.model(inputs, states)
        return value

    def evaluate_actions(self, inputs, states, action):
        actions, value = self.model(inputs, states)
        action_log_probs = actions.log_probs(action)
        dist_entropy = actions.entropy().mean()
        return action_log_probs, dist_entropy, value


class FCActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(FCActorCritic, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.actor = nn.Linear(64, num_outputs)
        self.critic = nn.Linear(64, 1)
        self.train()

    def forward(self, inputs, states):
        features = self.main(inputs)
        return torch.distributions.Categorical(logits=self.actor(features)), self.critic(features)


# class CNNBase(nn.Module):
#     def __init__(self, num_inputs):
#         super(CNNBase, self).__init__()
#
#         self.main = nn.Sequential(
#             nn.Conv2d(num_inputs, 32, 8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 32, 3, stride=1),
#             nn.ReLU(),
#             Flatten(),
#             nn.Linear(32 * 7 * 7, 512),
#             nn.ReLU()
#         )
#
#         self.critic = nn.Linear(512, 1)
#         self.train()
#
#     def forward(self, inputs, states):
#         features = self.main(inputs)
#         return torch.distributions.Categorical(logits=self.actor(features)), self.critic(features)
#
#
# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.view(x.size(0), -1)

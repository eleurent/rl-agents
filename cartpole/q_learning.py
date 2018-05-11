# -*- coding: utf-8 -*-

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(4, 16)
        # self.bn1 = nn.BatchNorm1d(16)
        self.lin2 = nn.Linear(16, 16)
        # self.bn2 = nn.BatchNorm1d(16)
        self.head = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.head(x)


BATCH_SIZE = 32
GAMMA = 0.95
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 20*200
TARGET_UPDATE = 10

policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if use_cuda:
    policy_net.cuda()
    target_net.cuda()

optimizer = optim.RMSprop(policy_net.parameters())
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        out = policy_net(
            Variable(torch.from_numpy(state), volatile=True).type(FloatTensor))
        values, indices = out.data.max(0)
        return indices.view(1, 1)[0, 0]
    else:
        return random.randrange(2)


episode_durations = []
predicted_values = np.array([])


def plot_durations():
    plt.figure(1)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(episode_durations)
    # Take 100 episode averages and plot them too
    if len(episode_durations) >= 100:
        means = np.hstack((np.zeros((100,)), np.convolve(episode_durations, np.ones((100,))/100, mode='valid')))
        plt.plot(means)
    else:
        plt.plot(0*episode_durations)

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def plot_values():
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Predicted value')

    plt.plot(predicted_values)

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def plot_value_map():
    xx, yy = np.meshgrid(np.linspace(-1, 1, 15), np.linspace(-1, 1, 15))
    xf = np.reshape(xx, (np.size(xx), 1))
    yf = np.reshape(yy, (np.size(yy), 1))
    states = np.hstack((2*xf, 2*xf, yf*12*np.pi/180, yf))
    values, actions = policy_net(Variable(Tensor(states))).max(1)
    values = np.reshape(values.data.cpu().numpy(), np.shape(xx))
    actions = np.reshape(actions.data.cpu().numpy(), np.shape(xx))
    plt.figure(3)
    plt.clf()
    plt.imshow(values)
    plt.colorbar()
    plt.figure(4)
    plt.clf()
    plt.imshow(actions)
    plt.colorbar()


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = 1-ByteTensor(batch.terminal)
    next_states_batch = Variable(torch.cat(batch.next_state))
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(LongTensor(batch.action))
    reward_batch = Variable(Tensor(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch)
    state_action_values = state_action_values.gather(1, action_batch.unsqueeze(1))

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    _, best_actions = policy_net(next_states_batch).max(1)
    next_state_values[non_final_mask] = target_net(next_states_batch).gather(1, best_actions.unsqueeze(1))
    # next_state_values[1-non_final_mask] = -1

    # Compute the expected Q values
    expected_state_action_values = reward_batch + GAMMA * next_state_values
    # Undo volatility (which was used to prevent unnecessary gradients)
    expected_state_action_values = Variable(expected_state_action_values.data)

    # Compute Huber loss
    # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    loss = F.mse_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def state_to_value(state):
    return policy_net(Variable(Tensor([state]))).max(1)[0].data.cpu().numpy()[0]


def store_values(values):
    value = state_to_value(np.zeros((4,)))
    tilt_left_value = state_to_value(np.array([-0.08936051, -0.37169457,  0.20398587,  1.03234316]))
    tilt_right_value = state_to_value(np.array([0.10718797,  0.97770614, -0.20473761, -1.6631015]))
    if values.size:
        values = np.vstack((values, [[value, tilt_left_value, tilt_right_value]]))
    else:
        values = np.array([[value, tilt_left_value, tilt_right_value]])
    return values


num_episodes = 5000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    obs = env.reset()
    for t in count():
        # Select and perform an action
        action = select_action(obs)
        next_obs, reward, done, _ = env.step(action)
        reward = reward

        # Store the transition in memory
        memory.push(Tensor([obs]), action, Tensor([next_obs]), reward, done)

        # Move to the next state
        obs = next_obs

        if i_episode % 25 == 0:
            env.render()

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            predicted_values = store_values(predicted_values)
            plot_values()
            plot_value_map()
            break
    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()

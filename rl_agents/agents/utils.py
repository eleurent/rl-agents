from collections import namedtuple
import numpy as np
import random

EPSILON = 0.01


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'terminal'))


class ReplayMemory(object):
    def __init__(self, config):
        self.capacity = int(config['memory_capacity'])
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # Faster than append and pop
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def constrain(x, a, b):
    return np.minimum(np.maximum(x, a), b)


def not_zero(x):
    if abs(x) > EPSILON:
        return x
    elif x > 0:
        return EPSILON
    else:
        return -EPSILON


def wrap_to_pi(x):
    return ((x+np.pi) % (2*np.pi)) - np.pi




from collections import namedtuple
import numpy as np
import random

from rl_agents.configuration import Configurable

EPSILON = 0.01


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'terminal'))


class ReplayMemory(Configurable):
    """
        Container that stores and samples transitions.
    """
    def __init__(self, config=None):
        super(ReplayMemory, self).__init__(config)
        self.capacity = int(self.config['memory_capacity'])
        self.memory = []
        self.position = 0

    @classmethod
    def default_config(cls):
        return dict(memory_capacity=10000,
                    n_steps=1,
                    gamma=0.99)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # Faster than append and pop
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, collapsed=True):
        """
            Sample a batch of transitions.

            If n_steps is greater than one, the batch will be composed of lists of successive transitions.
        :param batch_size: size of the batch
        :param collapsed: whether successive transitions must be collapsed into one n-step transition.
        :return: the sampled batch
        """
        # TODO: use agent's np_random for seeding
        if self.config["n_steps"] == 1:
            # Directly sample transitions
            return random.sample(self.memory, batch_size)
        else:
            # Sample initial transition indexes
            indexes = random.sample(range(len(self.memory)), batch_size)
            # Get the batch of n-consecutive-transitions starting from sampled indexes
            all_transitions = [self.memory[i:i+self.config["n_steps"]] for i in indexes]
            # Collapse transitions
            return map(self.collapse_n_steps, all_transitions) if collapsed else all_transitions

    def collapse_n_steps(self, transitions):
        """
            Collapse n transitions <s,a,r,s',t> of a trajectory into one transition <s0, a0, Sum(r_i), sp, tp>.

            We start from the initial state, perform the first action, and then the return estimate is formed by
            accumulating the discounted rewards along the trajectory until a terminal state or the end of the
            trajectory is reached.
        :param transitions: A list of n successive transitions
        :return: The corresponding n-step transition
        """
        state, action, cumulated_reward, next_state, done = transitions[0]
        discount = 1
        for transition in transitions[1:]:
            if done:
                break
            else:
                _, _, reward, next_state, done = transition
                discount *= self.config['gamma']
                cumulated_reward += discount*reward
        return state, action, cumulated_reward, next_state, done

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




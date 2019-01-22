from collections import namedtuple
import numpy as np
import random

from rl_agents.configuration import Configurable

EPSILON = 0.01


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'terminal', 'info'))


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
            self.position = len(self.memory) - 1
        elif len(self.memory) > self.capacity:
            self.memory = self.memory[:self.capacity]
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
        state, action, cumulated_reward, next_state, done, info = transitions[0]
        discount = 1
        for transition in transitions[1:]:
            if done:
                break
            else:
                _, _, reward, next_state, done, info = transition
                discount *= self.config['gamma']
                cumulated_reward += discount*reward
        return state, action, cumulated_reward, next_state, done, info

    def __len__(self):
        return len(self.memory)

    def is_full(self):
        return len(self.memory) == self.capacity


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


def bernoulli_kullback_leibler(p, q):
    """
        Compute the Kullback-Leibler divergence of two Bernoulli distributions.

    :param p: parameter of the first Bernoulli distribution
    :param q: parameter of the second Bernoulli distribution
    :return: KL(B(p), B(q))
    """
    kl1, kl2 = 0, np.infty
    if p > 0:
        if q > 0:
            kl1 = p*np.log(p/q)

    if q < 1:
        if p < 1:
            kl2 = (1 - p) * np.log((1 - p) / (1 - q))
        else:
            kl2 = 0
    return kl1 + kl2


def d_bernoulli_kullback_leibler_dq(p, q):
    """
        Compute the partial derivative of the Kullback-Leibler divergence of two Bernoulli distributions.

        With respect to the parameter q of the second distribution.

    :param p: parameter of the first Bernoulli distribution
    :param q: parameter of the second Bernoulli distribution
    :return: dKL/dq(B(p), B(q))
    """
    return (1 - p) / (1 - q) - p/q


def hoeffding_upper_bound(_sum, count, time):
    """
        Upper Confidence Bound of the empirical mean built on the Chernoff-Hoeffding inequality.

    :param _sum: Sum of sample values
    :param count: Number of samples
    :param time: Allows to set the bound confidence level to time^-4
    """
    return _sum / count + np.sqrt(2 * np.log(time) / count)


def kl_upper_bound(_sum, count, time, eps=1e-2):
    """
        Upper Confidence Bound of the empirical mean built on the Kullback-Leibler divergence.

        The computation involves solving a small convex optimization problem using Newton Iteration

    :param _sum: Sum of sample values
    :param count: Number of samples
    :param time: Allows to set the bound confidence level
    :param eps: Absolute accuracy of the Netwon Iteration
    """
    mu = _sum/count
    max_div = np.log(time)/count

    # Solve KL(mu, q) = max_div
    q = mu
    next_q = (1 + mu)/2
    while abs(q - next_q) > eps:
        q = next_q

        # Newton Iteration
        klq = bernoulli_kullback_leibler(mu, q) - max_div
        d_klq = d_bernoulli_kullback_leibler_dq(mu, q)
        next_q = q - klq / d_klq

        # Out of bounds: move toward the bound
        weight = 0.9
        if next_q > 1:
            next_q = weight*1 + (1 - weight)*q
        elif next_q < mu:
            next_q = weight*mu + (1 - weight)*q

    return constrain(q, 0, 1)



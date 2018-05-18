from abc import ABCMeta, abstractmethod


class AbstractAgent(object):
    """
        An abstract class specifying the interface of a generic agent.
    """
    metaclass__ = ABCMeta

    @abstractmethod
    def record(self, state, action, reward, next_state, done):
        """
            Record a transition of the environment to update the agent
        :param state: s, the current state of the agent
        :param action: a, the action performed
        :param reward: r(s, a), the reward collected
        :param next_state: s', the new state of the agent after the action was performed
        :param done: whether the next state is terminal
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def act(self, state):
        """
            Pick an action

        :param state: the current state
        :return: the action
        """
        raise NotImplementedError()

    @abstractmethod
    def plan(self, state):
        """
            Plan an optimal trajectory from an initial state.

        :param state: the initial state
        :return: the optimal sequence of actions [a0, a1, a2...]
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        """
            Reset the agent to its initial internal state
        """
        raise NotImplementedError()

    @abstractmethod
    def seed(self, seed=None):
        """
            Seed the agent's random number generator
        :param seed: the seed to be used
        :return: the used seed
        """
        raise NotImplementedError()

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

        :param state: s, the current state of the agent
        :return: a, the action to perform
        """
        raise NotImplementedError()

    @abstractmethod
    def plan(self, state):
        """
            Plan an optimal trajectory from an initial state.

        :param state: s, the initial state of the agent
        :return: [a0, a1, a2...], a sequence of actions to perform
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
        :param seed: the seed to be used to generate random numbers
        :return: the used seed
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, filename):
        """
            Save the model parameters to a file
        :param str filename: the path of the file to save the model parameters in
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self, filename):
        """
            Load the model parameters from a file
        :param str filename: the path of the file to load the model parameters from
        """
        raise NotImplementedError()

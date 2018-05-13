from abc import ABCMeta, abstractmethod


class AbstractAgent(object):
    """
        An abstract class specifying the interface of a generic agent.
    """
    metaclass__ = ABCMeta

    @abstractmethod
    def record(self, state, action, reward, next_state, done):
        raise NotImplementedError()

    @abstractmethod
    def act(self, state):
        """
            Pick an action

        :param state: the current state
        :return: the action
        """
        raise NotImplementedError()


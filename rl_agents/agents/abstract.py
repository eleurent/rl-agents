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


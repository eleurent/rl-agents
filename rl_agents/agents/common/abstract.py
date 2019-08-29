from abc import ABC, abstractmethod

from rl_agents.configuration import Configurable


class AbstractAgent(Configurable, ABC):

    def __init__(self, config=None):
        super(AbstractAgent, self).__init__(config)
        self.writer = None  # Tensorboard writer

    """
        An abstract class specifying the interface of a generic agent.
    """
    @abstractmethod
    def record(self, state, action, reward, next_state, done, info):
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

    def plan(self, state):
        """
            Plan an optimal trajectory from an initial state.

        :param state: s, the initial state of the agent
        :return: [a0, a1, a2...], a sequence of actions to perform
        """
        return [self.act(state)]

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

    def eval(self):
        """
            Set to testing mode. Disable any unnecessary exploration.
        """
        pass

    def set_writer(self, writer):
        """
            Set a tensorboard writer to log the agent internal variables.
        :param SummaryWriter writer: a summary writer
        """
        self.writer = writer

    def set_time(self, time):
        """ Set a local time, to control the agent internal schedules (e.g. exploration) """
        pass


class AbstractStochasticAgent(AbstractAgent):
    """
        Agents that implement a stochastic policy
    """
    def action_distribution(self, state):
        """
            Compute the distribution of actions for a given state
        :param state: the current state
        :return: a dictionary {action:probability}
        """
        raise NotImplementedError()

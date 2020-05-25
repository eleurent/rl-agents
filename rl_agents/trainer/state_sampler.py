from abc import abstractmethod
import numpy as np


class AbstractStateSampler(object):
    @abstractmethod
    def states_list(self):
        """
            Get a list of relevant states from a problem state-space
        :return: 2D array of vertically stacked state rows
        """
        raise NotImplementedError()

    @abstractmethod
    def states_mesh(self):
        """
            Get a 2D mesh of relevant states from a problem state-space
        :return: a tuple (xx, yy, states)
                 xx and yy are vectors of the coordinates of the state in the chosen 2D-manifold of the state space
                 states is an array of vertically stacked state rows
        """
        raise NotImplementedError()


class CartPoleStateSampler(AbstractStateSampler):
    def __init__(self, resolution=15):
        self.resolution = resolution

    def states_mesh(self):
        xx, yy = np.meshgrid(np.linspace(-1, 1, self.resolution), np.linspace(-1, 1, self.resolution))
        xf = np.reshape(xx, (np.size(xx), 1))
        yf = np.reshape(yy, (np.size(yy), 1))
        states = np.hstack((2 * xf, 2 * xf, yf * 12 * np.pi / 180, yf))
        return xx, yy, states

    def states_list(self):
        return np.array([[0, 0, 0, 0],
                         [-0.08936051, -0.37169457, 0.20398587, 1.03234316],
                         [0.10718797, 0.97770614, -0.20473761, -1.6631015]])


class MountainCarStateSampler(AbstractStateSampler):
    def __init__(self, resolution=15):
        self.resolution = resolution

    def states_mesh(self):
        xx, yy = np.meshgrid(np.linspace(-1, 1, self.resolution), np.linspace(-1, 1, self.resolution))
        xf = np.reshape(xx, (np.size(xx), 1))
        yf = np.reshape(yy, (np.size(yy), 1))
        states = np.hstack((-0.35+0.85*xf, 0.06*yf))
        return xx, yy, states

    def states_list(self):
        return np.array([[-0.5, 0],  # Initial
                         [-1.2, 0],  # Left side
                         [-0.5, 0.06],  # Bottom with forward speed
                         [0.5, 0.04]])  # Goal


class ObstacleStateSampler(AbstractStateSampler):
    def __init__(self, resolution=15):
        self.resolution = resolution

    def states_mesh(self):
        xx, yy = np.meshgrid(np.linspace(-1, 1, self.resolution), np.linspace(-1, 1, self.resolution))
        xf = np.reshape(xx, (np.size(xx), 1))
        yf = np.reshape(yy, (np.size(yy), 1))
        o = np.ones(np.shape(xf))
        states = np.hstack((1/2+xf/2, 1/2+yf/2, 0*o, 1*o, 0.1+1/2-xf/2, o, o, o, 0.1+1/2-xf/2,
                            o, o, o, o, o, o, o, o, o, o, o))
        return xx, yy, states

    def states_list(self):
        return np.array([[1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],  # Far
                         [1., 0., 1., 0., 0.6, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],  #
                         [1., 0., 1., 0., 0.3, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],  #
                         [1., 0., 1., 0., 0.1, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])  # Close

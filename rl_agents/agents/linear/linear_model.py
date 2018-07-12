import numpy as np
import pandas as pd
from sklearn import linear_model

from highway_env.vehicle.behavior import IntervalObserver
from highway_env.vehicle.control import MDPVehicle, ControlledVehicle
from rl_agents.agents.abstract import AbstractAgent
from highway_env.envs.abstract import AbstractEnv


class LinearModelAgent(AbstractAgent):
    def __init__(self, env, config=None):
        """
        :param highway_env.envs.abstract.AbstractEnv env: a highway-env environment
        :param config: the agent config
        """
        super(LinearModelAgent, self).__init__(config)
        if not isinstance(env, AbstractEnv):
            raise ValueError("Only compatible with highway-env environments.")
        self.env = env
        self.tracked_vehicles = []
        self.observers = []

    @classmethod
    def default_config(cls):
        return dict()

    def act(self, observation):
        self.update_interval_observer()
        return 1

    def update_interval_observer(self):
        if not self.observers:
            for vehicle in self.env.unwrapped.road.vehicles:
                if vehicle not in self.tracked_vehicles and isinstance(vehicle, ControlledVehicle):
                    self.tracked_vehicles.append(vehicle)
                    self.observers.append(IntervalObserver(vehicle))

    def linear_regression(self):
        for vehicle in self.tracked_vehicles:
            history = vehicle.get_log()
            if np.shape(history)[0] < 2:
                return

            for lane_index in range(len(vehicle.road.lanes)):
                dy = history['dy_lane_{}'.format(lane_index)]
                d_psi = history['psi_lane_{}'.format(lane_index)] - history['psi']
                v = history['v']
                l = vehicle.LENGTH
                x = pd.concat([l / v / v * dy, l / v * d_psi], axis=1)
                y = history['steering']
                regr = linear_model.LinearRegression(fit_intercept=False)
                regr.fit(x, y)
                print(vehicle.target_lane_index, lane_index, regr.score(x, y))

    def reset(self):
        pass

    def seed(self, seed=None):
        pass

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()

    def record(self, state, action, reward, next_state, done):
        pass

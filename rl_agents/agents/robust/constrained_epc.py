import itertools

import gym
import numpy as np
from gym import Wrapper
from numpy.linalg import LinAlgError

from rl_agents.agents.control.interval_feedback import IntervalFeedback
from rl_agents.agents.robust.robust_epc import RobustEPCAgent


class ConstrainedEPCAgent(RobustEPCAgent):
    """
        Robust Estimation, Prediction and Control.
    """
    def __init__(self, env, config=None):
        super().__init__(env, config)
        self.feedback = None

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update({
            "noise_bound": 1,
            "perturbation_bound": 1
        })
        return cfg

    def ellipsoid(self):
        """
            Compute a confidence ellipsoid over the parameter theta, where
                    dot{x} = A x + (phi x) theta + B u + D omega
                         y = dot{x} + C nu
            under a bounded noise assumption
        :return: estimated theta, and radius
        """
        d = self.phi.shape[0]
        if not self.data:
            theta_n = 10*np.ones(d)
            g_n = np.eye(d)
            beta_n = np.sqrt(d) * self.config["parameter_bound"]
        else:
            phi = np.array([np.squeeze(self.phi @ state, axis=2).transpose() for state, _, _ in self.data])
            dx = np.array([derivative for _, _, derivative in self.data])
            ax = np.array([self.A @ state for state, _, _ in self.data])
            bu = np.array([self.B @ control for _, control, _ in self.data])
            y = dx - ax - bu
            g_n = np.sum([np.transpose(phi_n) @ phi_n for phi_n in phi], axis=0)
            try:
                g_n_inv = np.linalg.inv(g_n)
                theta_n = (g_n_inv @ np.sum(
                    [np.transpose(phi[n]) @ y[n] for n in range(y.shape[0])], axis=0)).squeeze(axis=1)
                theta_n = theta_n.clip(-self.config["parameter_bound"], self.config["parameter_bound"])
                beta_n = np.linalg.norm(g_n_inv) * sum(np.linalg.norm(phi_n) for phi_n in phi) * self.config["noise_bound"]
            except LinAlgError:
                theta_n = np.zeros(d)
                g_n = np.eye(d)
                beta_n = self.config["parameter_bound"]

        return theta_n, g_n, beta_n

    def polytope(self):
        """
            Confidence polytope, computed from the confidence ellipsoid.
        :return: nominal matrix A0, list of vertices dA, such that A(theta) = A0 + alpha^T dA.
        """
        theta_n, _, beta_n = self.ellipsoids[-1]
        d = theta_n.shape[0]
        h = np.array(list(itertools.product([-1, 1], repeat=d)))
        d_theta_k = np.clip([beta_n * h_k for h_k in h],
                            -theta_n - self.config["parameter_bound"], -theta_n + self.config["parameter_bound"])
        a0 = self.A + np.tensordot(theta_n, self.phi, axes=[0, 0])
        da = [np.tensordot(d_theta, self.phi, axes=[0, 0]) for d_theta in d_theta_k]
        return a0, da

    def plan(self, observation):
        a0, da = self.polytope()
        self.config.update({
            "A0": a0,
            "dA": np.array(da)/10,
        })
        if not self.feedback:
            self.feedback = IntervalFeedback(self.env, self.config)
        observation["interval_min"] = observation["state"] - 0.1*np.abs(observation["state"])
        observation["interval_max"] = observation["state"] + 0.1*np.abs(observation["state"])
        observation["perturbation_min"] = [[self.config["perturbation_bound"]]]
        observation["perturbation_max"] = [[-self.config["perturbation_bound"]]]
        action = self.feedback.act(observation)
        return [action]

    def get_plan(self):
        return [0]


class IntervalWrapper(Wrapper):
    def __init__(self, lpv):
        self.lpv = lpv

    def step(self, action):
        pass



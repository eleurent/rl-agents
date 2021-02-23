import itertools

import gym
import numpy as np
from gym import Wrapper
from numpy.linalg import LinAlgError

from rl_agents.agents.common.factory import safe_deepcopy_env
from rl_agents.agents.control.interval_feedback import IntervalFeedback
from rl_agents.agents.robust.robust_epc import RobustEPCAgent


class ConstrainedEPCAgent(RobustEPCAgent):
    """
        Robust Estimation, Prediction and Control.
    """
    def __init__(self, env, config=None):
        super().__init__(env, config)
        self.feedback = IntervalFeedback(self.env, config)
        self.iteration = 0

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update({
            "noise_bound": 1,
            "perturbation_bound": 1,
            "update_frequency": 1
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
            theta_n = (np.array(self.config["parameter_box"][0]) + np.array(self.config["parameter_box"][1])) / 2
            g_n = np.eye(d)
            beta_n = np.sqrt(d) * self.config["parameter_bound"] / 2
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
                theta_n = theta_n.clip(self.config["parameter_box"][0], self.config["parameter_box"][1])
                beta_n = np.linalg.norm(g_n_inv) * sum(np.linalg.norm(phi_n) for phi_n in phi) * self.config["noise_bound"]
            except LinAlgError:
                theta_n = (np.array(self.config["parameter_box"][0]) + np.array(self.config["parameter_box"][1])) / 2
                g_n = np.eye(d)
                beta_n = np.sqrt(d) * self.config["parameter_bound"] / 2

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
                            -theta_n + self.config["parameter_box"][0], -theta_n + self.config["parameter_box"][1])
        a0 = self.A + np.tensordot(theta_n, self.phi, axes=[0, 0])
        da = [np.tensordot(d_theta, self.phi, axes=[0, 0]) for d_theta in d_theta_k]
        return a0, da

    def robustify_env(self):
        """
            Important distinction with RobustEPC: the nominal lpv model is stabilized.

            We start with a system:
                dx = A(theta)x + Bu + omega,
            that we first stabilize with u0 = Kx, without constraint satisfaction.
            Then, we predict the interval of the stabilized system under additional controls:
                dx = (A(theta) + BK)x + Bu' + omega
            where A0 + BK is stable, which eases the similarity transformation to a Metlzer system.
        """
        from highway_env.interval import LPV
        a0, da = self.config["A0"], self.config["dA"]
        K = 2 * self.feedback.K0[:, :(self.feedback.K0.shape[1] // 2)]
        da = da / 100
        # da = [np.zeros(a0.shape)]
        lpv = LPV(a0=a0, da=da, x0=self.env.unwrapped.state.squeeze(-1),
                  b=self.B, d=self.config["D"], k=K, omega_i=self.config["omega"])
        robust_env = safe_deepcopy_env(self.env)
        robust_env.unwrapped.lpv = lpv
        robust_env.unwrapped.automatic_record_callback = None
        return robust_env

    def update_model_and_controller(self):
            a0, da = self.polytope()
            self.config.update({
                "A0": a0,
                "dA": np.array(da),
            })
            self.config.update({"K0": None})
            self.feedback.update_config(self.config)
            self.feedback.reset()

    def act(self, observation):
        observation["interval_min"] = observation["state"]
        observation["interval_max"] = observation["state"]
        self.observation = observation

        if self.iteration < self.config["update_frequency"] or self.iteration % self.config["update_frequency"] == 0:
            self.update_model_and_controller()
        action = self.feedback.act(observation)
        return action

    def plan(self, observation):
        action = self.act(observation)
        self.iteration += 1
        return [action]

    def get_plan(self):
        return [0]


class IntervalWrapper(Wrapper):
    def __init__(self, lpv):
        self.lpv = lpv

    def step(self, action):
        pass



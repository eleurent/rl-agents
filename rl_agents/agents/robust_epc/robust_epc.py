import itertools

import numpy as np

from highway_env.interval import LPV
from rl_agents.agents.common.abstract import AbstractAgent


class RobustEPCAgent(AbstractAgent):
    """
        Cross-Entropy Method planner.
        The environment is copied and used as an oracle model to sample trajectories.
    """
    def __init__(self, env, config):
        super().__init__(config)
        self.env = env
        self.data = []

    @classmethod
    def default_config(cls):
        return {
            "gamma": 0.9,
            "delta": 0.9,
            "lambda": 1,
            "sigma": [[1]],
            "A": [[1]],
            "phi": [[[1]]],
            "dt": 1,
            "parameter_bound": 1
        }

    def record(self, state, action, reward, next_state, done, info):
        self.data.append((state, action, next_state))

    # def step(self, action):
    #
    #
    def ellipsoid(self):
        phi = np.array([self.config["phi"] @ state for state, _, _ in self.data])
        dx = np.array([(next_state - state) / self.config["dt"] for state, _, next_state in self.data])
        bu = np.array([self.config["B"] @ action for _, action, _ in self.data])
        y = dx - bu

        lambda_ = self.config["lambda"]
        sigma_inv = np.linalg.inv(self.config["sigma"])
        g_n = np.sum([np.transpose(phi_n) @ sigma_inv @ phi_n for phi_n in phi], axis=0)
        d = g_n.shape[0]
        g_n_lambda = g_n + lambda_ * np.identity(d)

        theta_n_lambda = np.linalg.inv(g_n_lambda) @ np.sum([np.transpose(phi_n) @ sigma_inv @ y_n
                                                             for phi_n in phi for y_n in y])
        beta_n = np.sqrt(2*np.log(np.sqrt(np.linalg.det(g_n_lambda) / lambda_ ** d) / self.config["delta"])) \
                 + np.sqrt(lambda_*d) * self.config["parameter_bound"]
        return theta_n_lambda, g_n_lambda, beta_n

    def polytope(self):
        theta_n_lambda, g_n_lambda, beta_n = self.ellipsoid()
        d = g_n_lambda.shape[0]
        values, p = np.linalg.eig(g_n_lambda)
        m = np.sqrt(beta_n) * np.linalg.inv(p) @ np.diag(np.sqrt(1 / values))
        h = np.array(list(itertools.product([-1, 1], repeat=d)))
        d_theta_k = [m @ h_k for h_k in h]

        a0 = self.config["A"] + np.tensordot(theta_n_lambda, self.config["phi"], axis=0)
        da = [np.tensordot(d_theta, self.config["phi"], axis=0) for d_theta in d_theta_k]
        return a0, da

    def act(self, observation):
        a0, da = self.polytope()
        lpv = LPV(a0=a0, da=da, x0=observation)
        for _ in range(self.config["horizon"]):
            lpv.step(self.config["dt"])


class RobustEnv(object):
    def __init__(self, lpv, config):
        self.lpv = lpv
        self.config = config

    def step(self, action):
        self.lpv.step(self.config["dt"])

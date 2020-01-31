import itertools

import numpy as np

from highway_env.interval import LPV
from rl_agents.agents.common.abstract import AbstractAgent
from rl_agents.agents.common.factory import load_agent, safe_deepcopy_env


class RobustEPCAgent(AbstractAgent):
    """
        Cross-Entropy Method planner.
        The environment is copied and used as an oracle model to sample trajectories.
    """
    def __init__(self, env, config):
        super().__init__(config)
        self.A = np.array(self.config["A"])
        self.B = np.array(self.config["B"])
        self.phi = np.array(self.config["phi"])
        self.env = env
        self.env.unwrapped.automatic_record_callback = self.automatic_record
        self.data = []
        self.robust_env = None
        self.sub_agent = load_agent(self.config['sub_agent_path'], env)
        self.theta_n_lambda = None
        self.g_n_lambda = None
        self.beta_n = None

    @classmethod
    def default_config(cls):
        return {
            "gamma": 0.9,
            "delta": 0.9,
            "lambda": 0.0001,
            "sigma": [[1]],
            "A": [[1]],
            "B": [[1]],
            "D": [[1]],
            "omega": [[0], [0]],
            "phi": [[[1]]],
            "simulation_frequency": 10,
            "policy_frequency": 2,
            "parameter_bound": 1
        }

    def record(self, state, action, reward, next_state, done, info):
        control = self.env.unwrapped.dynamics.action_to_control(action)
        derivative = self.env.unwrapped.dynamics.derivative
        self.data.append((state.copy(), control.copy(), derivative.copy()))

    def automatic_record(self, state, derivative, control):
        self.data.append((state.copy(), control.copy(), derivative.copy()))

    def plan(self, observation):
        self.robust_env = self.robustify_env()
        self.sub_agent.env = self.robust_env
        return self.sub_agent.plan(observation)

    def get_plan(self):
        return self.sub_agent.planner.get_plan()

    def reset(self):
        return self.sub_agent.reset()

    def seed(self, seed=None):
        return self.sub_agent.seed(seed)

    def ellipsoid(self):
        d = self.phi.shape[0]
        lambda_ = self.config["lambda"]
        if not self.data:
            g_n_lambda = lambda_ * np.identity(d)
            theta_n_lambda = np.zeros((d, 1))
        else:
            phi = np.array([np.squeeze(self.phi @ state, axis=2).transpose() for state, _, _ in self.data])
            dx = np.array([derivative for _, _, derivative in self.data])
            ax = np.array([self.A @ state for state, _, _ in self.data])
            bu = np.array([self.B @ control for _, control, _ in self.data])
            y = dx - ax - bu

            sigma_inv = np.linalg.inv(self.config["sigma"])
            g_n = np.sum([np.transpose(phi_n) @ sigma_inv @ phi_n for phi_n in phi], axis=0)
            g_n_lambda = g_n + lambda_ * np.identity(d)

            theta_n_lambda = np.linalg.inv(g_n_lambda) @ np.sum(
                [np.transpose(phi[n]) @ sigma_inv @ y[n] for n in range(y.shape[0])], axis=0)
        beta_n = np.sqrt(2*np.log(np.sqrt(np.linalg.det(g_n_lambda) / lambda_ ** d) / self.config["delta"])) \
                 + np.sqrt(lambda_*d) * self.config["parameter_bound"]
        return theta_n_lambda.squeeze(axis=1), g_n_lambda, beta_n

    def polytope(self):
        theta_n_lambda, g_n_lambda, beta_n = self.ellipsoid()
        self.theta_n_lambda = theta_n_lambda
        self.g_n_lambda = g_n_lambda
        self.beta_n = beta_n
        d = g_n_lambda.shape[0]
        values, p = np.linalg.eig(g_n_lambda)
        m = np.sqrt(beta_n) * np.linalg.inv(p) @ np.diag(np.sqrt(1 / values))
        h = np.array(list(itertools.product([-1, 1], repeat=d)))
        d_theta_k = np.clip([m @ h_k for h_k in h], -self.config["parameter_bound"], self.config["parameter_bound"])

        a0 = self.A + np.tensordot(theta_n_lambda, self.phi, axes=[0, 0])
        da = [np.tensordot(d_theta, self.phi, axes=[0, 0]) for d_theta in d_theta_k]
        return a0, da

    def act(self, state):
        return self.plan(state)[0]

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def robustify_env(self):
        a0, da = self.polytope()
        lpv = LPV(a0=a0, da=da, x0=self.env.unwrapped.dynamics.state.squeeze(-1),
                  b=self.config["D"], d_i=self.config["omega"])
        robust_env = safe_deepcopy_env(self.env)
        robust_env.unwrapped.lpv = lpv
        robust_env.unwrapped.automatic_record_callback = None
        return robust_env

#
# class RobustEnv(object):
#     def __init__(self, true_env, lpv, config):
#         self.true_env = true_env
#         self.lpv = lpv
#         self.config = config
#         self.action_space = true_env.action_space
#         self.interval_trajectory = []
#         self.time = true_env.time
#
#     def step(self, action):
#         control = self.true_env.unwrapped.dynamics.action_to_control(action)
#         self.lpv.set_control(np.array(self.config["B"]) @ control)
#         for _ in range(self.config["simulation_frequency"] // self.config["policy_frequency"]):
#             self.interval_trajectory.append(self.lpv.x_i_t)
#             self.lpv.step(1 / self.config["simulation_frequency"])
#         return self.lpv.x_i_t, self._reward(), self._done(), {}
#
#     def _reward(self):
#         return self.true_env.unwrapped.pessimistic_reward(self.lpv.x_i_t)
#
#     def _done(self):
#         return self.true_env.unwrapped.pessimistic_is_terminal(self.lpv.x_i_t)

import itertools
import numpy as np

from rl_agents.agents.common.abstract import AbstractAgent
from rl_agents.agents.common.factory import load_agent, safe_deepcopy_env


class RobustEPCAgent(AbstractAgent):
    """
        Robust Estimation, Prediction and Control.
    """

    def __init__(self, env, config):
        super().__init__(config)
        self.A = np.array(self.config["A"])
        self.B = np.array(self.config["B"])
        self.phi = np.array(self.config["phi"])
        self.env = env
        if hasattr(self.env.unwrapped, "automatic_record_callback"):
            self.env.unwrapped.automatic_record_callback = self.record_transition
        self.data = []
        self.robust_env = None
        self.sub_agent = load_agent(self.config['sub_agent_path'], env)
        self.ellipsoids = [self.ellipsoid()]

    @classmethod
    def default_config(cls):
        return {
            "gamma": 0.9,
            "delta": 0.9,
            "lambda": 1e-6,
            "sigma": [[1]],
            "A": [[1]],
            "B": [[1]],
            "D": [[1]],
            "omega": [[0], [0]],
            "phi": [[[1]]],
            "simulation_frequency": 10,
            "policy_frequency": 2,
            "parameter_bound": 1,
            "parameter_box": [[0], [1]],
        }

    def record(self, observation, action, reward, next_observation, done, info):
        """
            Add a transition to the dataset D_[N].

            The state space should be a Dict with two fields: "state" and "derivative"
            The environment might be able to convert a discrete action to a continuous control.
        :param observation: state x_t and derivative \dot{x}_t at time t
        :param action: action a_t performed
        :param reward: reward r_t obtained
        :param next_observation: next state and derivative at time t+dt
        :param done: is the state terminal
        :param info: information about the transition
        """
        if hasattr(self.env.unwrapped, "automatic_record_callback"):
            return
        try:
            control = self.env.unwrapped.dynamics.action_to_control(action)
        except AttributeError:
            control = np.array([action])
        state = next_observation["state"]
        derivative = next_observation["derivative"]
        self.record_transition(state, derivative, control)

    def record_transition(self, state, derivative, control):
        """
            A callback that can be called by the environment in place of RobustEPCAgent.record(),
            in order to record several transitions when the environment simulates an action execution.
        :param state: state x_t at time t
        :param derivative: state derivative dot x_t at time t
        :param control: control u_t at time t
        """
        self.data.append((state.copy(), control.copy(), derivative.copy()))
        self.ellipsoids.append(self.ellipsoid())

    def plan(self, observation):
        """
            Perform OPD planning with make a pessimistic version of the environment, that propagates
            state intervals and computes pessimistic rewards.
        """
        self.robust_env = self.robustify_env()
        self.sub_agent.env = self.robust_env
        return self.sub_agent.plan(observation)

    def ellipsoid(self):
        """
            Compute a confidence ellipsoid over the parameter theta, where
                    dot{x} = A x + (phi x) theta + B u + D omega
                         y = dot{x} + C nu
            under a sub-Gaussian noise assumption.
        :return: estimated theta, Gramian matrix G_N_lambda, and radius beta_N_lambda
        """
        d = self.phi.shape[0]
        lambda_ = self.config["lambda"]
        if not self.data:
            g_n_lambda = lambda_ * np.identity(d)
            theta_n_lambda = np.zeros(d)
        else:
            phi = np.array([np.squeeze(self.phi @ state, axis=2).transpose() for state, _, _ in self.data])
            dx = np.array([derivative for _, _, derivative in self.data])
            ax = np.array([self.A @ state for state, _, _ in self.data])
            bu = np.array([self.B @ control for _, control, _ in self.data])
            y = dx - ax - bu

            sigma_inv = np.linalg.inv(self.config["sigma"])
            g_n = np.sum([np.transpose(phi_n) @ sigma_inv @ phi_n for phi_n in phi], axis=0)
            g_n_lambda = g_n + lambda_ * np.identity(d)

            theta_n_lambda = (np.linalg.inv(g_n_lambda) @ np.sum(
                [np.transpose(phi[n]) @ sigma_inv @ y[n] for n in range(y.shape[0])], axis=0)).squeeze(axis=1)
            theta_n_lambda = theta_n_lambda.clip(0, 1)
        beta_n = \
            np.sqrt(2 * np.log(np.sqrt(np.linalg.det(g_n_lambda) / lambda_ ** d) / self.config["delta"])) \
            + np.sqrt(lambda_ * d) * self.config["parameter_bound"]
        return theta_n_lambda, g_n_lambda, beta_n

    def polytope(self):
        """
            Confidence polytope, computed from the confidence ellipsoid.
        :return: nominal matrix A0, list of vertices dA, such that A(theta) = A0 + alpha^T dA.
        """
        theta_n_lambda, g_n_lambda, beta_n = self.ellipsoids[-1]
        d = g_n_lambda.shape[0]
        values, p = np.linalg.eig(g_n_lambda)
        m = beta_n * np.linalg.inv(p) @ np.diag(np.sqrt(1 / values))
        h = np.array(list(itertools.product([-1, 1], repeat=d)))
        d_theta_k = np.clip([m @ h_k for h_k in h], -self.config["parameter_bound"], self.config["parameter_bound"])
        a0 = self.A + np.tensordot(theta_n_lambda, self.phi, axes=[0, 0])
        da = [np.tensordot(d_theta, self.phi, axes=[0, 0]) for d_theta in d_theta_k]
        return a0, da

    def robustify_env(self):
        """
            Make a robust version of the environment:
                1. compute the dynamics polytope (A0, dA)
                2. set the LPV interval predictor, so that it can be stepped with the environment
                3. the environment, when provided with an interval predictor, should return pessimistic rewards
                4. disable the recording of environment transitions, since we are not observing when planning.
        :return: the robust version of the environment.
        """
        a0, da = self.polytope()
        from highway_env.interval import LPV
        lpv = LPV(a0=a0, da=da, x0=self.env.unwrapped.state.squeeze(-1), b=self.B,
                  d=self.config["D"], omega_i=self.config["omega"])
        robust_env = safe_deepcopy_env(self.env)
        robust_env.unwrapped.lpv = lpv
        robust_env.unwrapped.automatic_record_callback = None
        return robust_env

    def act(self, state):
        return self.plan(state)[0]

    def get_plan(self):
        return self.sub_agent.planner.get_plan()

    def reset(self):
        self.data = []
        self.ellipsoids = [self.ellipsoid()]
        return self.sub_agent.reset()

    def seed(self, seed=None):
        return self.sub_agent.seed(seed)

    def save(self, filename):
        pass

    def load(self, filename):
        pass


class NominalEPCAgent(RobustEPCAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.config["omega"] = np.zeros(np.shape(self.config["omega"])).tolist()

    def polytope(self):
        """
            Do not consider uncertainty over the estimated dynamics.
        """
        a0, da = super().polytope()
        da = [np.zeros(a0.shape)]
        return a0, da

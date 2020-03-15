import numpy as np

from rl_agents.agents.control.linear_feedback import LinearFeedbackAgent
from rl_agents.utils import pos, neg


class IntervalFeedback(LinearFeedbackAgent):
    def __init__(self, env, config=None):
        super().__init__(env, config)
        self.env = env
        self.K0 = np.array(self.config["K0"])
        self.K1 = np.array(self.config["K1"])
        self.K2 = np.array(self.config["K2"])
        self.S = np.array(self.config["S"])
        self.D = np.array(self.config["D"])
        if self.config["K0"] is None:
            self.synthesize_controller()

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update({
            "K0": None,
            "K1": None,
            "K2": None,
            "S": None,
            "D": None,
            "discrete": False
        })
        return cfg

    def act(self, observation):
        if not isinstance(observation, dict):
            raise ValueError("The observation should be a dict containing the two interval bounds")
        x_m = observation["interval_min"]
        x_M = observation["interval_max"]
        xi = np.concatenate((x_m, x_M))
        omega_m = observation["perturbation_min"]
        omega_M = observation["perturbation_max"]
        delta = np.array([[pos(self.D), -neg(self.D)],
                          [-neg(self.D), pos(self.D)]]) \
                @ np.concatenate((omega_m, omega_M))
        control = self.K0 @ xi + self.K1 @ pos(xi) + self.K2 * neg(xi) + self.S @ delta
        return control.squeeze(-1)

    def synthesize_controller(self):
        import cvxpy as cp

        np.set_printoptions(precision=2)

        # Input matrices
        A0 = np.array(self.config["A0"])
        dA = np.array(self.config["dA"])
        B = np.array(self.config["B"])
        print("A0:\n", A0)
        print("dA:\n", dA)
        print("B:\n", B)
        dAp = sum(pos(dAi) for dAi in dA)
        dAn = sum(neg(dAi) for dAi in dA)
        p, q = int(B.shape[0]), int(B.shape[1])

        # Extended matrices
        zero = np.zeros((p, p))
        cA0 = np.concatenate((np.concatenate((A0, zero), axis=1),
                              np.concatenate((zero, A0), axis=1)))
        cA1 = np.concatenate((np.concatenate((zero, -dAn), axis=1),
                              np.concatenate((zero, dAp), axis=1)))
        cA2 = np.concatenate((np.concatenate((-dAp, zero), axis=1),
                              np.concatenate((dAn, zero), axis=1)))
        cB = np.concatenate((B, B))

        # Optimisation variables
        P = cp.Variable((2*p,))  # In fact P^{-1}
        Q = cp.Variable((2*p,))
        Qp = cp.Variable((2*p,))
        Qn = cp.Variable((2*p,))
        Zp = cp.Variable((2*p,))  # In fact Zp^{-1}
        Zn = cp.Variable((2*p,))  # In fact Zn^{-1}
        Psi = cp.Variable((2*p,))
        Psi_p = cp.Variable((2*p,))
        Psi_n = cp.Variable((2*p,))
        Tau = cp.Variable((2*p,))
        U0 = cp.Variable((q, 2*p))
        U1 = cp.Variable((q, 2*p))
        U2 = cp.Variable((q, 2*p))

        # Constraints
        Pi_11 = cp.diag(P)*cA0.T + cA0*cp.diag(P) + U0.T*cB.T + cB*U0 + cp.diag(Q)
        Pi_12 = cA1*cp.diag(Zp) + cB*U1 + cp.diag(P)*cA0.T + U0.T*cB.T + cp.diag(Psi_p)
        Pi_13 = cA2*cp.diag(Zn) + cB*U2 - cp.diag(P)*cA0.T - U0.T*cB.T - cp.diag(Psi_n)
        Pi_22 = cp.diag(Zp)*cA1.T + cA1*cp.diag(Zp) + U1.T*cB.T + cB*U1 + cp.diag(Qp)
        Pi_23 = cA2*cp.diag(Zn) + cB*U2 - cp.diag(Zp)*cA1.T - U1.T*cB.T + cp.diag(Psi)
        Pi_33 = cp.diag(Qn) - cp.diag(Zn)*cA2.T - cA2*cp.diag(Zn) - U2.T*cB.T - cB*U2
        Id = np.eye(2*p)
        Pi = cp.bmat([  # Block matrix
            [Pi_11,   Pi_12,    Pi_13,  Id],
            [Pi_12.T, Pi_22,    Pi_23,  Id],
            [Pi_13.T, Pi_23.T,  Pi_33, -Id],
            [Id,      Id,      -Id,    -cp.diag(Tau)]
        ])

        C = Q + cp.minimum(Qp, Qn) + 2*cp.minimum(Psi_p, Psi_n)

        constraints = [
            P >= 0,
            Zp >= 0,
            Zn >= 0,
            Tau >= 0,
            C >= 0,
            Pi << 0
        ]

        prob = cp.Problem(cp.Minimize(0), constraints=constraints)
        prob.solve(solver=cp.SCS, verbose=True)

        print("Status:", prob.status)
        print("Matrices:")
        print("- P:", P.value)
        print("- Q:", Q.value)
        print("- Qp:", Qp.value)
        print("- Qn:", Qn.value)
        print("- Zp:", Zp.value)
        print("- Zn:", Zn.value)
        print("- Psi:", Psi.value)
        print("- Psi_p:", Psi_p.value)
        print("- Psi_n:", Psi_n.value)
        print("- Tau:", Tau.value)
        print("- U0:", U0.value)
        print("- U1:", U1.value)
        print("- U2:", U2.value)
        print("- Pi:\n", Pi.value)
        print("Constraints:")
        print("- Q + cp.minimum(Qp, Qn) + 2*cp.minimum(Psi_p, Psi_n):", C.value)
        if Pi.value is not None:
            print("- lambda(Pi):", np.linalg.eigvals(Pi.value))

        self.K0 = U0.value @ np.diag(1 / P.value)
        self.K1 = U1.value @ np.diag(1 / Zp.value)
        self.K2 = U2.value @ np.diag(1 / Zn.value)

    def example(self):
        A0 = np.array(
            [[-1., 0.],
             [0., -2.]])
        dA = np.array([
            [[0., 0.],
             [0., 0.]],

            [[0.1, 0.],
             [0., 0.1]],
        ])
        B = np.array(
            [[1., 0.],
             [0., 1.]]
        )
        self.config.update({"A0": A0, "dA": dA, "B": B})
        self.synthesize_controller()

    # def example(self):
    #     import cvxpy as cp
    #     A = np.array([[0, 1],
    #                   [-1, -1]])
    #     p = A.shape[0]
    #     B = cp.Variable((2*p, 2*p), PSD=True)
    #
    #     tau = cp.bmat([
    #             [A,  np.zeros((p, p))],
    #             [np.zeros((p, p)), A.T]
    #               ]) + B
    #
    #     cons1 = tau >> 0
    #     cons2 = B == B.T
    #
    #     prob = cp.Problem(cp.Minimize(0), constraints=[cons1, cons2])
    #     prob.solve(solver=cp.SCS)
    #     print("status", prob.status, "value", prob.value)
    #
    #     print("B")
    #     print(B.value)
    #     print(np.linalg.eigvals(B.value))
    #     print("tau")
    #     print(tau.value)
    #     print(np.linalg.eigvals(tau.value))

    def reset(self):
        pass

    def seed(self, seed=None):
        pass

    def save(self, filename):
        return False

    def load(self, filename):
        return False

    def record(self, state, action, reward, next_state, done, info):
        pass


if __name__ == '__main__':
    agent = IntervalFeedback(None)
    # agent.example()
    agent.synthesize_controller()



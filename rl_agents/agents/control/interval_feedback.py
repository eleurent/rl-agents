import numpy as np
import logging

from rl_agents.agents.control.linear_feedback import LinearFeedbackAgent
from rl_agents.trainer.logger import configure
from rl_agents.utils import pos, neg

logger = logging.getLogger(__name__)


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

    def synthesize_controller(self, pole_placement=True):
        """
            Synthesize a controller the interval predictor via an LMI
        :param pole_placement: use pole placement to synthesize the controller instead, and check stability
        :return: the control gains
        """

        # Input matrices
        A0 = np.array(self.config["A0"])
        dA = np.array(self.config["dA"])
        B = np.array(self.config["B"])
        logger.debug("A0:\n{}".format(A0))
        logger.debug("dA:\n{}".format(dA))
        logger.debug("B:\n{}".format(B))
        dAp = sum(pos(dAi) for dAi in dA)
        dAn = sum(neg(dAi) for dAi in dA)
        DA = sum(dAi for dAi in dA)
        p, q = int(B.shape[0]), int(B.shape[1])
        if pole_placement:
            import control
            logger.debug("The eigenvalues of the matrix A0 = {},  Uncontrollable states = {}".format(
                np.linalg.eigvals(A0), p - np.rank(control.ctrb(A0, B))))
            K = -control.place(A0, B, np.linalg.eigvals(A0)-np.arange(1, p+1))
            logger.debug("The eigenvalues of the matrix A0+BK = {}".format(np.linalg.eigvals(A0+B*K)))
            logger.debug("The eigenvalues of the matrix A0+BK+DA = {}".format(np.linalg.eigvals(A0+B*K+DA)))
            logger.debug("The eigenvalues of the matrix A0+BK-DA = {}".format(np.linalg.eigvals(A0+B*K-DA)))

        # Extended matrices
        zero = np.zeros((p, p))
        cA0 = np.concatenate((np.concatenate((A0, zero), axis=1),
                              np.concatenate((zero, A0), axis=1)))
        cA1 = np.concatenate((np.concatenate((zero, -dAn), axis=1),
                              np.concatenate((zero, dAp), axis=1)))
        cA2 = np.concatenate((np.concatenate((-dAp, zero), axis=1),
                              np.concatenate((dAn, zero), axis=1)))
        cB = np.concatenate((B, B))
        if pole_placement:
            K0 = 0.5*np.concatenate((K, K), axis=1)
            cA0 = cA0 + cB@K0
            self.synthesis_lmi(cA0, cA1, cA2, cB, stab_check_only=True)
        else:
            success = self.synthesis_lmi(cA0, cA1, cA2, cB, stab_check_only=False)
            if not success:
                self.synthesize_controller(pole_placement=True)

    def synthesis_lmi(self, cA0, cA1, cA2, cB, stab_check_only=False):
        import cvxpy as cp
        np.set_printoptions(precision=2)
        p = cB.shape[0] // 2
        q = cB.shape[1]

        # Optimisation variables
        P = cp.Variable((2*p, 2*p), diag=True)  # In fact P^{-1}
        Q = cp.Variable((2*p, 2*p), diag=True)
        Qp = cp.Variable((2*p, 2*p), diag=True)
        Qn = cp.Variable((2*p, 2*p), diag=True)
        Zp = cp.Variable((2*p, 2*p), diag=True)  # In fact Zp^{-1}
        Zn = cp.Variable((2*p, 2*p), diag=True)  # In fact Zn^{-1}
        Psi = cp.Variable((2*p, 2*p), diag=True)
        Psi_p = cp.Variable((2*p, 2*p), diag=True)
        Psi_n = cp.Variable((2*p, 2*p), diag=True)
        Gamma = cp.Variable((2*p, 2*p), diag=True)

        # Constraints
        if stab_check_only:
            Pi_11 = cA0.T*P + P*cA0 + Q
            Pi_12 = cA0.T*Zp + P*cA1 + Psi_p
            Pi_13 = P*cA2 - cA0.T*Zn - Psi_n
            Pi_22 = Zp*cA1 + cA1.T*Zp + Qp
            Pi_23 = Zp*cA2 - cA1.T*Zn + Psi
            Pi_33 = Qn - Zn * cA2 - cA2.T*Zn
            Pi = cp.bmat([  # Block matrix
                [Pi_11,   Pi_12,    Pi_13,  P],
                [Pi_12.T, Pi_22,    Pi_23,  Zp],
                [Pi_13.T, Pi_23.T,  Pi_33, -Zn],
                [P,       Zp,      -Zn,    -Gamma]
            ])
            U0, U1, U2 = None, None, None
        else:
            U0 = cp.Variable((q, 2*p))
            U1 = cp.Variable((q, 2*p))
            U2 = cp.Variable((q, 2*p))

            Pi_11 = P*cA0.T + cA0*P + U0.T*cB.T + cB*U0 + Q
            Pi_12 = cA1*Zp + cB*U1 + P*cA0.T + U0.T*cB.T + Psi_p
            Pi_13 = cA2*Zn + cB*U2 - P*cA0.T - U0.T*cB.T - Psi_n
            Pi_22 = Zp*cA1.T + cA1*Zp + U1.T*cB.T + cB*U1 + Qp
            Pi_23 = cA2*Zn + cB*U2 - Zp*cA1.T - U1.T*cB.T + Psi
            Pi_33 = Qn - Zn*cA2.T - cA2*Zn - U2.T*cB.T - cB*U2
            Id = np.eye(2*p)
            Pi = cp.bmat([  # Block matrix
                [Pi_11,   Pi_12,    Pi_13,  Id],
                [Pi_12.T, Pi_22,    Pi_23,  Id],
                [Pi_13.T, Pi_23.T,  Pi_33, -Id],
                [Id,      Id,      -Id,    -Gamma]
            ])

        C = Q + cp.minimum(Qp, Qn) + 2*cp.minimum(Psi_p, Psi_n)

        constraints = [
            P + cp.minimum(Zp, Zn) >= 0,
            Gamma >= 0,
            C >= 0,
            Pi << 0
        ]

        prob = cp.Problem(cp.Minimize(0), constraints=constraints)
        prob.solve(solver=cp.SCS, verbose=True)

        print("Status:", prob.status)
        if prob.status == "infeasible_inaccurate":
            return False
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
        print("- Gamma:", Gamma.value)
        print("- Pi:\n", Pi.value)
        print("Constraints:")
        print("- Q + cp.minimum(Qp, Qn) + 2*cp.minimum(Psi_p, Psi_n):", C.value)
        if Pi.value is not None:
            print("- lambda(Pi):", np.linalg.eigvals(Pi.value))

        if not stab_check_only:
            print("- U0:", U0.value)
            print("- U1:", U1.value)
            print("- U2:", U2.value)
            self.K0 = U0.value @ np.diag(1 / P.value)
            self.K1 = U1.value @ np.diag(1 / Zp.value)
            self.K2 = U2.value @ np.diag(1 / Zn.value)
        return True

    @staticmethod
    def example():
        A0 = np.array(
            [[0., 1.],
             [0., 0.]])
        dA = np.array([
            [[0., 0.],
             [0., 0.]],

            [[0., 0.1],
             [0., 0.]],
        ])
        B = np.array(
            [[1.],
             [0.]]
        )
        config = {"A0": A0, "dA": dA, "B": B}
        return IntervalFeedback(None, config)

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
    configure("configs/verbose.json")
    IntervalFeedback.example()



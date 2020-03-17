import numpy as np
import logging

from rl_agents.agents.control.linear_feedback import LinearFeedbackAgent
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
            self.synthesize_controller(self.config["pole_placement"], self.config["ensure_stability"])
        if config["S"] is None:
            self.synthesize_perturbation_rejection()

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update({
            "K0": None,
            "K1": None,
            "K2": None,
            "S": None,
            "D": None,
            "discrete": False,
            "pole_placement": False,
            "ensure_stability": True
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
        delta = np.concatenate((np.concatenate((pos(self.D), -neg(self.D)), axis=1),
                                np.concatenate((-neg(self.D), pos(self.D)), axis=1)))
        delta = delta @ np.concatenate((omega_m, omega_M))
        control = self.K0 @ xi + self.K1 @ pos(xi) + self.K2 @ neg(xi) + self.S @ delta
        return np.asarray(control).squeeze(-1)

    def synthesize_controller(self, pole_placement=False, ensure_stability=True):
        """
            Synthesize a controller the interval predictor via an LMI
        :param pole_placement: use pole placement to synthesize the controller instead
        :param ensure_stability: whether we need to check stability when the pole placement method is used
        :return: whether we found stabilising controls (True if ensure_stability is False)
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

        # Extended matrices
        zero = np.zeros((p, p))
        cA0 = np.concatenate((np.concatenate((A0, zero), axis=1),
                              np.concatenate((zero, A0), axis=1)))
        cA1 = np.concatenate((np.concatenate((zero, -dAn), axis=1),
                              np.concatenate((zero, dAp), axis=1)))
        cA2 = np.concatenate((np.concatenate((-dAp, zero), axis=1),
                              np.concatenate((dAn, zero), axis=1)))
        cB = np.concatenate((B, B))

        # Pole placement
        if pole_placement:
            import control
            logger.debug("The eigenvalues of the matrix A0 = {},  Uncontrollable states = {}".format(
                np.linalg.eigvals(A0), p - np.rank(control.ctrb(A0, B))))
            K = -control.place(A0, B, 0*np.linalg.eigvals(A0)-np.arange(1, p+1))
            logger.debug("The eigenvalues of the matrix A0+BK = {}".format(np.linalg.eigvals(A0+B*K)))
            logger.debug("The eigenvalues of the matrix A0+BK+DA = {}".format(np.linalg.eigvals(A0+B*K+DA)))
            logger.debug("The eigenvalues of the matrix A0+BK-DA = {}".format(np.linalg.eigvals(A0+B*K-DA)))
            self.K0 = 0.5*np.concatenate((K, K), axis=1)
            self.K1 = self.K2 = np.zeros(self.K0.shape)
            cA0 += cB @ self.K0
            if not ensure_stability:
                return True

        # Solve LMI
        success = self.stability_lmi(cA0, cA1, cA2, cB, synthesize_control=not pole_placement)
        # If control synthesis via LMI fails, try pole placement instead
        if not success and not pole_placement:
            success = self.synthesize_controller(pole_placement=True, ensure_stability=ensure_stability)
        return success

    def stability_lmi(self, cA0, cA1, cA2, cB, synthesize_control=True):
        """
            Solve an LMI to check the stability of an interval controller
        :param cA0: extended nominal matrix
        :param cA1: extended state matrix uncertainty
        :param cA2: extended state matrix uncertainty
        :param cB: extended control matrix
        :param synthesize_control: if true, the controls will be synthesized via the LMI
        :return: whether the controlled system is stable
        """
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
        if synthesize_control:
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
        else:
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

        C = Q + cp.minimum(Qp, Qn) + 2*cp.minimum(Psi_p, Psi_n)

        constraints = [
            P + cp.minimum(Zp, Zn) >= 0,
            Gamma >= 0,
            C >= 0,
            Pi << 0
        ]

        prob = cp.Problem(cp.Minimize(0), constraints=constraints)
        prob.solve(solver=cp.SCS, verbose=True)

        logger.debug("Status: {}".format(prob.status))
        success = prob.status == "optimal"
        if success:
            logger.debug("Matrices:")
            logger.debug("- P:".format(P.value))
            logger.debug("- Q:".format(Q.value))
            logger.debug("- Qp:".format(Qp.value))
            logger.debug("- Qn:".format(Qn.value))
            logger.debug("- Zp:".format(Zp.value))
            logger.debug("- Zn:".format(Zn.value))
            logger.debug("- Psi:".format(Psi.value))
            logger.debug("- Psi_p:".format(Psi_p.value))
            logger.debug("- Psi_n:".format(Psi_n.value))
            logger.debug("- Gamma:".format(Gamma.value))
            logger.debug("- Pi:\n".format(Pi.value))
            logger.debug("Constraints:")
            logger.debug("- Q + cp.minimum(Qp, Qn) + 2*cp.minimum(Psi_p, Psi_n):".format(C.value))
            logger.debug("- lambda(Pi):".format(np.linalg.eigvals(Pi.value)))
            if synthesize_control:
                logger.debug("- U0:".format(U0.value))
                logger.debug("- U1:".format(U1.value))
                logger.debug("- U2:".format(U2.value))
                self.K0 = U0.value @ np.linalg.inv(P.value)
                self.K1 = U1.value @ np.linalg.inv(Zp.value)
                self.K2 = U2.value @ np.linalg.inv(Zn.value)
        return success

    def synthesize_perturbation_rejection(self):
        """
            Design S so as to minimize ||cB S + I||_2

            min_S ||cB S + I||_2 is solved by framing it in an equivalent form:
            min lambda s.t. (cB S + I)^T(cB S + I) << lambda I
        """
        B = np.array(self.config["B"])
        p, q = B.shape[0], B.shape[1]
        cB = np.concatenate((B, B))
        I2p = np.eye(2*p)

        # Norm minimization
        import cvxpy as cp
        S = cp.Variable((q, 2*p))
        A = cB*S + I2p
        prob = cp.Problem(cp.Minimize(cp.norm(A, p=2)))
        prob.solve(solver=cp.SCS, verbose=True)
        assert prob.status == "optimal"
        self.S = S.value
        logger.debug("Synthesized perturbation gain S: {}".format(self.S))

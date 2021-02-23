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
        self.Xf = np.array(self.config["Xf"])
        # self.reset()

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
            "ensure_stability": True,
            "control_bound": np.infty
        })
        return cfg

    def reset(self):
        if self.config["S"] is None:
            self.synthesize_perturbation_rejection()
        if self.config["K0"] is None:
            self.synthesize_controller(self.config["pole_placement"], self.config["ensure_stability"])
        super().reset()

    def act(self, observation):
        if not isinstance(observation, dict):
            raise ValueError("The observation should be a dict containing the two interval bounds")
        x_m = observation["interval_min"]
        x_M = observation["interval_max"]
        x_ref = observation["reference_state"]
        xi = np.concatenate((x_m - x_ref, x_M - x_ref))

        control = self.K0 @ xi + self.K1 @ pos(xi) + self.K2 @ neg(xi) + self.S @ self.delta()
        control = np.clip(control, -self.config["control_bound"], self.config["control_bound"])
        return np.asarray(control).squeeze(-1)

    def delta(self):
        """ Extended perturbation interval. """
        omega_m = [[self.config["perturbation_bound"]]]
        omega_M = [[-self.config["perturbation_bound"]]]
        cD = np.concatenate((np.concatenate((pos(self.D), -neg(self.D)), axis=1),
                             np.concatenate((-neg(self.D), pos(self.D)), axis=1)))
        delta = cD @ np.concatenate((omega_m, omega_M))
        return delta

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
                np.linalg.eigvals(A0), p - np.linalg.matrix_rank(control.ctrb(A0, B))))
            poles = self.config.get("poles", np.minimum(np.linalg.eigvals(A0), -np.arange(1, p+1)))
            K = -control.place(A0, B, poles)
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

    def stability_lmi(self, cA0, cA1, cA2, cB, synthesize_control=True, epsilon=1e-9):
        """
            Solve an LMI to check the stability of an interval controller
        :param cA0: extended nominal matrix
        :param cA1: extended state matrix uncertainty
        :param cA2: extended state matrix uncertainty
        :param cB: extended control matrix
        :param synthesize_control: if true, the controls will be synthesized via the LMI
        :param epsilon: accuracy for checking that a matrix is positive definite.
        :return: whether the controlled system is stable
        """
        import cvxpy as cp
        np.set_printoptions(precision=2)
        p = cB.shape[0] // 2
        q = cB.shape[1]

        # Optimisation variables
        P = cp.Variable((2*p, 2*p), diag=True)
        Q = cp.Variable((2*p, 2*p), diag=True)
        Qp = cp.Variable((2*p, 2*p), diag=True)
        Qn = cp.Variable((2*p, 2*p), diag=True)
        Zp = cp.Variable((2*p, 2*p), diag=True)
        Zn = cp.Variable((2*p, 2*p), diag=True)
        Psi = cp.Variable((2*p, 2*p), diag=True)
        Psi_p = cp.Variable((2*p, 2*p), diag=True)
        Psi_n = cp.Variable((2*p, 2*p), diag=True)
        Gamma = cp.Variable((2*p, 2*p), diag=True)

        Omega = Q + cp.minimum(Qp, Qn) + 2*cp.minimum(Psi_p, Psi_n)

        # Constraints
        if synthesize_control:
            # In fact P is P^{-1}
            # In fact Zp is Zp^{-1}
            # In fact Zn is Zn^{-1}
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
            constraints = [
                P >= epsilon,
                Zp >= epsilon,
                Zn >= epsilon,
                Gamma >= epsilon,
                Omega >= epsilon,
                Pi << 0
            ]
        else:
            Ups_11 = cA0.T*P + P*cA0 + Q
            Ups_12 = cA0.T*Zp + P*cA1 + Psi_p
            Ups_13 = P*cA2 - cA0.T*Zn - Psi_n
            Ups_22 = Zp*cA1 + cA1.T*Zp + Qp
            Ups_23 = Zp*cA2 - cA1.T*Zn + Psi
            Ups_33 = Qn - Zn * cA2 - cA2.T*Zn
            Ups = cp.bmat([  # Block matrix
                [Ups_11,   Ups_12,    Ups_13,  P],
                [Ups_12.T, Ups_22,    Ups_23,  Zp],
                [Ups_13.T, Ups_23.T,  Ups_33, -Zn],
                [P,       Zp,      -Zn,    -Gamma]
            ])
            U0, U1, U2 = None, None, None
            constraints = [
                P >= epsilon,
                P + cp.minimum(Zp, Zn) >= epsilon,
                Ups << 0,
                Gamma >= epsilon,
                Omega >= epsilon,
            ]

        prob = cp.Problem(cp.Minimize(0), constraints=constraints)
        prob.solve(solver=cp.SCS, verbose=True)

        logger.debug("Status: {}".format(prob.status))
        success = prob.status == "optimal"
        if success:
            logger.debug("- P + min(Zp, Zn): {}".format(np.diagonal(P.value.todense() +
                                                        np.minimum(Zp.value.todense(), Zn.value.todense()))))
            logger.debug("- Gamma: {}".format(Gamma.value))
            logger.debug("- Omega: {}".format(Omega.value))
            P = P.value.todense()
            Zp = Zp.value.todense()
            Zn = Zn.value.todense()

            if synthesize_control:
                P = np.linalg.inv(P)
                Zp = np.linalg.inv(Zp)
                Zn = np.linalg.inv(Zn)
                logger.debug("- U0:".format(U0.value))
                logger.debug("- U1:".format(U1.value))
                logger.debug("- U2:".format(U2.value))
                self.K0 = U0.value @ P
                self.K1 = U1.value @ Zp
                self.K2 = U2.value @ Zn

            self.compute_attraction_basin(cB, Gamma.value.todense(), Omega.value, P, Zp, Zn)
        return success

    def compute_attraction_basin(self, cB, Gamma, Omega, P, Zp, Zn):
        r"""
            Compute the attraction basin X_f that asymptotically contains \xi = [\underline{x}, \overline{x}]
        :param cB: Extended control matrix
        :param Gamma: LMI matrix
        :param Omega: LMI matrix
        :param P: LMI matrix
        :param Zp: LMI matrix
        :param Zn: LMI matrix
        :return: An interval asymptotically containing \xi, under the closed-loop dynamics tested in the stability_lmi.
        """
        Id = np.eye(Gamma.shape[0])
        delta_tilde = (cB @ self.S + Id) @ self.delta()
        alpha = np.amin(np.linalg.eigvals(Omega @ np.linalg.inv(P + pos(Zp) + pos(Zn))))
        V_max = 1/alpha * np.abs(delta_tilde.T @ Gamma @ delta_tilde)
        self.Xf = 1 / np.sqrt(np.diagonal(P / V_max))

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

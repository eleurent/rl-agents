from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import os
import numpy as np
import torch

from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from rl_agents.agents.budgeted_ftq.convex_hull_graham import convex_hull_graham

OptimalPolicy = namedtuple('OptimalPolicy',
                           ('id_action_inf', 'id_action_sup', 'proba_inf', 'proba_sup', 'budget_inf', 'budget_sup'))

TransitionBFTQ = namedtuple('TransitionBFTQ',
                            ('state', 'action', 'reward', "next_state", 'constraint', 'beta', "hull_id"))


def compute_interest_points_NN_Qsb(Qsb, action_mask, betas, disp=False, path="tmp", id="default",
                                   hull_options=None, clamp_Qc=None):
    with torch.no_grad():

        if clamp_Qc is not None:
            Qsb[:, len(action_mask):] = np.clip(Qsb[:, len(action_mask):],
                                                    a_min=clamp_Qc[0],
                                                    a_max=clamp_Qc[1])

        if not type(action_mask) == type(np.zeros(1)):
            action_mask = np.asarray(action_mask)
        N_OK_actions = int(len(action_mask) - np.sum(action_mask))

        dtype = [('Qc', 'f4'), ('Qr', 'f4'), ('beta', 'f4'), ('action', 'i4')]

        if path:
            path = Path(path) / "interest_points"
        colinearity = False
        if disp:
            if not os.path.exists(path):
                os.makedirs(path)

        all_points = np.zeros((N_OK_actions * len(betas), 2))
        all_betas = np.zeros((N_OK_actions * len(betas),))
        all_Qs = np.zeros((N_OK_actions * len(betas),), dtype=int)
        max_Qr = -np.inf
        Qc_for_max_Qr = None
        l = 0
        x = np.zeros((N_OK_actions, len(betas)))
        y = np.zeros((N_OK_actions, len(betas)))
        i_beta = 0
        for ibeta, beta in enumerate(betas):
            QQ = Qsb[ibeta]
            for i_a, mask in enumerate(action_mask):
                i_a_ok_act = 0
                if mask == 1:
                    pass
                else:
                    Qr = QQ[i_a]
                    Qc = QQ[i_a + len(action_mask)]
                    x[i_a_ok_act][i_beta] = Qc
                    y[i_a_ok_act][i_beta] = Qr
                    if Qr > max_Qr:
                        max_Qr = Qr
                        Qc_for_max_Qr = Qc
                    all_points[l] = (Qc, Qr)
                    all_Qs[l] = i_a
                    all_betas[l] = beta
                    l += 1
                    i_a_ok_act += 1

            i_beta += 1

        if disp:
            for i_a in range(0, N_OK_actions):  # len(Q_as)):
                if action_mask[i_a] == 1:
                    pass
                else:
                    plt.plot(x[i_a], y[i_a], linewidth=6, alpha=0.2)
        k = 0
        points = []
        betas = []
        Qs = []
        for point in all_points:
            Qc, Qr = point
            if not (Qr < max_Qr and Qc >= Qc_for_max_Qr):
                # on ajoute que les points non dominés
                points.append(point)
                Qs.append(all_Qs[k])
                betas.append(all_betas[k])
            k += 1
        if hull_options is not None and hull_options["decimals"] is not None:
            points = np.round(np.array(points), decimals=hull_options["decimals"])
        else:
            points = np.array(points)

        betas = np.array(betas)
        Qs = np.array(Qs)

        # on remove les duplications
        if hull_options is not None and hull_options["remove_duplicated_points"]:
            points, indices = np.unique(points, axis=0, return_index=True)
            betas = betas[indices]
            Qs = Qs[indices]

        if disp:
            plt.rcParams["figure.figsize"] = (5, 5)
            plt.plot(all_points[:, 0], all_points[:, 1], 'o', markersize=7, color="blue", alpha=0.1)
            plt.plot(points[:, 0], points[:, 1], 'o', markersize=3, color="red")
            plt.grid()

        true_colinearity = False
        expection = False

        if len(points) < 3:
            colinearity = True
            true_colinearity = True
        else:
            if hull_options is None or "library" not in hull_options or hull_options["library"] == "scipy":
                try:
                    if hull_options is not None and hull_options["qhull_options"] is not None:
                        hull = ConvexHull(points, qhull_options=hull_options["qhull_options"])
                    else:
                        hull = ConvexHull(points)
                    vertices = hull.vertices
                except QhullError:
                    colinearity = True
                    expection = True
            elif hull_options is not None and hull_options["library"] == "pure_python":
                if not hull_options["remove_duplicated_points"]:
                    raise Exception("pure_python convexe_hull can't work without removing duplicate points")
                hull = convex_hull_graham(points.tolist())
                vertices = []
                for vertex in hull:
                    vertices.append(np.where(np.all(points == vertex, axis=1)))
                vertices = np.asarray(vertices).squeeze()
            else:
                raise Exception("Wrong hull options : {}".format(hull_options))
        if colinearity:
            idxs_interest_points = range(0, len(points))  # tous les points d'intéret sont bon a prendre
        else:
            stop = False
            max_Qr = -np.inf
            corr_Qc = None
            max_Qr_index = None
            for k in range(0, len(vertices)):
                idx_vertex = vertices[k]
                Qc, Qr = points[idx_vertex]
                if Qr > max_Qr:
                    max_Qr = Qr
                    max_Qr_index = k
                    corr_Qc = Qc
                else:
                    if Qr == max_Qr and Qc < corr_Qc:
                        max_Qr = Qr
                        max_Qr_index = k
                        corr_Qc = Qc
                        k += 1

                    stop = True
                stop = stop or k >= len(vertices)
            idxs_interest_points = []
            stop = False
            # on va itera a l'envers jusq'a ce que Qc change de sens
            Qc_previous = np.inf
            k = max_Qr_index
            j = 0
            # on peut procéder comme ça car c'est counterclockwise
            while not stop:
                idx_vertex = vertices[k]
                Qc, Qr = points[idx_vertex]
                if Qc_previous >= Qc:
                    idxs_interest_points.append(idx_vertex)
                    Qc_previous = Qc
                else:
                    stop = True
                j += 1
                k = (k + 1) % len(vertices)  # counterclockwise
                if j >= len(vertices):
                    stop = True

        if disp:
            plt.title("interest_points_colinear={}".format(colinearity))
            plt.plot(points[idxs_interest_points, 0], points[idxs_interest_points, 1], 'r--', lw=1, color="red")
            plt.plot(points[idxs_interest_points][:, 0], points[idxs_interest_points][:, 1], 'x', markersize=15,
                     color="tab:pink")
            plt.grid()
            plt.savefig(path / "{}.png".format(id), dpi=300, bbox_inches="tight")
            plt.close()

        rez = np.zeros(len(idxs_interest_points), dtype=dtype)
        k = 0
        for idx in idxs_interest_points:
            Qc, Qr = points[idx]
            beta = betas[idx]
            action = Qs[idx]
            rez[k] = np.array([(Qc, Qr, beta, action)], dtype=dtype)
            k += 1
        if colinearity:
            rez = np.sort(rez, order="Qc")
        else:
            rez = np.flip(rez, 0)  # normalement si ya pas colinearité c'est deja trié dans l'ordre decroissant
        return rez, colinearity, true_colinearity, expection  # betas, points, idxs_interest_points, Qs, colinearity


def compute_interest_points_NN(s, Q, action_mask, betas, device, hull_options, clamp_Qc,
                               disp=False, path=None, id="default"):
    with torch.no_grad():
        ss = s.repeat((len(betas), 1, 1))
        bb = torch.from_numpy(betas).float().unsqueeze(1).unsqueeze(1).to(device=device)
        sb = torch.cat((ss, bb), dim=2)
        Qsb = Q(sb).detach().cpu().numpy()
    return compute_interest_points_NN_Qsb(Qsb, action_mask, betas, disp=disp, path=path, id=id,
                                              hull_options=hull_options, clamp_Qc=clamp_Qc)


def convex_hull(s, action_mask, Q, disp, betas, device, hull_options, clamp_Qc, path=None, id="default"):
    if not type(action_mask) == type(np.zeros(1)):
        action_mask = np.asarray(action_mask)
    hull, colinearity, true_colinearity, expection = compute_interest_points_NN(
        s=s,
        Q=Q,
        action_mask=action_mask,
        betas=betas,
        device=device,
        disp=disp,
        path=path,
        id=id,
        hull_options=hull_options,
        clamp_Qc=clamp_Qc)
    return hull


def optimal_pia_pib(beta, hull, statistic):
    with torch.no_grad():
        if len(hull) == 0:
            raise Exception("Hull is empty")
        elif len(hull) == 1:
            Qc_inf, Qr_inf, beta_inf, action_inf = hull[0]
            res = OptimalPolicy(action_inf, 0, 1., 0., beta_inf, 0.)
            if beta == Qc_inf:
                status = "exact"
            elif beta > Qc_inf:
                status = "too_much_budget"
            else:
                status = "not_solvable"
        else:
            Qc_inf, Qr_inf, beta_inf, action_inf = hull[0]
            if beta < Qc_inf:
                status = "not_solvable"
                res = OptimalPolicy(action_inf, 0, 1., 0., beta_inf, 0.)
            else:
                founded = False
                for k in range(1, len(hull)):
                    Qc_sup, Qr_sup, beta_sup, action_sup = hull[k]
                    if Qc_inf == beta:
                        founded = True
                        status = "exact"  # en realité avec Qc_inf <= beta and beta < Qc_sup ca devrait marcher aussi
                        res = OptimalPolicy(action_inf, 0, 1., 0., beta_inf, 0.)
                        break
                    elif Qc_inf < beta and beta < Qc_sup:
                        founded = True
                        p = (beta - Qc_inf) / (Qc_sup - Qc_inf)
                        status = "regular"
                        res = OptimalPolicy(action_inf, action_sup, 1. - p, p, beta_inf, beta_sup)
                        break
                    else:
                        Qc_inf, Qr_inf, beta_inf, action_inf = Qc_sup, Qr_sup, beta_sup, action_sup
                if not founded:  # we have at least Qc_sup budget
                    status = "too_much_budget"
                    res = OptimalPolicy(action_inf, 0, 1., 0., beta_inf,
                                        0.)  # action_inf = action_sup, beta_inf=beta_sup
        return res, status

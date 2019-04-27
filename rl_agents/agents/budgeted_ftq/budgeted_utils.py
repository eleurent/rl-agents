from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import os
import numpy as np
import torch

from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from rl_agents.agents.budgeted_ftq.convex_hull_graham import convex_hull_graham

HullPoint = namedtuple('HullPoint', ('action', 'budget', 'qr', 'qc'))
Mixture = namedtuple('Mixture', ('inf', 'sup', "probability_sup"))

TransitionBFTQ = namedtuple('TransitionBFTQ',
                            ('state', 'action', 'reward', 'next_state', 'terminal', 'constraint', 'beta'))


def compute_convex_hull_from_values(qsb, betas, hull_options=None, clamp_Qc=None,
                                    disp=False, path="tmp", id="default"):
    with torch.no_grad():

        n_actions = qsb.shape[2] // 2
        if clamp_Qc is not None:
            qsb[:, n_actions:] = np.clip(qsb[:, n_actions:], a_min=clamp_Qc[0], a_max=clamp_Qc[1])

        dtype = [('action', 'i4'), ('beta', 'f4'), ('Qr', 'f4'), ('Qc', 'f4')]

        if path:
            path = Path(path) / "interest_points"
        colinearity = False
        if disp:
            if not os.path.exists(path):
                os.makedirs(path)

        all_points = np.zeros((n_actions * len(betas), 2))
        all_betas = np.zeros((n_actions * len(betas),))
        all_Qs = np.zeros((n_actions * len(betas),), dtype=int)
        max_Qr = -np.inf
        Qc_for_max_Qr = None
        l = 0
        x = np.zeros((n_actions, len(betas)))
        y = np.zeros((n_actions, len(betas)))
        i_beta = 0
        for ibeta, beta in enumerate(betas):
            QQ = qsb[ibeta]
            for i_a in range(n_actions):
                Qr = QQ[i_a]
                Qc = QQ[i_a + n_actions]
                x[i_a][i_beta] = Qc
                y[i_a][i_beta] = Qr
                if Qr > max_Qr:
                    max_Qr = Qr
                    Qc_for_max_Qr = Qc
                all_points[l] = (Qc, Qr)
                all_Qs[l] = i_a
                all_betas[l] = beta
                l += 1

            i_beta += 1

        if disp:
            for i_a in range(n_actions):
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
        exception = False

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
                    exception = True
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

        hull = np.zeros(len(idxs_interest_points), dtype=dtype)
        k = 0
        for idx in idxs_interest_points:
            Qc, Qr = points[idx]
            beta = betas[idx]
            action = Qs[idx]
            hull[k] = np.array([(action, beta, Qr, Qc)], dtype=dtype)
            k += 1
        if colinearity:
            hull = np.sort(hull, order="Qc")
        else:
            hull = np.flip(hull, 0)  # already sorted in decreasing order
        return hull.to_list(), colinearity, true_colinearity, exception


def compute_interest_points_NN(s, Q, betas, device, hull_options, clamp_Qc,
                               disp=False, path=None, id="default"):
    with torch.no_grad():
        ss = s.repeat((len(betas), 1, 1))
        bb = torch.from_numpy(betas).float().unsqueeze(1).unsqueeze(1).to(device=device)
        sb = torch.cat((ss, bb), dim=2)
        Qsb = Q(sb).detach().cpu().numpy()
    return compute_convex_hull_from_values(Qsb, betas, disp=disp, path=path, id=id,
                                           hull_options=hull_options, clamp_Qc=clamp_Qc)


def optimal_mixture(hull, beta):
    """
        Given a hull H and a cost budget beta, find the mixture.

        1. Solve: k = min{k: beta > qc with (qc, qr) = H[k]}
        2. Pick points: inf, sup = H[k−1], H[k]
        3. Mix with probability = (beta−qc_inf)/(qc_sup − qc_inf)

    :param hull: a hull of values qr,qc for different action/budgets
    :param beta: a maximumal cost budget
    :return: the mixture policy with maximal qr and expected cost under beta
    """
    with torch.no_grad():
        if not hull:
            raise Exception("Hull is empty")
        elif len(hull) == 1:
            inf = HullPoint(*hull[0])
            mixture = Mixture(inf, inf, 0)
            if beta == inf.qc:
                status = "exact"
            elif beta > inf.qc:
                status = "too_much_budget"
            else:
                status = "not_solvable"
        else:
            inf = HullPoint(*hull[0])
            if beta < inf.qc:
                status = "not_solvable"
                mixture = Mixture(inf, inf, 0)
            else:
                for point in hull[1:]:
                    sup = HullPoint(point)
                    if inf.qc == beta:
                        status = "exact"
                        mixture = Mixture(inf, inf, 0)
                        break
                    elif inf.qc < beta < sup.qc:
                        status = "regular"
                        mixture = Mixture(inf, sup, (beta - inf.qc) / (sup.qc - inf.qc))
                        break
                    else:
                        inf = sup
                else:
                    status = "too_much_budget"
                    mixture = Mixture(sup, sup, 1)
    return mixture, status

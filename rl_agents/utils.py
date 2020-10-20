import itertools
from functools import partial
from typing import Callable

import numpy as np
from numba import jit


def constrain(x, a, b):
    return np.minimum(np.maximum(x, a), b)


def not_zero(x, eps=0.01):
    if abs(x) > eps:
        return x
    elif x > 0:
        return eps
    else:
        return -eps


def wrap_to_pi(x):
    return ((x+np.pi) % (2*np.pi)) - np.pi


def remap(v, x, y, clip=False):
    if x[1] == x[0]:
        return y[0]
    out = y[0] + (v-x[0])*(y[1]-y[0])/(x[1]-x[0])
    if clip:
        out = constrain(out, y[0], y[1])
    return out


def near_split(x, num_bins=None, size_bins=None):
    """
        Split a number into several bins with near-even distribution.

        You can either set the number of bins, or their size.
        The sum of bins always equals the total.
    :param x: number to split
    :param num_bins: number of bins
    :param size_bins: size of bins
    :return: list of bin sizes
    """
    if num_bins:
        quotient, remainder = divmod(x, num_bins)
        return [quotient + 1] * remainder + [quotient] * (num_bins - remainder)
    elif size_bins:
        return near_split(x, num_bins=int(np.ceil(x / size_bins)))


def zip_with_singletons(*args):
    """
        Zip lists and singletons by repeating singletons

        Behaves usually for lists and repeat other arguments (including other iterables such as tuples np.array!)
    :param args: arguments to zip x1, x2, .. xn
    :return: zipped tuples (x11, x21, ..., xn1), ... (x1m, x2m, ..., xnm)
    """
    return zip(*(arg if isinstance(arg, list) else itertools.repeat(arg) for arg in args))


def kullback_leibler(p: np.ndarray, q: np.ndarray) -> float:
    """
        KL between two categorical distributions
    :param p: categorical distribution
    :param q: categorical distribution
    :return: KL(p||q)
    """
    kl = 0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi > 0:
                kl += pi * np.log(pi/qi)
            else:
                kl = np.inf
    return kl


def bernoulli_kullback_leibler(p: float, q: float) -> float:
    """
        Compute the Kullback-Leibler divergence of two Bernoulli distributions.

    :param p: parameter of the first Bernoulli distribution
    :param q: parameter of the second Bernoulli distribution
    :return: KL(B(p) || B(q))
    """
    kl1, kl2 = 0, np.infty
    if p > 0:
        if q > 0:
            kl1 = p*np.log(p/q)

    if q < 1:
        if p < 1:
            kl2 = (1 - p) * np.log((1 - p) / (1 - q))
        else:
            kl2 = 0
    return kl1 + kl2


def d_bernoulli_kullback_leibler_dq(p: float, q: float) -> float:
    """
        Compute the partial derivative of the Kullback-Leibler divergence of two Bernoulli distributions.

        With respect to the parameter q of the second distribution.

    :param p: parameter of the first Bernoulli distribution
    :param q: parameter of the second Bernoulli distribution
    :return: dKL/dq(B(p) || B(q))
    """
    return (1 - p) / (1 - q) - p/q


def kl_upper_bound(_sum: float, count: int, threshold: float = 1, eps: float = 1e-2, lower: bool = False) -> float:
    """
        Upper Confidence Bound of the empirical mean built on the Kullback-Leibler divergence.

        The computation involves solving a small convex optimization problem using Newton Iteration

    :param _sum: Sum of sample values
    :param count: Number of samples
    :param time: Allows to set the bound confidence level
    :param threshold: the maximum kl-divergence * count
    :param eps: Absolute accuracy of the Netwon Iteration
    :param lower: Whether to compute a lower-bound instead of upper-bound
    """
    if count == 0:
        return 0 if lower else 1

    mu = _sum/count
    max_div = threshold/count

    # Solve KL(mu, q) = max_div
    kl = lambda q: bernoulli_kullback_leibler(mu, q) - max_div
    d_kl = lambda q: d_bernoulli_kullback_leibler_dq(mu, q)
    a, b = (0, mu) if lower else (mu, 1)

    return newton_iteration(kl, d_kl, eps, a=a, b=b)


def newton_iteration(f: Callable, df: Callable, eps: float, x0: float = None, a: float = None, b: float = None,
                     weight: float = 0.9, display: bool = False, max_iterations: int = 100) -> float:
    """
        Run Newton Iteration to solve f(x) = 0, with x in [a, b]
    :param f: a function R -> R
    :param df: the function derivative
    :param eps: the desired accuracy
    :param x0: an initial value
    :param a: an optional lower-bound
    :param b: an optional upper-bound
    :param weight: a weight to handle out of bounds events
    :param display: plot the function
    :return: x such that f(x) = 0
    """
    x = np.inf
    if x0 is None:
        x0 = (a + b) / 2
    if a is not None and b is not None and a == b:
        return a
    x_next = x0
    iterations = 0
    while abs(x - x_next) > eps and iterations < max_iterations:
        iterations += 1
        x = x_next

        if display:
            import matplotlib.pyplot as plt
            xx0 = a or x-1
            xx1 = b or x+1
            xx = np.linspace(xx0, xx1, 100)
            yy = np.array(list(map(f, xx)))
            plt.plot(xx, yy)
            plt.axvline(x=x)
            plt.show()

        f_x = f(x)
        try:
            df_x = df(x)
        except ZeroDivisionError:
            df_x = (f_x - f(x-eps))/eps
        if df_x != 0:
            x_next = x - f_x / df_x

        if a is not None and x_next < a:
            x_next = weight * a + (1 - weight) * x
        elif b is not None and x_next > b:
            x_next = weight * b + (1 - weight) * x

    if a is not None and x_next < a:
        x_next = a
    if b is not None and x_next > b:
        x_next = b

    return x_next


def binary_search(f: Callable, eps: float, a: float, b: float = None,
                  display: bool = False, max_iterations: int = 100) -> float:
    """
    Binary search the zero of a non-increasing function.
    :param f: the function
    :param eps: accuracy
    :param a: lower bound for the zero
    :param b: optional upper bound for the zero
    :param display: display the function
    :return: x such that |f(x)| < eps
    """
    x = np.nan
    find_b = False
    if b is None:
        find_b = True
        b = a + 1
    for _ in range(max_iterations):
        x = (a + b) / 2
        f_x = f(x)

        if display:
            import matplotlib.pyplot as plt
            xx0 = a
            xx1 = b
            xx = np.linspace(xx0, xx1, 100)
            yy = np.array(list(map(f, xx)))
            plt.plot(xx, yy)
            plt.axvline(x=x)
            plt.show()

        if f_x > 0:
            a = x
            if find_b:
                b = 2*max(b, 1)
        else:
            b = x
            find_b = False

        if abs(f_x) <= eps:
            break
    else:
        # print("Error: Reached maximum iteration", b)
        pass
    return x


@jit(nopython=True)
def binary_search_theta(q_p, f_p, c, eps: float, a: float, b: float = None, max_iterations: int = 100):
    x = np.nan
    find_b = False
    if b is None:
        find_b = True
        b = a + 1
    for _ in range(max_iterations):
        x = (a + b) / 2
        l_m_f_p = x - f_p
        f_x = q_p @ np.log(l_m_f_p) + np.log(q_p @ (1 / l_m_f_p)) - c
        if f_x > 0:
            a = x
            if find_b:
                b = 2*max(b, 1)
        else:
            b = x
            find_b = False

        if abs(f_x) <= eps:
            break
    else:
        # print("Error: Reached maximum iteration")
        pass
    return x


@jit(nopython=True)
def theta_func(l, q_p, f_p, c):
    l_m_f_p = l - f_p
    return q_p @ np.log(l_m_f_p) + np.log(q_p @ (1 / l_m_f_p)) - c


@jit(nopython=True)
def d_theta_dl_func(l, q_p, f_p):
    l_m_f_p_inv = 1 / (l - f_p)
    q_l_m_f_p_inv = q_p @ l_m_f_p_inv
    return q_l_m_f_p_inv - (q_p @ (l_m_f_p_inv ** 2)) / q_l_m_f_p_inv


def max_expectation_under_constraint(f: np.ndarray, q: np.ndarray, c: float, eps: float = 1e-2,
                                     display: bool = False) -> np.ndarray:
    """
        Solve the following constrained optimisation problem:
             max_p E_p[f]    s.t.    KL(q || p) <= c
    :param f: an array of values f(x), np.array of size n
    :param q: a discrete distribution q(x), np.array of size n
    :param c: a threshold for the KL divergence between p and q.
    :param eps: desired accuracy on the constraint
    :param display: plot the function
    :return: the argmax p*
    """
    np.seterr(all="warn")
    if np.all(q == 0):
        q = np.ones(q.size) / q.size
    x_plus = np.where(q > 0)
    x_zero = np.where(q == 0)
    p_star = np.zeros(q.shape)
    lambda_, z = None, 0

    q_p = q[x_plus]
    f_p = f[x_plus]
    f_star = np.amax(f)
    theta = partial(theta_func, q_p=q_p, f_p=f_p, c=c)
    d_theta_dl = partial(d_theta_dl_func, q_p=q_p, f_p=f_p)
    if f_star > np.amax(f_p):
        theta_star = theta_func(f_star, q_p=q_p, f_p=f_p, c=c)
        if theta_star < 0:
            lambda_ = f_star
            z = 1 - np.exp(theta_star)
            p_star[x_zero] = 1.0 * (f[x_zero] == np.amax(f[x_zero]))
            p_star[x_zero] *= z / p_star[x_zero].sum()
    if lambda_ is None:
        if np.isclose(f_p, f_p[0]).all():
            return q
        else:
            # Binary search seems slightly (10%) faster than newton
            # lambda_ = binary_search(theta, eps, a=f_star, display=display)
            lambda_ = newton_iteration(theta, d_theta_dl, eps, x0=f_star + 1, a=f_star, display=display)

            # numba jit binary search is twice as fast as python version
            # lambda_ = binary_search_theta(q_p=q_p, f_p=f_p, c=c, eps=eps, a=f_star)

    beta = (1 - z) / (q_p @ (1 / (lambda_ - f_p)))
    if beta == 0:
        x_uni = np.where((q > 0) & (f == f_star))
        if np.size(x_uni) > 0:
            p_star[x_uni] = (1 - z) / np.size(x_uni)
    else:
        p_star[x_plus] = beta * q_p / (lambda_ - f_p)
    return p_star


def all_argmax(x: np.ndarray) -> np.ndarray:
    """
    :param x: a set
    :return: the list of indexes of all maximums of x
    """
    m = np.amax(x)
    return np.nonzero(np.isclose(x, m))[0]


def random_argmax(x: np.ndarray) -> int:
    """
        Randomly tie-breaking arg max
    :param x: an array
    :return: a random index among the maximums
    """
    indices = all_argmax(x)
    return np.random.choice(indices)


def random_dist(n):
    q = np.random.random(n)
    return q / q.sum()


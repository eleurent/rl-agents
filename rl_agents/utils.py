import itertools

import numpy as np


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


def kullback_leibler(p, q):
    kl = 0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi > 0:
                kl += pi * np.log(pi/qi)
            else:
                kl = np.inf
    return kl


def bernoulli_kullback_leibler(p, q):
    """
        Compute the Kullback-Leibler divergence of two Bernoulli distributions.

    :param p: parameter of the first Bernoulli distribution
    :param q: parameter of the second Bernoulli distribution
    :return: KL(B(p), B(q))
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


def d_bernoulli_kullback_leibler_dq(p, q):
    """
        Compute the partial derivative of the Kullback-Leibler divergence of two Bernoulli distributions.

        With respect to the parameter q of the second distribution.

    :param p: parameter of the first Bernoulli distribution
    :param q: parameter of the second Bernoulli distribution
    :return: dKL/dq(B(p), B(q))
    """
    return (1 - p) / (1 - q) - p/q


def hoeffding_upper_bound(_sum, count, time, c=4):
    """
        Upper Confidence Bound of the empirical mean built on the Chernoff-Hoeffding inequality.

    :param _sum: Sum of sample values
    :param count: Number of samples
    :param time: Allows to set the bound confidence level to time^(-c)
    :param c: Time exponent in the confidence level
    """
    return _sum / count + np.sqrt(c * np.log(time) / (2 * count))


def laplace_upper_bound(_sum, count, time, c=2):
    """
        Upper Confidence Bound of the empirical mean built on the Laplace time-uniform concentration inequality.

    :param _sum: Sum of sample values
    :param count: Number of samples
    :param time: Allows to set the bound confidence level to time^(-c)
    :param c: Time exponent in the confidence level
    """
    return _sum / count + np.sqrt((1 + 1 / count) * c * np.log(np.sqrt(count + 1) * time) / (2 * count))


def kl_upper_bound(_sum, count, time, threshold="2*np.log(time)", eps=1e-2, lower=False):
    """
        Upper Confidence Bound of the empirical mean built on the Kullback-Leibler divergence.

        The computation involves solving a small convex optimization problem using Newton Iteration

    :param _sum: Sum of sample values
    :param count: Number of samples
    :param time: Allows to set the bound confidence level
    :param threshold: expression to define the maximum kl-divergence
    :param eps: Absolute accuracy of the Netwon Iteration
    :param lower: Whether to compute a lower-bound instead of upper-bound
    """
    mu = _sum/count
    max_div = eval(threshold)/count

    # Solve KL(mu, q) = max_div
    kl = lambda q: bernoulli_kullback_leibler(mu, q) - max_div
    d_kl = lambda q: d_bernoulli_kullback_leibler_dq(mu, q)
    a, b = (0, mu) if lower else (mu, 1)
    return newton_iteration(kl, d_kl, eps, a=a, b=b)


def newton_iteration(f, df, eps, x0=None, a=None, b=None, weight=0.9, display=False):
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
    while abs(x - x_next) > eps:
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

        f_x, df_x = f(x), df(x)
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


def max_expectation_under_constraint(f, q, c, eps=1e-2, display=False):
    """
        Solve the following constrained optimisation problem:
             max_p E_p[f]    s.t.    KL(q || p) <= c
    :param f: an array of values f(x), np.array of size n
    :param q: a discrete distribution q(x), np.array of size n
    :param c: a threshold for the KL divergence between p and q.
    :param eps: desired accuracy on the constraint
    :return: the argmax p*
    """
    np.seterr(all='warn')
    x_plus = np.where(q > 0)
    x_zero = np.where(q == 0)
    p_star = np.zeros(q.shape)
    lambda_, z = None, 0

    q_p = q[x_plus]
    f_p = f[x_plus]
    f_star = np.amax(f)
    theta = lambda l: q_p @ np.log(l - f_p) + np.log(q_p @ (1 / (l - f_p))) - c
    d_theta_dl = lambda l: q_p @ (1 / (l - f_p)) - (q_p @ (1 / (l - f_p)**2)) / (q_p @ (1 / (l - f_p)))
    if f_star > np.amax(f_p):
        theta_star = theta(f_star)
        if theta_star < 0:
            lambda_ = f_star
            z = 1 - np.exp(theta_star)
            p_star[x_zero] = 1.0 * (f[x_zero] == np.amax(f[x_zero]))
            p_star[x_zero] *= z / p_star[x_zero].sum()
    if lambda_ is None:
        if np.allclose(f_p, f_p[0]):
            return q
        else:
            lambda_ = newton_iteration(theta, d_theta_dl, eps, x0=f_star + 1, a=f_star, display=display)

    beta = (1 - z) / (q_p @ (1 / (lambda_ - f_p)))
    if beta == 0:
        x_uni = np.where((q > 0) & (f == f_star))
        p_star[x_uni] = (1 - z) / np.size(x_uni)
    else:
        p_star[x_plus] = beta * q_p / (lambda_ - f_p)
    return p_star

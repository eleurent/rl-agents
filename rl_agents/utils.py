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


def kl_upper_bound(_sum, count, time, c=2, eps=1e-2):
    """
        Upper Confidence Bound of the empirical mean built on the Kullback-Leibler divergence.

        The computation involves solving a small convex optimization problem using Newton Iteration

    :param _sum: Sum of sample values
    :param count: Number of samples
    :param time: Allows to set the bound confidence level
    :param c: Coefficient before the log(t) in the maximum divergence
    :param eps: Absolute accuracy of the Netwon Iteration
    """
    mu = _sum/count
    max_div = c*np.log(time)/count

    # Solve KL(mu, q) = max_div
    q = mu
    next_q = (1 + mu)/2
    while abs(q - next_q) > eps:
        q = next_q

        # Newton Iteration
        klq = bernoulli_kullback_leibler(mu, q) - max_div
        d_klq = d_bernoulli_kullback_leibler_dq(mu, q)
        next_q = q - klq / d_klq

        # Out of bounds: move toward the bound
        weight = 0.9
        if next_q > 1:
            next_q = weight*1 + (1 - weight)*q
        elif next_q < mu:
            next_q = weight*mu + (1 - weight)*q

    return constrain(q, 0, 1)

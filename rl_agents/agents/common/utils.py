import numpy as np


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


def sample_simplex(coeff, bias, min_x, max_x, np_random=np.random):
    """
    Sample from a simplex.

    The simplex is defined by:
        w.x + b <= 0
        x_min <= x <= x_max

    Warning: this is not uniform sampling.

    :param coeff: coefficient w
    :param bias: bias b
    :param min_x: lower bound on x
    :param max_x: upper bound on x
    :param np_random: source of randomness
    :return: a sample from the simplex
    """
    x = np.zeros(len(coeff))
    indexes = np.asarray(range(0, len(coeff)))
    np_random.shuffle(indexes)
    remain_indexes = np.copy(indexes)
    for i_index, index in enumerate(indexes):
        remain_indexes = remain_indexes[1:]
        current_coeff = np.take(coeff, remain_indexes)
        full_min = np.full(len(remain_indexes), min_x)
        full_max = np.full(len(remain_indexes), max_x)
        dot_max = np.dot(current_coeff, full_max)
        dot_min = np.dot(current_coeff, full_min)
        min_xi = (bias - dot_max) / coeff[index]
        max_xi = (bias - dot_min) / coeff[index]
        min_xi = np.max([min_xi, min_x])
        max_xi = np.min([max_xi, max_x])
        xi = min_xi + np_random.random_sample() * (max_xi - min_xi)
        bias = bias - xi * coeff[index]
        x[index] = xi
        if len(remain_indexes) == 1:
            break
    last_index = remain_indexes[0]
    x[last_index] = bias / coeff[last_index]
    return x

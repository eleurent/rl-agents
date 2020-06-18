import logging
import os
import re
import shutil

import numpy as np
from subprocess import PIPE, run, check_output
import torch


logger = logging.getLogger(__name__)


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


def load_pytorch():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    logger.info("Using torch.multiprocessing.set_start_method('spawn')")
    import torch.multiprocessing as multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError as e:
        logger.warning(str(e))


def get_gpu_memory_map():
    result = check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
    return [int(x) for x in result.split()]


def least_used_device():
    """ Get the  GPU device with most available memory. """
    if not torch.cuda.is_available():
        raise RuntimeError("cuda unavailable")

    if shutil.which('nvidia-smi') is None:
        raise RuntimeError("nvidia-smi unavailable: cannot select device with most least memory used.")

    memory_map = get_gpu_memory_map()
    device_id = np.argmin(memory_map)
    logger.info("Choosing GPU device: {}, memory used: {}".format(device_id, memory_map[device_id]))
    return torch.device("cuda:{}".format(device_id))


def choose_device(preferred_device, default_device="cpu"):
    try:
        if preferred_device == "cuda:best":
            preferred_device = least_used_device()
        torch.zeros((1,), device=preferred_device)  # Test availability
        return preferred_device
    except (RuntimeError, AssertionError):
        logger.warning("Preferred device {} unavailable, switching to default {}"
                     .format(preferred_device, default_device))
        return default_device


def get_memory(pid=None):
    if not pid:
        pid = os.getpid()
    command = "nvidia-smi"
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True).stdout
    m = re.findall("\| *[0-9] *" + str(pid) + " *C *.*python.*? +([0-9]+).*\|", result, re.MULTILINE)
    return [int(mem) for mem in m]

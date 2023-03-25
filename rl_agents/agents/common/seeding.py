import logging
import os
import re
import shutil
from typing import Optional, Tuple, Any

import numpy as np
from subprocess import PIPE, run, check_output
import torch

from gymnasium import error
from gym.utils.seeding import RandomNumberGenerator

logger = logging.getLogger(__name__)


def np_random(seed: Optional[int] = None) -> Tuple["RandomNumberGenerator", Any]:
    """Generates a random number generator from the seed and returns the Generator and seed.
    Args:
        seed: The seed used to create the generator
    Returns:
        The generator and resulting seed
    Raises:
        Error: Seed must be a non-negative integer or omitted
    """
    if seed is not None:
        if isinstance(seed, int) and 0 <= seed:
            seed_seq = np.random.SeedSequence(seed)
        elif isinstance(seed, np.ndarray):
            seed_seq = seed
        else:
            raise error.Error(f"Seed must be a non-negative integer or omitted, not {seed}")
    np_seed = seed_seq.entropy
    rng = RandomNumberGenerator(np.random.PCG64(seed_seq))
    return rng, np_seed

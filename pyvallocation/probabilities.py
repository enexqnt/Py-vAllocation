"""
Module for generating and computing scenario probabilities.

Includes uniform, exponential decay, and Gaussian kernel probability generators,
as well as a function to compute the effective number of scenarios.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_uniform_probabilities(num_observations: int) -> np.ndarray:
    if num_observations <= 0:
        logger.error("num_observations must be greater than 0, got %d", num_observations)
        raise ValueError("num_observations must be greater than 0.")
    return np.full(num_observations, 1.0 / num_observations)

def generate_exp_decay_probabilities(num_observations: int, half_life: int) -> np.ndarray:
    if half_life <= 0:
        raise ValueError("half_life must be greater than 0.")
    p = np.exp(-np.log(2) / half_life * (num_observations - np.arange(1, num_observations + 1)))
    return p / np.sum(p)

def silverman_bandwidth(x: np.ndarray) -> float:
    x = np.asarray(x)
    n = len(x)
    sigma = np.std(x, ddof=1)
    return 1.06 * sigma * n**(-1/5)

def generate_gaussian_kernel_probabilities(
    x: np.ndarray,
    v: np.ndarray | None = None,
    h: float | None = None,
    x_T: float | None = None
) -> np.ndarray:
    x = np.asarray(x)
    if v is None:
        v = x.copy()
    else:
        v = np.asarray(v)
    if h is None:
        h = silverman_bandwidth(x)
    if x_T is None:
        x_T = x[-1]
    w = np.exp(-((v - x_T) ** 2) / (2 * h**2))
    p = w / np.sum(w)
    return p

def compute_effective_number_scenarios(probabilities: np.ndarray) -> float:
    return 1 / np.sum(probabilities**2)

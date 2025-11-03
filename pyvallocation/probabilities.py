import logging
from typing import Optional, Sequence, Union
import numpy as np

logger = logging.getLogger(__name__)

ProbabilityLike = Union[np.ndarray, Sequence[float]]


def normalize_probability_vector(
    probabilities: ProbabilityLike,
    *,
    strictly_positive: bool = False,
    name: str = "probabilities",
) -> np.ndarray:
    """
    Validate and normalise a 1-D probability vector.

    Parameters
    ----------
    probabilities :
        Array-like of probabilities. Must be one-dimensional and contain at least
        one entry.
    strictly_positive :
        When ``True`` all entries must be strictly greater than zero. When
        ``False`` non-negative weights are accepted.
    name :
        Identifier used in error messages.

    Returns
    -------
    numpy.ndarray
        Normalised probability vector summing to one.
    """
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError(f"{name} must be a one-dimensional array with at least one entry.")
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"{name} must contain only finite values.")
    if strictly_positive:
        if np.any(probs <= 0.0):
            raise ValueError(f"{name} must be strictly positive.")
    else:
        if np.any(probs < 0.0):
            raise ValueError(f"{name} must be non-negative.")
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must sum to a positive finite value.")
    if not np.isclose(total, 1.0):
        probs = probs / total
    return probs


def resolve_probabilities(
    probabilities: Optional[ProbabilityLike],
    n_observations: int,
    *,
    strictly_positive: bool = False,
    name: str = "probabilities",
) -> np.ndarray:
    """
    Return a normalised probability vector of length ``n_observations``.

    Parameters
    ----------
    probabilities :
        Optional array-like of probabilities. When ``None`` a uniform
        distribution is generated.
    n_observations :
        Expected number of observations (length of scenario set).
    strictly_positive :
        Forwarded to :func:`normalize_probability_vector`.
    name :
        Identifier used in error messages.

    Returns
    -------
    numpy.ndarray
        Probability vector summing to one with shape ``(n_observations,)``.
    """
    if probabilities is None:
        return generate_uniform_probabilities(n_observations)
    probs = normalize_probability_vector(
        probabilities, strictly_positive=strictly_positive, name=name
    )
    if probs.shape[0] != n_observations:
        raise ValueError(f"{name} length must match the number of observations ({n_observations}).")
    return probs


def generate_uniform_probabilities(num_observations: int) -> np.ndarray:
    """Return equal probabilities for ``num_observations`` scenarios."""

    if num_observations <= 0:
        logger.error(
            "num_observations must be greater than 0, got %d", num_observations
        )
        raise ValueError("num_observations must be greater than 0.")
    return np.full(num_observations, 1.0 / num_observations)


def generate_exp_decay_probabilities(
    num_observations: int, half_life: int
) -> np.ndarray:
    """Return exponentially decaying probabilities with the given ``half_life``."""

    if half_life <= 0:
        raise ValueError("half_life must be greater than 0.")
    p = np.exp(
        -np.log(2) / half_life * (num_observations - np.arange(1, num_observations + 1))
    )
    return p / np.sum(p)


def silverman_bandwidth(x: np.ndarray) -> float:
    """Return Silverman's rule-of-thumb bandwidth for ``x``."""

    x = np.asarray(x)
    n = len(x)
    sigma = np.std(x, ddof=1)
    return 1.06 * sigma * n ** (-1 / 5)


def generate_gaussian_kernel_probabilities(
    x: np.ndarray,
    v: Union[np.ndarray, None] = None,
    h: Union[float, None] = None,
    x_T: Union[float, None] = None,
) -> np.ndarray:
    """Generate kernel-based probabilities for ``v`` centred on ``x_T``."""

    x = np.asarray(x)
    if v is None:
        v = x.copy()
    else:
        v = np.asarray(v)
    if h is None:
        h = silverman_bandwidth(x)
    h = float(h)
    if h <= 0:
        raise ValueError("Bandwidth `h` must be strictly positive.")
    if x_T is None:
        x_T = x[-1]
    w = np.exp(-((v - x_T) ** 2) / (2 * h**2))
    weight_sum = np.sum(w)
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("Kernel weights sum to zero; supply a positive bandwidth and non-degenerate inputs.")
    return w / weight_sum


def compute_effective_number_scenarios(probabilities: np.ndarray) -> float:
    """Return the effective number of scenarios given a probability vector."""

    return 1 / np.sum(probabilities**2)

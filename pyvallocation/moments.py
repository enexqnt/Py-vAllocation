
"""Module for estimation and shrinkage of portfolio moments.

Provides functions to estimate weighted sample mean and covariance,
apply Bayes-Stein shrinkage to the mean, and Ledoit-Wolf shrinkage to the covariance matrix.

Functions:
- estimate_sample_moments: Compute weighted mean and covariance matrix.
- shrink_mean_jorion: Apply Bayes-Stein shrinkage to mean vector.
- shrink_covariance_ledoit_wolf: Apply Ledoit-Wolf shrinkage to covariance matrix.

All functions support pandas objects and numpy arrays interchangeably.
"""

from __future__ import annotations
import logging
from typing import Union, Tuple, Sequence, Optional

import numpy as np
from numpy.linalg import inv, LinAlgError

try:
    import pandas as pd
except ImportError:
    pd = None

from pyvallocation.utils.validation import (
    ensure_psd_matrix,
    check_non_negativity,
    check_weights_sum_to_one,
)

ArrayLike = Union[np.ndarray, "pd.Series", "pd.DataFrame"] if pd is not None else np.ndarray

logger = logging.getLogger(__name__)


def _labels(*objs: ArrayLike) -> Optional[Sequence[str]]:
    if pd is None:
        return None
    for obj in objs:
        if isinstance(obj, pd.DataFrame):
            return obj.columns.to_list()
        if isinstance(obj, pd.Series):
            return obj.index.to_list()
    return None


def _wrap(x: np.ndarray, labels: Optional[Sequence[str]], vector: bool) -> ArrayLike:
    if labels is None or pd is None:
        return x
    if vector:
        return pd.Series(x, index=labels, name="mu")
    return pd.DataFrame(x, index=labels, columns=labels)


def estimate_sample_moments(R: ArrayLike, p: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    R_arr, p_arr = np.asarray(R), np.asarray(p)
    T, N = R_arr.shape

    if p_arr.shape[0] != T:
        logger.error(
            "Weight length mismatch: weights length %d, returns length %d", p_arr.shape[0], T
        )
        raise ValueError("Weight length mismatch.")
    if not (check_non_negativity(p_arr) and check_weights_sum_to_one(p_arr)):
        logger.error("Weights must be non-negative and sum to one.")
        raise ValueError("Weights must be non-negative and sum to one.")

    mu = R_arr.T @ p_arr
    X = R_arr - mu
    S = (X.T * p_arr) @ X
    S = (S + S.T) / 2

    labels = _labels(R, p)
    logger.debug("Estimated weighted mean and covariance matrix.")
    return _wrap(mu, labels, True), _wrap(S, labels, False)


def shrink_mean_jorion(mu: ArrayLike, S: ArrayLike, T: int) -> ArrayLike:
    mu_arr, S_arr = np.asarray(mu), np.asarray(S)
    N = mu_arr.size
    if T <= 0 or N <= 2 or S_arr.shape != (N, N):
        logger.error(
            "Invalid dimensions for Jorion shrinkage: T=%d, N=%d, S shape=%s", T, N, S_arr.shape
        )
        raise ValueError("Invalid dimensions for Jorion shrinkage.")

    S_arr = (S_arr + S_arr.T) / 2
    try:
        S_inv = inv(S_arr + 1e-8 * np.eye(N))
    except LinAlgError as e:
        logger.error("Covariance matrix singular during inversion.")
        raise ValueError("Covariance matrix singular.") from e

    ones = np.ones(N)
    mu_gmv = (ones @ S_inv @ mu_arr) / (ones @ S_inv @ ones)
    diff = mu_arr - mu_gmv
    v = (N + 2) / ((N + 2) + T * (diff @ S_inv @ diff))
    v_clipped = np.clip(v, 0.0, 1.0)
    mu_bs = mu_arr - v_clipped * diff

    logger.debug("Applied Bayes-Stein shrinkage to mean vector.")
    return _wrap(mu_bs, _labels(mu, S), True)


def shrink_covariance_ledoit_wolf(
    R: ArrayLike,
    S_hat: ArrayLike,
    target: str = "identity",
) -> ArrayLike:
    R_arr, S_arr = np.asarray(R), np.asarray(S_hat)
    T, N = R_arr.shape
    if T == 0 or S_arr.shape != (N, N):
        logger.error(
            "Shape mismatch in inputs: R shape %s, S_hat shape %s", R_arr.shape, S_arr.shape
        )
        raise ValueError("Shape mismatch in inputs.")

    S_arr = (S_arr + S_arr.T) / 2
    X = R_arr - R_arr.mean(0)

    if target == "identity":
        F = np.eye(N) * np.trace(S_arr) / N
    elif target == "constant_correlation":
        std = np.sqrt(np.diag(S_arr))
        corr = S_arr / np.outer(std, std)
        r_bar = (corr.sum() - N) / (N * (N - 1))
        F = r_bar * np.outer(std, std)
        np.fill_diagonal(F, np.diag(S_arr))
    else:
        logger.error("Unsupported shrinkage target: %s", target)
        raise ValueError("Unsupported target: " + target)

    M = X[:, :, None] * X[:, None, :]
    pi_mat = np.mean((M - S_arr) ** 2, axis=0)
    pi_hat = np.mean(pi_mat)
    diag_pi = np.trace(pi_mat)
    off_pi = pi_hat - diag_pi

    if target == "identity":
        rho_hat = diag_pi
    else:
        rho_hat = diag_pi + ((F - np.diag(np.diag(F))).sum() / (N * (N - 1))) * off_pi

    gamma_hat = np.linalg.norm(S_arr - F, "fro") ** 2
    kappa = (pi_hat - rho_hat) / gamma_hat
    delta = float(np.clip(kappa if target == "identity" else kappa / T, 0.0, 1.0))

    Sigma = ensure_psd_matrix(delta * F + (1 - delta) * S_arr)
    Sigma = (Sigma + Sigma.T) / 2

    logger.debug("Applied Ledoit-Wolf shrinkage to covariance matrix.")
    return _wrap(Sigma, _labels(R, S_hat), False)

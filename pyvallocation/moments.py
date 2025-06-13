"""
This module provides functions for estimating and shrinking statistical moments
(mean and covariance matrix) of asset returns.

It includes:
-   `estimate_sample_moments`: For computing weighted sample mean and covariance.
-   `shrink_mean_jorion`: Implements the Bayes-Stein shrinkage estimator for the mean vector.
-   `shrink_covariance_ledoit_wolf`: Implements the Ledoit-Wolf shrinkage estimator for the covariance matrix.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy.linalg import LinAlgError, inv

from pyvallocation.utils.validation import (
    check_non_negativity,
    check_weights_sum_to_one,
    ensure_psd_matrix,
)

import pandas as pd

ArrayLike = Union[np.ndarray, "pd.Series", "pd.DataFrame"]

logger = logging.getLogger(__name__)


def _labels(*objs: ArrayLike) -> Optional[Sequence[str]]:
    for obj in objs:
        if isinstance(obj, pd.DataFrame):
            return obj.columns.to_list()
        if isinstance(obj, pd.Series):
            return obj.index.to_list()
    return None


def _wrap(x: np.ndarray, labels: Optional[Sequence[str]], vector: bool) -> ArrayLike:
    if labels is None:
        return x
    if vector:
        return pd.Series(x, index=labels, name="mu")
    return pd.DataFrame(x, index=labels, columns=labels)


def estimate_sample_moments(R: ArrayLike, p: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """
    Estimates the weighted mean vector and covariance matrix from scenarios.

    This function computes the first two statistical moments (mean and covariance)
    of asset returns, given a set of scenarios and their associated probabilities.
    The scenarios `R` represent different possible outcomes for asset returns,
    and `p` represents the probability of each scenario.

    Args:
        R (ArrayLike): A 2D array-like object (e.g., :class:`numpy.ndarray`,
            :class:`pandas.DataFrame`) of shape (T, N), where T is the number of
            scenarios/observations and N is the number of assets. Each row
            represents a scenario of asset returns.
        p (ArrayLike): A 1D array-like object (e.g., :class:`numpy.ndarray`,
            :class:`pandas.Series`) of shape (T,), representing the probabilities
            associated with each scenario in `R`. These probabilities must be
            non-negative and sum to one.

    Returns:
        Tuple[ArrayLike, ArrayLike]: A tuple containing:
            -   **mu** (ArrayLike): The weighted mean vector of asset returns.
                If `R` or `p` were pandas objects, `mu` will be a :class:`pandas.Series`.
            -   **S** (ArrayLike): The weighted covariance matrix of asset returns.
                If `R` or `p` were pandas objects, `S` will be a :class:`pandas.DataFrame`.

    Raises:
        ValueError: If `p` has a length mismatch with `R`, or if `p` contains
            negative values or does not sum to one.
    """
    R_arr, p_arr = np.asarray(R), np.asarray(p)
    T, N = R_arr.shape

    if p_arr.shape[0] != T:
        logger.error(
            "Weight length mismatch: weights length %d, returns length %d",
            p_arr.shape[0],
            T,
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
    """
    Applies Bayes–Stein shrinkage to the mean vector as in Jorion :cite:p:`jorion1986bayes`.

    This shrinkage estimator aims to improve the out-of-sample performance of
    mean estimates, especially when the number of assets (N) is large relative
    to the number of observations (T). It shrinks the sample mean towards a
    common mean (e.g., the global minimum variance portfolio mean).

    Args:
        mu (ArrayLike): The sample mean vector (1D array-like, length N).
            Can be a :class:`numpy.ndarray` or :class:`pandas.Series`.
        S (ArrayLike): The sample covariance matrix (2D array-like, N×N).
            Can be a :class:`numpy.ndarray` or :class:`pandas.DataFrame`.
        T (int): The number of observations (scenarios) used to estimate `mu` and `S`.

    Returns:
        ArrayLike: The Bayes-Stein shrunk mean vector. If `mu` was a
        :class:`pandas.Series`, the output will also be a :class:`pandas.Series`.

    Raises:
        ValueError: If input dimensions are invalid (e.g., T <= 0, N <= 2,
            or `S` shape mismatch), or if the covariance matrix `S` is singular.

    Notes:
        A small jitter (1e-8 * identity matrix) is added to `S` before inversion
        to handle potential singularity issues. The shrinkage intensity `v` is
        clipped between 0 and 1 to ensure a valid shrinkage factor.
    """
    mu_arr, S_arr = np.asarray(mu), np.asarray(S)
    N = mu_arr.size
    if T <= 0 or N <= 2 or S_arr.shape != (N, N):
        logger.error(
            "Invalid dimensions for Jorion shrinkage: T=%d, N=%d, S shape=%s",
            T,
            N,
            S_arr.shape,
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
    """
    Applies the Ledoit–Wolf shrinkage estimator for the covariance matrix :cite:p:`ledoit2004well`.

    This estimator provides a well-conditioned covariance matrix, especially useful
    when the number of observations is small relative to the number of assets,
    or when the sample covariance matrix is ill-conditioned. It shrinks the
    sample covariance matrix towards a structured target matrix.

    Args:
        R (ArrayLike): A 2D array-like object (e.g., :class:`numpy.ndarray`,
            :class:`pandas.DataFrame`) of shape (T, N), where T is the number of
            observations and N is the number of assets. These are the returns data.
        S_hat (ArrayLike): The sample covariance matrix (2D array-like, N×N).
            Can be a :class:`numpy.ndarray` or :class:`pandas.DataFrame`.
        target (str, optional): The shrinkage target.
            -   ``"identity"``: Shrinks towards a scaled identity matrix.
            -   ``"constant_correlation"``: Shrinks towards a constant correlation matrix.
            Defaults to ``"identity"``.

    Returns:
        ArrayLike: The shrunk covariance matrix. If `R` or `S_hat` were pandas
        objects, the output will be a :class:`pandas.DataFrame`.

    Raises:
        ValueError: If input dimensions are invalid (e.g., T = 0, or `S_hat`
            shape mismatch), or if an unsupported `target` is specified.

    Notes:
        The function calculates various components of the Ledoit-Wolf formula:

        *   `F`: The target matrix.
        *   `pi_mat`, `pi_hat`, `diag_pi`, `off_pi`, `rho_hat`: Components related
            to the estimation of the optimal shrinkage intensity.
        *   `gamma_hat`: The squared Frobenius norm of the difference between
            the sample covariance and the target matrix.
        *   `kappa`: Intermediate value for shrinkage intensity.
        *   `delta`: The optimal shrinkage intensity, clipped between 0 and 1.

        The final shrunk covariance matrix is ensured to be positive semi-definite
        using `ensure_psd_matrix`.
    """
    R_arr, S_arr = np.asarray(R), np.asarray(S_hat)
    T, N = R_arr.shape
    if T == 0 or S_arr.shape != (N, N):
        logger.error(
            "Shape mismatch in inputs: R shape %s, S_hat shape %s",
            R_arr.shape,
            S_arr.shape,
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

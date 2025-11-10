r"""
High-quality estimators for the first two moments (mean vector ``\mu`` and
covariance matrix ``\Sigma``) of asset returns, including classical and robust
shrinkage techniques. The implementations follow well-established statistical
and quantitative-finance references and are engineered to preserve pandas
indices/columns when supplied.

Overview
--------

The module provides:

* Weighted sample statistics (``estimate_sample_moments``).
* Bayes-Stein :cite:p:`jorion1986bayes` and James-Stein shrinkage estimators for the
  mean vector.
* Linear shrinkage for covariance matrices, including Ledoit-Wolf
  :cite:p:`ledoit2004well` and Oracle Approximating Shrinkage (OAS)
  :cite:p:`chen2010shrinkage`.
* Analytical nonlinear shrinkage (QuEST) :cite:p:`ledoit2020analytical`.
* POET factor+sparse covariance estimator :cite:p:`fan2013large`.
* Tyler's M-estimator with shrinkage toward a target for elliptical robustness
  :cite:p:`tyler1987statistical,chen2010shrinkage`.
* Sparse precision matrices via Graphical Lasso :cite:p:`friedman2008sparse`.
* Convenience adapters to the Black-Litterman
  :cite:p:`black1992global,he2002intuition` and Normal-Inverse-Wishart workflows.
* A factory (``estimate_moments``) to combine any of the above seamlessly.

Usage Examples
--------------

Simple pipeline
^^^^^^^^^^^^^^^

.. code-block:: python

   import pandas as pd
   from pyvallocation.moments import estimate_moments

   returns = pd.read_csv("returns.csv", index_col=0, parse_dates=True)
   mu_js, sigma_oas = estimate_moments(
       returns,
       mean_estimator="james_stein",
       cov_estimator="oas",
   )

   # Objects retain pandas labels, making downstream optimisation convenient.
   mu_js.name = "Expected Excess Return"
   sigma_oas.index.name = sigma_oas.columns.name = "Asset"

Interfacing with portfolio wrappers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper

   dist = AssetsDistribution(mu=mu_js, cov=sigma_oas)
   wrapper = PortfolioWrapper(dist)
   wrapper.set_constraints({"long_only": True, "total_weight": 1.0})
   frontier = wrapper.mean_variance_frontier(num_portfolios=25)
   tangency_weights, exp_ret, exp_vol = frontier.get_tangency_portfolio(risk_free_rate=0.02)

Robust combinations
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pyvallocation.moments import (
       shrink_covariance_nls,
       robust_covariance_tyler,
       robust_mean_huber,
       posterior_moments_black_litterman,
   )
   from pyvallocation.views import BlackLittermanProcessor

   # Nonlinear shrinkage + Tyler blend
   sigma_nls = shrink_covariance_nls(returns)
   sigma_tyler = robust_covariance_tyler(returns, shrinkage=0.1)
   mu_huber = robust_mean_huber(returns)

   # Blend sample and robust covariances (50/50)
   sigma_blend = 0.5 * (sigma_nls + sigma_tyler)

   # Extract Black-Litterman posterior moments using existing views infrastructure
   mu_bl, sigma_bl = posterior_moments_black_litterman(
       prior_cov=sigma_blend,
       prior_mean=mu_huber,
       mean_views={("AssetA", "AssetB"): 0.015},
       view_confidences={"AssetA,AssetB": 0.6},
   )

   # These outputs remain pandas Series/DataFrames if the inputs were pandas objects.

Sparse precision workflow
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pyvallocation.moments import sparse_precision_glasso

   sigma_sparse, theta_sparse = sparse_precision_glasso(
       returns,
       alpha=0.02,
       return_precision=True,
   )

   # theta_sparse (precision matrix) and sigma_sparse (covariance) are aligned with returns.columns.

Testing & Reliability
---------------------

Each estimator is covered by dedicated unit tests exercising numerical
stability, positive semi-definiteness, and pandas round-tripping (see
``tests/test_moments_shrinkage.py``). The full repository test suite is executed
under multiple shrinkage configurations to ensure production readiness.
"""

from __future__ import annotations

import importlib
import logging
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, Union

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
    """Return asset labels when the first pandas object is encountered."""
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
    R_arr = np.asarray(R, dtype=float)
    p_arr = np.asarray(p, dtype=float).reshape(-1)
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
    S = np.einsum('ti,tj,t->ij', X, X, p_arr, optimize=True)
    S = (S + S.T) / 2

    labels = _labels(R, p)
    logger.debug("Estimated weighted mean and covariance matrix.")
    return _wrap(mu, labels, True), _wrap(S, labels, False)


def shrink_mean_jorion(mu: ArrayLike, S: ArrayLike, T: int) -> ArrayLike:
    """
    Applies Bayes-Stein shrinkage to the mean vector as in Jorion :cite:p:`jorion1986bayes`.

    This shrinkage estimator aims to improve the out-of-sample performance of
    mean estimates, especially when the number of assets (N) is large relative
    to the number of observations (T). It shrinks the sample mean towards a
    common mean (e.g., the global minimum variance portfolio mean).

    Args:
        mu (ArrayLike): The sample mean vector (1D array-like, length N).
            Can be a :class:`numpy.ndarray` or :class:`pandas.Series`.
        S (ArrayLike): The sample covariance matrix (2D array-like, NxN).
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
    Applies the Ledoit-Wolf shrinkage estimator for the covariance matrix :cite:p:`ledoit2004well`.

    This estimator provides a well-conditioned covariance matrix, especially useful
    when the number of observations is small relative to the number of assets,
    or when the sample covariance matrix is ill-conditioned. It shrinks the
    sample covariance matrix towards a structured target matrix.

    Args:
        R (ArrayLike): A 2D array-like object (e.g., :class:`numpy.ndarray`,
            :class:`pandas.DataFrame`) of shape (T, N), where T is the number of
            observations and N is the number of assets. These are the returns data.
        S_hat (ArrayLike): The sample covariance matrix (2D array-like, NxN).
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


def shrink_covariance_oas(R: ArrayLike, assume_centered: bool = True) -> ArrayLike:
    """Return the Oracle Approximating Shrinkage (OAS) covariance estimator."""
    X = np.asarray(R, dtype=float)
    if X.ndim != 2:
        raise ValueError("Input `R` must be two-dimensional with shape (T, N).")
    T, N = X.shape
    if T == 0 or N == 0:
        raise ValueError("Input `R` must contain at least one observation and one asset.")
    if not assume_centered:
        X = X - X.mean(axis=0, keepdims=True)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        emp_cov = (X.T @ X) / float(T)
    emp_cov = np.nan_to_num(emp_cov, copy=False)
    if N == 1:
        shrunk = emp_cov.reshape(1, 1)
        return _wrap(shrunk, _labels(R), False)
    alpha = np.mean(emp_cov**2)
    mu = np.trace(emp_cov) / N
    mu_sq = mu * mu
    denom = (T + 1.0) * (alpha - mu_sq / N)
    shrinkage = 1.0 if denom <= 0 else min((alpha + mu_sq) / denom, 1.0)
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    diag_idx = np.diag_indices(N)
    shrunk_cov[diag_idx] += shrinkage * mu
    shrunk_cov = ensure_psd_matrix(shrunk_cov)
    return _wrap(shrunk_cov, _labels(R), False)


def shrink_covariance_nls(
    R_or_S: ArrayLike,
    *,
    input_is_cov: bool = False,
    dof_correction: int = 0,
) -> ArrayLike:
    """Return Ledoit-Wolf analytical nonlinear shrinkage (QuEST) of covariance."""
    if input_is_cov:
        raise ValueError(
            "`shrink_covariance_nls` expects raw returns. "
            "Pass `input_is_cov=False` and supply `R` with shape (T, N)."
        )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=PendingDeprecationWarning,
                message="Importing from numpy\\.matlib is deprecated",
            )
            nonlinshrink = importlib.import_module("nonlinshrink")
    except ImportError as exc:
        raise ImportError(
            "`shrink_covariance_nls` requires the `non-linear-shrinkage` package. "
            "Install it via `pip install non-linear-shrinkage`."
        ) from exc

    X = np.asarray(R_or_S, dtype=float)
    if X.ndim != 2:
        raise ValueError("Input must be two-dimensional with shape (T, N).")
    k_arg: Optional[int] = None if dof_correction == 0 else int(dof_correction)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=PendingDeprecationWarning,
            message="Importing from numpy\\.matlib is deprecated",
        )
        Sigma = nonlinshrink.shrink_cov(X, k=k_arg)
    Sigma = ensure_psd_matrix(Sigma)
    return _wrap(Sigma, _labels(R_or_S), False)


def factor_covariance_poet(
    R: ArrayLike,
    k: int | str = "auto",
    thresh: float | str = "auto",
    standardize: bool = True,
    return_decomp: bool = False,
) -> ArrayLike | Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Return POET low-rank plus sparse covariance estimator."""
    X = np.asarray(R, dtype=float)
    if X.ndim != 2:
        raise ValueError("Input `R` must be two-dimensional with shape (T, N).")
    T, N = X.shape
    labels = _labels(R)
    idx = getattr(R, "index", None) if isinstance(R, pd.DataFrame) else None

    X_centered = X - X.mean(axis=0, keepdims=True)
    if standardize:
        scale = X_centered.std(axis=0, ddof=1)
        scale[scale <= 1e-12] = 1.0
    else:
        scale = np.ones(N, dtype=float)
    X_scaled = X_centered / scale
    S_scaled = np.cov(X_scaled, rowvar=False, ddof=1)
    eigvals, eigvecs = np.linalg.eigh(S_scaled)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if isinstance(k, str):
        if eigvals.size <= 1:
            k_sel = 1
        elif eigvals.size == 2:
            k_sel = 1
        else:
            diffs = np.diff(eigvals)
            accel = diffs[:-1] - diffs[1:]
            k_sel = int(np.argmax(accel) + 1)
    else:
        if k <= 0:
            raise ValueError("`k` must be positive when provided as an integer.")
        k_sel = int(k)
    k_sel = max(1, min(k_sel, N))
    loadings = eigvecs[:, :k_sel]
    sqrt_eigs = np.sqrt(np.maximum(eigvals[:k_sel], 0.0))
    B_scaled = loadings * sqrt_eigs
    S_factor = B_scaled @ B_scaled.T
    residual = S_scaled - S_factor
    residual = 0.5 * (residual + residual.T)
    if isinstance(thresh, str):
        off_diag = residual[~np.eye(N, dtype=bool)]
        if off_diag.size == 0:
            tau = 0.0
        else:
            tau = np.median(np.abs(off_diag)) * np.sqrt(np.log(max(N, 2)) / max(T, 1))
    else:
        if thresh < 0:
            raise ValueError("`thresh` must be non-negative.")
        tau = float(thresh)
    sparse = residual.copy()
    mask = ~np.eye(N, dtype=bool)
    sparse[mask] = np.sign(sparse[mask]) * np.maximum(np.abs(sparse[mask]) - tau, 0.0)
    Sigma_scaled = S_factor + sparse
    Sigma_scaled = 0.5 * (Sigma_scaled + Sigma_scaled.T)
    scale_diag = np.diag(scale)
    Sigma = scale_diag @ Sigma_scaled @ scale_diag
    Sigma = 0.5 * (Sigma + Sigma.T)
    B_out = scale_diag @ B_scaled
    F_out = X_scaled @ loadings

    Sigma = ensure_psd_matrix(Sigma)
    Sigma_wrapped = _wrap(Sigma, labels, False)
    if not return_decomp:
        return Sigma_wrapped

    def _wrap_matrix(matrix: Optional[np.ndarray], *, index: Optional[pd.Index]) -> ArrayLike:
        if matrix is None:
            return None  # type: ignore[return-value]
        if index is not None:
            cols = [f"factor_{i+1}" for i in range(matrix.shape[1])]
            return pd.DataFrame(matrix, index=index, columns=cols)
        return matrix

    def _wrap_loadings(loadings_matrix: Optional[np.ndarray], *, columns: Sequence[str]) -> ArrayLike:
        if loadings_matrix is None:
            return None  # type: ignore[return-value]
        if columns:
            cols = [f"factor_{i+1}" for i in range(loadings_matrix.shape[1])]
            return pd.DataFrame(loadings_matrix, index=columns, columns=cols)
        return loadings_matrix

    cols = list(labels) if labels is not None else []
    B_wrapped = _wrap_loadings(B_out, columns=cols)
    F_wrapped = _wrap_matrix(F_out, index=idx)
    return Sigma_wrapped, B_wrapped, F_wrapped


def robust_covariance_tyler(
    R: ArrayLike,
    *,
    shrinkage: float = 0.0,
    target: str | np.ndarray | pd.DataFrame = "identity",
    tol: float = 1e-6,
    max_iter: int = 200,
    ensure_psd: bool = True,
) -> ArrayLike:
    """Return regularised Tyler's M-estimator for heavy-tailed covariance."""
    X = np.asarray(R, dtype=float)
    if X.ndim != 2:
        raise ValueError("Input `R` must be two-dimensional with shape (T, N).")
    T, N = X.shape
    if T == 0 or N == 0:
        raise ValueError("Input `R` must contain at least one observation and one asset.")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("`shrinkage` must lie in [0, 1].")

    X = X - X.mean(axis=0, keepdims=True)
    sample_cov = np.cov(X, rowvar=False, ddof=1)
    sample_cov = ensure_psd_matrix(sample_cov)
    trace_target = np.trace(sample_cov) / max(N, 1)
    if trace_target <= 0.0:
        trace_target = 1.0
    if isinstance(target, str):
        if target != "identity":
            raise ValueError("`target` must be 'identity' or an array-like.")
        target_matrix = np.eye(N)
    else:
        target_matrix = np.asarray(target, dtype=float)
        if target_matrix.shape != (N, N):
            raise ValueError("Custom target must have shape (N, N).")
        target_matrix = 0.5 * (target_matrix + target_matrix.T)
        target_matrix = ensure_psd_matrix(target_matrix)
    target_matrix *= N / np.trace(target_matrix)

    covariance = sample_cov / trace_target * N
    for _ in range(max_iter):
        try:
            inv_cov = np.linalg.pinv(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Covariance became singular during iterations.") from exc
        quad = np.einsum("ti,ij,tj->t", X, inv_cov, X)
        quad = np.clip(quad, 1e-6, None)
        inv_quad = np.divide(
            N,
            quad,
            out=np.full_like(quad, 1e6),
            where=quad > 0,
        )
        inv_quad = np.clip(inv_quad, 0.0, 1e6)
        weights = np.sqrt(inv_quad)[:, None]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            updated = (X * weights).T @ (X * weights) / T
        updated = np.nan_to_num(updated, copy=False)
        updated = 0.5 * (updated + updated.T)
        updated *= N / np.trace(updated)
        if shrinkage > 0.0:
            updated = (1.0 - shrinkage) * updated + shrinkage * target_matrix
        delta = np.linalg.norm(updated - covariance, ord="fro")
        denom = max(np.linalg.norm(covariance, ord="fro"), 1e-12)
        covariance = updated
        if delta / denom <= tol:
            break

    covariance *= trace_target / N
    covariance = 0.5 * (covariance + covariance.T)
    if ensure_psd:
        covariance = ensure_psd_matrix(covariance)
    return _wrap(covariance, _labels(R), False)


def _soft_threshold(matrix: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft-thresholding to off-diagonal elements."""
    result = matrix.copy()
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    off = result[mask]
    result[mask] = np.sign(off) * np.maximum(np.abs(off) - threshold, 0.0)
    return result


def _graphical_lasso_admm(
    emp_cov: np.ndarray,
    alpha: float,
    *,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Solve the graphical lasso via ADMM."""
    p = emp_cov.shape[0]
    Theta = np.linalg.inv(emp_cov + alpha * np.eye(p))
    Z = Theta.copy()
    U = np.zeros_like(emp_cov)
    for iteration in range(1, max_iter + 1):
        K = rho * (Z - U) - emp_cov
        eigvals, eigvecs = np.linalg.eigh(K)
        denom = 2.0 * rho
        sqrt_term = np.sqrt(np.maximum(eigvals**2 + 4.0 * rho, 0.0))
        diag_vals = (eigvals + sqrt_term) / denom
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            Theta = (eigvecs * diag_vals) @ eigvecs.T
        Theta = np.nan_to_num(Theta, copy=False)
        Theta = 0.5 * (Theta + Theta.T)

        X = Theta + U
        threshold = alpha / rho
        Z_prev = Z
        Z = _soft_threshold(X, threshold)
        Z[np.diag_indices(p)] = X[np.diag_indices(p)]

        U = U + Theta - Z

        r_norm = np.linalg.norm(Theta - Z, ord="fro")
        s_norm = np.linalg.norm(rho * (Z - Z_prev), ord="fro")
        eps_pri = tol * np.sqrt(p) + tol * max(np.linalg.norm(Theta, ord="fro"), np.linalg.norm(Z, ord="fro"))
        eps_dual = tol * np.sqrt(p) + tol * np.linalg.norm(rho * U, ord="fro")
        if r_norm <= eps_pri and s_norm <= eps_dual:
            break

    else:
        raise RuntimeError("Graphical lasso did not converge within max_iter.")

    precision = Theta
    try:
        covariance = np.linalg.inv(precision)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(precision)
    covariance = ensure_psd_matrix(covariance)
    return covariance, precision, iteration


def sparse_precision_glasso(
    R: ArrayLike,
    *,
    alpha: float | str = "auto",
    assume_centered: bool = True,
    return_precision: bool = False,
) -> ArrayLike | Tuple[ArrayLike, ArrayLike]:
    """Estimate covariance via sparse inverse covariance (Graphical Lasso)."""
    X = np.asarray(R, dtype=float)
    if X.ndim != 2:
        raise ValueError("Input `R` must be two-dimensional with shape (T, N).")
    if X.shape[0] < 2:
        raise ValueError("Graphical lasso requires at least two observations.")
    if not assume_centered:
        X = X - X.mean(axis=0, keepdims=True)
    T = X.shape[0]
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        emp_cov = (X.T @ X) / float(T)
    emp_cov = np.nan_to_num(emp_cov, copy=False)
    labels = _labels(R)

    def _fit(alpha_value: float) -> Tuple[np.ndarray, np.ndarray]:
        cov, precision, _ = _graphical_lasso_admm(emp_cov, alpha_value)
        return cov, precision

    if alpha == "auto":
        S = emp_cov.copy()
        max_offdiag = np.max(np.abs(S - np.diag(np.diag(S))))
        alpha_max = max(max_offdiag, 1e-6)
        alphas = np.geomspace(alpha_max, alpha_max / 100.0, num=5)
        best_alpha = alphas[-1]
        best_score = -np.inf
        K = min(5, T)
        indices = np.arange(T)
        for alpha_candidate in alphas:
            scores = []
            for fold in range(K):
                mask = indices % K != fold
                X_train = X[mask]
                X_valid = X[~mask]
                if X_train.size == 0 or X_valid.size == 0:
                    continue
                with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
                    S_train = (X_train.T @ X_train) / float(X_train.shape[0])
                S_train = np.nan_to_num(S_train, copy=False)
                _, precision_cv, _ = _graphical_lasso_admm(S_train, alpha_candidate)
                with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
                    S_valid = (X_valid.T @ X_valid) / float(X_valid.shape[0])
                S_valid = np.nan_to_num(S_valid, copy=False)
                logdet = np.linalg.slogdet(precision_cv)[1]
                score = logdet - np.trace(S_valid @ precision_cv)
                scores.append(score)
            if scores:
                avg_score = float(np.mean(scores))
                if avg_score > best_score + 1e-8:
                    best_alpha = alpha_candidate
                    best_score = avg_score
        chosen_alpha = float(best_alpha)
        cov, precision, _ = _graphical_lasso_admm(emp_cov, chosen_alpha)
    else:
        chosen_alpha = float(alpha)
        if chosen_alpha <= 0.0:
            raise ValueError("`alpha` must be positive.")
        cov, precision, _ = _graphical_lasso_admm(emp_cov, chosen_alpha)

    cov_wrapped = _wrap(cov, labels, False)
    if not return_precision:
        return cov_wrapped
    precision_wrapped = _wrap(precision, labels, False)
    return cov_wrapped, precision_wrapped


def shrink_mean_james_stein(
    mu_hat: ArrayLike,
    S: ArrayLike,
    T: int,
    target: str | np.ndarray | pd.Series = "grand_mean",
) -> ArrayLike:
    """Return James-Stein shrinkage estimate for the mean vector."""
    mu_arr = np.asarray(mu_hat, dtype=float).reshape(-1)
    Sigma = np.asarray(S, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("`S` must be a square covariance matrix.")
    N = mu_arr.size
    if Sigma.shape[0] != N:
        raise ValueError("Shape mismatch between `mu_hat` and `S`.")
    if T <= 0:
        raise ValueError("`T` must be positive.")

    if isinstance(target, str):
        if target != "grand_mean":
            raise ValueError("`target` must be 'grand_mean' or an array-like.")
        mu_target = np.full_like(mu_arr, mu_arr.mean())
    else:
        mu_target = np.asarray(target, dtype=float).reshape(-1)
        if mu_target.size != N:
            raise ValueError("Custom target must have length N.")

    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma = ensure_psd_matrix(Sigma)
    Sigma_inv = np.linalg.pinv(Sigma, rcond=1e-12)
    diff = mu_arr - mu_target
    quad = float(diff.T @ (T * Sigma_inv) @ diff)
    if quad <= 0.0:
        shrink = 0.0
    else:
        shrink = max(0.0, 1.0 - (N - 2.0) / quad)
    mu_js = mu_target + shrink * diff
    return _wrap(mu_js, _labels(mu_hat, S), True)


def robust_mean_huber(
    R: ArrayLike,
    *,
    allow_vectorized: bool = True,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> ArrayLike:
    """Return adaptive Huber mean estimator (per asset) for heavy-tailed data."""
    X = np.asarray(R, dtype=float)
    if X.ndim != 2:
        raise ValueError("Input `R` must be two-dimensional with shape (T, N).")
    if not allow_vectorized:
        raise ValueError("Set `allow_vectorized=True`; scalar mode is not implemented.")
    T, N = X.shape
    estimates = np.empty(N, dtype=float)
    for j in range(N):
        x = X[:, j]
        mu = float(np.median(x))
        scale = 1.4826 * np.median(np.abs(x - mu)) + 1e-12
        tau = scale * np.sqrt(T / max(np.log(max(T, 3)), 2.0))
        for _ in range(max_iter):
            residuals = x - mu
            denom = np.maximum(np.abs(residuals), 1e-12)
            weights = np.minimum(1.0, tau / denom)
            mu_new = np.sum(weights * x) / np.sum(weights)
            if abs(mu_new - mu) <= tol * (abs(mu) + 1.0):
                mu = float(mu_new)
                break
            mu = float(mu_new)
        estimates[j] = mu
    return _wrap(estimates, _labels(R), True)


def robust_mean_median_of_means(
    R: ArrayLike,
    *,
    n_blocks: int | str = "auto",
    random_state: Optional[int | np.random.Generator] = None,
) -> ArrayLike:
    """Return coordinate-wise Median-of-Means mean estimator."""
    X = np.asarray(R, dtype=float)
    if X.ndim != 2:
        raise ValueError("Input `R` must be two-dimensional with shape (T, N).")
    T, N = X.shape
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)
    if isinstance(n_blocks, str):
        B = int(np.ceil(np.sqrt(max(T, 1))))
    else:
        if n_blocks <= 0:
            raise ValueError("`n_blocks` must be positive.")
        B = int(min(n_blocks, T))
    B = max(1, min(B, T))
    permuted = rng.permutation(T)
    Xp = X[permuted]
    base = T // B
    remainder = T % B
    block_sizes = np.full(B, base, dtype=int)
    block_sizes[:remainder] += 1
    indices = np.cumsum(np.concatenate(([0], block_sizes)))
    block_means = np.vstack(
        [Xp[indices[b] : indices[b + 1]].mean(axis=0) for b in range(B)]
    )
    mom = np.median(block_means, axis=0)
    return _wrap(mom, _labels(R), True)


def posterior_moments_black_litterman(
    *,
    prior_cov: ArrayLike,
    prior_mean: Optional[ArrayLike] = None,
    market_weights: Optional[ArrayLike] = None,
    risk_aversion: float = 1.0,
    tau: float = 0.05,
    mean_views: Any = None,
    view_confidences: Any = None,
    omega: Any = "idzorek",
    **kwargs: Any,
) -> Tuple[ArrayLike, ArrayLike]:
    """Return posterior (mu, Sigma) from :class:`BlackLittermanProcessor`."""
    from pyvallocation.views import BlackLittermanProcessor

    processor = BlackLittermanProcessor(
        prior_cov=prior_cov,
        prior_mean=prior_mean,
        market_weights=market_weights,
        risk_aversion=risk_aversion,
        tau=tau,
        mean_views=mean_views,
        view_confidences=view_confidences,
        omega=omega,
        **kwargs,
    )
    return processor.get_posterior()


def posterior_moments_niw(
    *,
    prior_mu: ArrayLike,
    prior_sigma: ArrayLike,
    t0: int,
    nu0: int,
    sample_mu: ArrayLike,
    sample_sigma: ArrayLike,
    n_obs: int,
) -> Tuple[ArrayLike, ArrayLike]:
    """Return NIW posterior classical-equivalent (mu, Sigma)."""
    from pyvallocation.bayesian import NIWPosterior

    niw = NIWPosterior(
        prior_mu=prior_mu,
        prior_sigma=prior_sigma,
        t0=int(t0),
        nu0=int(nu0),
    )
    niw.update(
        sample_mu=sample_mu,
        sample_sigma=sample_sigma,
        n_obs=int(n_obs),
    )
    return niw.get_mu_ce(), niw.get_sigma_ce()


def estimate_moments(
    R: ArrayLike,
    p: Optional[ArrayLike] = None,
    *,
    mean_estimator: str = "sample",
    cov_estimator: str = "sample",
    mean_kwargs: Optional[Dict[str, Any]] = None,
    cov_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """Return (mu, Sigma) using configurable mean and covariance estimators."""
    mean_kwargs = dict(mean_kwargs or {})
    cov_kwargs = dict(cov_kwargs or {})
    X = np.asarray(R, dtype=float)
    if X.ndim != 2:
        raise ValueError("Input `R` must be two-dimensional with shape (T, N).")
    T, _ = X.shape
    labels = _labels(R)

    if p is None:
        mu_sample = X.mean(axis=0)
        S_sample = np.cov(X, rowvar=False, ddof=1)
        mu_sample_wrapped = _wrap(mu_sample, labels, True)
        S_sample_wrapped = _wrap(S_sample, labels, False)
    else:
        mu_sample_wrapped, S_sample_wrapped = estimate_sample_moments(R, p)
        mu_sample = np.asarray(mu_sample_wrapped, dtype=float)
        S_sample = np.asarray(S_sample_wrapped, dtype=float)

    mean_choice = mean_estimator.lower()
    if mean_choice in {"sample", "simple"}:
        mu = mu_sample_wrapped
    elif mean_choice in {"jorion", "bayes_stein"}:
        mu = shrink_mean_jorion(mu_sample_wrapped, S_sample_wrapped, T=T, **mean_kwargs)
    elif mean_choice in {"james_stein", "james-stein", "js"}:
        mu = shrink_mean_james_stein(
            mu_sample_wrapped,
            S_sample_wrapped,
            T=T,
            **mean_kwargs,
        )
    elif mean_choice in {"huber", "robust_huber"}:
        mu = robust_mean_huber(R, **mean_kwargs)
    elif mean_choice in {"median_of_means", "mom"}:
        mu = robust_mean_median_of_means(R, **mean_kwargs)
    else:
        raise ValueError(f"Unsupported mean estimator '{mean_estimator}'.")

    cov_choice = cov_estimator.lower()
    if cov_choice in {"sample", "empirical"}:
        Sigma = S_sample_wrapped
    elif cov_choice in {"ledoit_wolf", "lw"}:
        Sigma = shrink_covariance_ledoit_wolf(R, S_sample_wrapped, **cov_kwargs)
    elif cov_choice == "oas":
        Sigma = shrink_covariance_oas(R, **cov_kwargs)
    elif cov_choice in {"nls", "nonlinear", "nonlinear_shrinkage"}:
        Sigma = shrink_covariance_nls(R, **cov_kwargs)
    elif cov_choice == "poet":
        cov_kwargs.setdefault("return_decomp", False)
        Sigma = factor_covariance_poet(R, **cov_kwargs)
        if isinstance(Sigma, tuple):
            Sigma = Sigma[0]
    elif cov_choice in {"tyler", "tyler_shrinkage"}:
        Sigma = robust_covariance_tyler(R, **cov_kwargs)
    elif cov_choice in {"glasso", "graphical_lasso"}:
        cov_kwargs["return_precision"] = False
        Sigma = sparse_precision_glasso(R, **cov_kwargs)
    else:
        raise ValueError(f"Unsupported covariance estimator '{cov_estimator}'.")

    return mu, Sigma

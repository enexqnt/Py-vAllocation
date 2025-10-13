"""
Stress-testing helpers built on top of the single-period scenario engine.

The utilities in this module reuse the existing scenario notation (``R`` for
scenarios, ``p`` for probabilities) and risk functions to evaluate allocations
under probability tilts and linear scenario transforms.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .moments import estimate_sample_moments
from .probabilities import (
    compute_effective_number_scenarios,
    generate_exp_decay_probabilities,
    generate_gaussian_kernel_probabilities,
    generate_uniform_probabilities,
)
from .utils.functions import portfolio_cvar, portfolio_var

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
ProbabilityLike = Union[np.ndarray, pd.Series, Sequence[float]]
WeightsLike = Union[np.ndarray, pd.Series, pd.DataFrame, Mapping[str, float]]

__all__ = [
    "stress_test",
    "exp_decay_stress",
    "kernel_focus_stress",
    "entropy_pooling_stress",
    "linear_map",
]


def _ensure_probabilities(p: Optional[ProbabilityLike], n_obs: int) -> np.ndarray:
    if p is None:
        return generate_uniform_probabilities(n_obs)
    p_arr = np.asarray(p, dtype=float).reshape(-1)
    if p_arr.shape[0] != n_obs:
        raise ValueError("Probability vector length must match the number of scenarios.")
    if np.any(p_arr < 0):
        raise ValueError("Scenario probabilities must be non-negative.")
    total = float(np.sum(p_arr))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Scenario probabilities must sum to a positive finite value.")
    if not np.isclose(total, 1.0):
        p_arr = p_arr / total
    return p_arr


def _align_weights(
    weights: WeightsLike,
    asset_names: Optional[Sequence[str]],
) -> Tuple[np.ndarray, Optional[Sequence[str]], Sequence[str]]:
    """
    Return weight matrix shaped (N, M), the asset order, and portfolio names.
    """
    if isinstance(weights, pd.DataFrame):
        matrix = weights.to_numpy(dtype=float)
        idx = list(weights.index)
        names = list(weights.columns)
    elif isinstance(weights, pd.Series):
        matrix = weights.to_numpy(dtype=float).reshape(-1, 1)
        idx = list(weights.index)
        names = [weights.name or "portfolio_0"]
    elif isinstance(weights, Mapping):
        series = pd.Series(weights, dtype=float)
        matrix = series.to_numpy(dtype=float).reshape(-1, 1)
        idx = list(series.index)
        names = [series.name or "portfolio_0"]
    else:
        arr = np.asarray(weights, dtype=float)
        if arr.ndim == 1:
            matrix = arr.reshape(-1, 1)
            idx = None
            names = ["portfolio_0"]
        elif arr.ndim == 2:
            matrix = arr
            idx = None
            names = [f"portfolio_{i}" for i in range(arr.shape[1])]
        else:
            raise ValueError("`weights` must be 1D or 2D.")

    if asset_names is not None:
        if idx is None:
            if matrix.shape[0] != len(asset_names):
                raise ValueError("Weight dimension does not match number of assets.")
        else:
            if set(idx) != set(asset_names):
                missing = sorted(set(asset_names) - set(idx))
                extra = sorted(set(idx) - set(asset_names))
                raise ValueError(
                    f"Weight labels do not match asset names. Missing={missing}, extra={extra}"
                )
            order = [idx.index(name) for name in asset_names]
            matrix = matrix[order, :]
            idx = list(asset_names)
    return matrix, idx, names


def _kl_divergence(p_star: np.ndarray, p_nom: np.ndarray) -> float:
    eps = 1e-16
    p1 = np.clip(p_star, eps, None)
    p0 = np.clip(p_nom, eps, None)
    return float(np.sum(p1 * np.log(p1 / p0)))


def stress_test(
    weights: WeightsLike,
    scenarios: ArrayLike,
    *,
    probabilities: Optional[ProbabilityLike] = None,
    stressed_probabilities: Optional[ProbabilityLike] = None,
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    alpha: float = 0.95,
    demean: bool = False,
) -> pd.DataFrame:
    """
    Evaluate allocations under nominal and stressed conditions.

    Parameters
    ----------
    weights
        Portfolio weights (Series, DataFrame, numpy array, or mapping).
    scenarios
        Scenario matrix ``R`` with shape ``(T, N)``. Can be a pandas object or ndarray.
    probabilities
        Optional nominal scenario probabilities ``p``. Defaults to uniform.
    stressed_probabilities
        Optional stressed probabilities ``p*`` on the same scenarios.
    transform
        Callable applied to ``R`` (after conversion to ``float``) to obtain stressed scenarios.
    alpha
        Confidence level for VaR/CVaR metrics.
    demean
        When ``True`` the P&L used for VaR/CVaR is demeaned by the respective probabilities.

    Returns
    -------
    pandas.DataFrame
        Tidy DataFrame with nominal metrics, stressed metrics (when supplied),
        effective number of scenarios, and KL divergence (when both probability
        vectors are provided).
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("`alpha` must be in (0, 1).")

    if isinstance(scenarios, pd.DataFrame):
        R_arr = scenarios.to_numpy(dtype=float)
        asset_names = list(scenarios.columns)
    else:
        R_arr = np.asarray(scenarios, dtype=float)
        asset_names = None

    if R_arr.ndim != 2:
        raise ValueError("`scenarios` must be a 2D array-like.")

    n_scenarios, n_assets = R_arr.shape
    p_nom = _ensure_probabilities(probabilities, n_scenarios)
    W, aligned_assets, portfolio_names = _align_weights(weights, asset_names)
    if W.shape[0] != n_assets:
        raise ValueError("Weight dimension does not match number of assets in `scenarios`.")

    def _metrics(
        R_matrix: np.ndarray, probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        mu, sigma = estimate_sample_moments(R_matrix, probs)
        mu_vec = np.asarray(mu, dtype=float).reshape(-1, 1)
        sigma_mat = np.asarray(sigma, dtype=float)
        mean = (W.T @ mu_vec).reshape(-1)
        var = np.asarray(portfolio_var(W, R_matrix, probs, alpha=alpha, demean=demean)).reshape(-1)
        cvar = np.asarray(portfolio_cvar(W, R_matrix, probs, alpha=alpha, demean=demean)).reshape(-1)
        stdev = np.sqrt(np.einsum("ik,kl,il->i", W.T, sigma_mat, W.T))
        ens = compute_effective_number_scenarios(probs)
        return mean, stdev, var, cvar, ens

    ret_nom, sd_nom, var_nom, cvar_nom, ens_nom = _metrics(R_arr, p_nom)

    columns = {
        "return_nom": ret_nom,
        "stdev_nom": sd_nom,
        f"VaR{int(round(alpha * 100))}_nom": var_nom,
        f"CVaR{int(round(alpha * 100))}_nom": cvar_nom,
        "ENS_nom": np.full_like(ret_nom, ens_nom, dtype=float),
    }

    stressed_R = R_arr
    if transform is not None:
        stressed_R = np.asarray(transform(R_arr), dtype=float)
        if stressed_R.shape != R_arr.shape:
            raise ValueError("`transform` must return an array with the same shape as `scenarios`.")

    if stressed_probabilities is not None or transform is not None:
        p_star = (
            _ensure_probabilities(stressed_probabilities, n_scenarios)
            if stressed_probabilities is not None
            else p_nom
        )
        ret_st, sd_st, var_st, cvar_st, ens_st = _metrics(stressed_R, p_star)
        columns.update(
            {
                "return_stress": ret_st,
                "stdev_stress": sd_st,
                f"VaR{int(round(alpha * 100))}_stress": var_st,
                f"CVaR{int(round(alpha * 100))}_stress": cvar_st,
                "ENS_stress": np.full_like(ret_st, ens_st, dtype=float),
            }
        )
        if stressed_probabilities is not None:
            columns["KL_q_p"] = np.full_like(ret_st, _kl_divergence(p_star, p_nom), dtype=float)

    df = pd.DataFrame(columns, index=portfolio_names)
    if aligned_assets is not None:
        df.index.name = "portfolio"
    return df


def exp_decay_stress(
    weights: WeightsLike,
    scenarios: ArrayLike,
    *,
    probabilities: Optional[ProbabilityLike] = None,
    half_life: int = 60,
    alpha: float = 0.95,
    demean: bool = False,
) -> pd.DataFrame:
    """
    Stress test using historical-simulation flexible probabilities with half-life decay.
    """
    scenario_array = np.asarray(scenarios, dtype=float)
    p_star = generate_exp_decay_probabilities(scenario_array.shape[0], half_life)
    return stress_test(
        weights,
        scenarios,
        probabilities=probabilities,
        stressed_probabilities=p_star,
        alpha=alpha,
        demean=demean,
    )


def kernel_focus_stress(
    weights: WeightsLike,
    scenarios: ArrayLike,
    *,
    focus_series: ArrayLike,
    probabilities: Optional[ProbabilityLike] = None,
    bandwidth: Optional[float] = None,
    target: Optional[float] = None,
    alpha: float = 0.95,
    demean: bool = False,
) -> pd.DataFrame:
    """
    Stress test that concentrates probabilities around a target state via Gaussian kernels.
    """
    v = np.asarray(focus_series, dtype=float).reshape(-1)
    if v.shape[0] != np.asarray(scenarios, dtype=float).shape[0]:
        raise ValueError("`focus_series` length must match the number of scenarios.")
    p_star = generate_gaussian_kernel_probabilities(v, h=bandwidth, x_T=target)
    return stress_test(
        weights,
        scenarios,
        probabilities=probabilities,
        stressed_probabilities=p_star,
        alpha=alpha,
        demean=demean,
    )


def entropy_pooling_stress(
    weights: WeightsLike,
    scenarios: ArrayLike,
    *,
    posterior_probabilities: ProbabilityLike,
    probabilities: Optional[ProbabilityLike] = None,
    alpha: float = 0.95,
    demean: bool = False,
) -> pd.DataFrame:
    """
    Evaluate allocations under posterior probabilities generated by entropy pooling.
    """
    return stress_test(
        weights,
        scenarios,
        probabilities=probabilities,
        stressed_probabilities=posterior_probabilities,
        alpha=alpha,
        demean=demean,
    )


def linear_map(
    *,
    mean_shift: Optional[ArrayLike] = None,
    scale: Optional[float] = None,
    matrix: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a linear scenario transform ``R -> R @ matrix.T * scale + mean_shift``.
    """

    def _transform(R: np.ndarray) -> np.ndarray:
        X = np.asarray(R, dtype=float)
        if matrix is not None:
            mat = np.asarray(matrix, dtype=float)
            X = X @ mat.T
        if scale is not None:
            X = X * float(scale)
        if mean_shift is not None:
            shift = np.asarray(mean_shift, dtype=float)
            if shift.ndim == 1:
                if shift.shape[0] != X.shape[1]:
                    raise ValueError("`mean_shift` length must match the number of assets.")
                X = X + shift
            else:
                raise ValueError("`mean_shift` must be a 1D array-like.")
        return X

    return _transform

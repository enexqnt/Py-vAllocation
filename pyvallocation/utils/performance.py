"""
Performance summaries for single-period scenario analysis.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..moments import estimate_sample_moments
from ..probabilities import compute_effective_number_scenarios, generate_uniform_probabilities
from ..utils.functions import portfolio_cvar, portfolio_var

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
ProbabilityLike = Union[np.ndarray, pd.Series, Sequence[float]]
WeightsLike = Union[np.ndarray, pd.Series, pd.DataFrame]

__all__ = ["scenario_pnl", "performance_report"]


def scenario_pnl(weights: WeightsLike, scenarios: ArrayLike) -> ArrayLike:
    """
    Compute scenario-by-scenario portfolio P&L.

    Parameters
    ----------
    weights :
        Portfolio weights. Accepts numpy arrays, pandas Series/DataFrames, or a
        mapping ``asset -> weight``.
    scenarios :
        Scenario matrix ``R`` of shape ``(T, N)`` (NumPy or pandas). When a
        pandas object is supplied, the returned object preserves the original
        index/columns.

    Returns
    -------
    array-like
        Scenario P&L with shape ``(T,)`` or ``(T, M)`` depending on the number
        of portfolios supplied.

    Raises
    ------
    ValueError
        If scenario dimensions are inconsistent with the weight vector/matrix.
    """
    if isinstance(scenarios, pd.DataFrame):
        arr = scenarios.to_numpy(dtype=float)
        columns = list(scenarios.columns)
        index = scenarios.index
    else:
        arr = np.asarray(scenarios, dtype=float)
        columns = None
        index = None

    if arr.ndim != 2:
        raise ValueError("`scenarios` must be a 2D array-like.")

    if isinstance(weights, Mapping):
        weights = pd.Series(weights, dtype=float)

    if isinstance(weights, pd.Series):
        if columns is not None and list(weights.index) != columns:
            weights = weights.reindex(columns)
        w = weights.to_numpy(dtype=float).reshape(-1, 1)
    elif isinstance(weights, pd.DataFrame):
        if columns is not None and list(weights.index) != columns:
            weights = weights.reindex(columns)
        w = weights.to_numpy(dtype=float)
    else:
        arr_w = np.asarray(weights, dtype=float)
        if arr_w.ndim == 1:
            w = arr_w.reshape(-1, 1)
        else:
            w = arr_w

    if w.shape[0] != arr.shape[1]:
        raise ValueError("Weight dimension must match the number of assets.")

    pnl = arr @ w
    if pnl.ndim == 1:
        pnl = pnl.reshape(-1, 1)

    if isinstance(weights, pd.DataFrame):
        df = pd.DataFrame(pnl, index=index, columns=weights.columns)
        return df
    if isinstance(weights, pd.Series):
        series = pd.Series(pnl[:, 0], index=index, name=weights.name)
        return series
    return pnl.squeeze()


def performance_report(
    weights: WeightsLike,
    scenarios: ArrayLike,
    *,
    probabilities: Optional[ProbabilityLike] = None,
    alpha: float = 0.95,
    demean: bool = False,
) -> pd.Series:
    """
    Summarise mean, volatility, VaR, CVaR, and ENS for a single allocation.

    Parameters
    ----------
    weights :
        Allocation vector (numpy array or pandas Series/DataFrame with a single
        column). Labels are aligned with ``scenarios`` when present.
    scenarios :
        Scenario matrix ``R`` with shape ``(T, N)`` (NumPy or pandas).
    probabilities :
        Optional scenario weights ``p``. When omitted a uniform distribution is
        used.
    alpha :
        Confidence level used for VaR/CVaR (default 0.95).
    demean :
        If ``True`` the scenario P&L is demeaned before VaR/CVaR are computed.

    Returns
    -------
    pandas.Series
        Series containing the portfolio mean, standard deviation, VaR, CVaR, and
        effective number of scenarios.

    Raises
    ------
    ValueError
        If inputs are inconsistent (e.g. mismatched dimensions or invalid
        probabilities).
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

    n_obs = R_arr.shape[0]
    if probabilities is None:
        p = generate_uniform_probabilities(n_obs)
    else:
        p = np.asarray(probabilities, dtype=float).reshape(-1)
        if p.shape[0] != n_obs:
            raise ValueError("Probability vector length must match number of scenarios.")
        total = p.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Scenario probabilities must sum to a positive finite value.")
        if not np.isclose(total, 1.0):
            p = p / total

    if isinstance(weights, pd.Series):
        if asset_names is not None and list(weights.index) != asset_names:
            weights = weights.reindex(asset_names)
        w = weights.to_numpy(dtype=float).reshape(-1, 1)
    elif isinstance(weights, pd.DataFrame):
        if weights.shape[1] != 1:
            raise ValueError("`weights` must represent a single allocation.")
        if asset_names is not None and list(weights.index) != asset_names:
            weights = weights.reindex(asset_names)
        w = weights.to_numpy(dtype=float)
    else:
        arr_w = np.asarray(weights, dtype=float)
        if arr_w.ndim == 1:
            w = arr_w.reshape(-1, 1)
        else:
            if arr_w.shape[1] != 1:
                raise ValueError("`weights` must represent a single allocation.")
            w = arr_w
    if w.shape[0] != R_arr.shape[1]:
        raise ValueError("Weight dimension must match the number of assets.")
    w_vec = w.reshape(-1)

    mu, cov = estimate_sample_moments(R_arr, p)
    mu_vec = np.asarray(mu, dtype=float).reshape(-1, 1)
    cov_mat = np.asarray(cov, dtype=float)

    mean = float(np.dot(w_vec, mu_vec.reshape(-1)))
    stdev = float(np.sqrt(w_vec @ cov_mat @ w_vec))
    w_matrix = w_vec.reshape(-1, 1)
    var = float(portfolio_var(w_matrix, R_arr, p, alpha=alpha, demean=demean))
    cvar = float(portfolio_cvar(w_matrix, R_arr, p, alpha=alpha, demean=demean))
    ens = compute_effective_number_scenarios(p)

    return pd.Series(
        {
            "mean": mean,
            "stdev": stdev,
            f"VaR{int(round(alpha * 100))}": var,
            f"CVaR{int(round(alpha * 100))}": cvar,
            "ENS": ens,
        }
    )

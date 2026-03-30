"""
Performance summaries for single-period scenario analysis.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..moments import estimate_sample_moments
from ..probabilities import (
    compute_effective_number_scenarios,
    resolve_probabilities,
)
from ..utils.functions import portfolio_cvar, portfolio_var

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
ProbabilityLike = Union[np.ndarray, pd.Series, Sequence[float]]
WeightsLike = Union[np.ndarray, pd.Series, pd.DataFrame]

__all__ = ["scenario_pnl", "performance_report", "horizon_report", "drawdown_quantile"]


def scenario_pnl(weights: WeightsLike, scenarios: ArrayLike) -> ArrayLike:
    """
    Compute scenario-by-scenario portfolio P&L.

    Args:
        weights: Portfolio weights. Accepts numpy arrays, pandas Series/DataFrames,
            or a mapping ``asset -> weight``.
        scenarios: Scenario matrix ``R`` of shape ``(T, N)`` (NumPy or pandas). When a
            pandas object is supplied, the returned object preserves the original
            index/columns.

    Returns:
        ArrayLike: Scenario P&L with shape ``(T,)`` or ``(T, M)`` depending on the
        number of portfolios supplied.

    Raises:
        ValueError: If scenario dimensions are inconsistent with the weight vector/matrix.

    Examples:
        >>> import numpy as np
        >>> from pyvallocation.utils.performance import scenario_pnl
        >>> scenarios = np.array([[0.02, 0.00], [0.00, 0.02]])
        >>> scenario_pnl([0.5, 0.5], scenarios)
        array([0.01, 0.01])
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
    confidence: float = 0.95,
    demean: bool = False,
) -> pd.Series:
    """
    Summarise mean, volatility, VaR, CVaR, and ENS for a single allocation.

    Args:
        weights: Allocation vector (numpy array or pandas Series/DataFrame with a single
            column). Labels are aligned with ``scenarios`` when present.
        scenarios: Scenario matrix ``R`` with shape ``(T, N)`` (NumPy or pandas).
        probabilities: Optional scenario weights ``p``. When omitted a uniform
            distribution is used.
        confidence: Confidence level for VaR/CVaR (e.g. 0.95 means 5% tail).
            Defaults to 0.95.
        demean: If ``True`` the scenario P&L is demeaned before VaR/CVaR are computed.

    Returns:
        pd.Series: Series containing the portfolio mean, standard deviation, VaR, CVaR,
        and effective number of scenarios.

    Raises:
        ValueError: If inputs are inconsistent (e.g. mismatched dimensions or invalid
            probabilities).

    Notes:
        VaR and CVaR follow the loss convention: profitable scenarios appear as
        negative numbers (gains) while losses are positive.

    Examples:
        >>> import numpy as np
        >>> from pyvallocation.utils.performance import performance_report
        >>> scenarios = np.array([[0.02, 0.00], [0.02, 0.00]])
        >>> performance_report([0.5, 0.5], scenarios).round(4)
        mean      0.0100
        stdev     0.0000
        VaR95    -0.0100
        CVaR95   -0.0100
        ENS       2.0000
        dtype: float64
    """
    conf = confidence
    if not 0.0 < conf < 1.0:
        raise ValueError("`confidence` must be in (0, 1).")

    if isinstance(scenarios, pd.DataFrame):
        R_arr = scenarios.to_numpy(dtype=float)
        asset_names = list(scenarios.columns)
    else:
        R_arr = np.asarray(scenarios, dtype=float)
        asset_names = None
    if R_arr.ndim != 2:
        raise ValueError("`scenarios` must be a 2D array-like.")

    p = resolve_probabilities(probabilities, R_arr.shape[0])

    if isinstance(weights, pd.Series):
        if asset_names is not None and list(weights.index) != asset_names:
            weights = weights.reindex(asset_names)
        if weights.isna().any():
            raise ValueError(
                "Weight labels do not match scenario asset names after reindexing."
            )
        w = weights.to_numpy(dtype=float).reshape(-1, 1)
    elif isinstance(weights, pd.DataFrame):
        if weights.shape[1] != 1:
            raise ValueError("`weights` must represent a single allocation.")
        if asset_names is not None and list(weights.index) != asset_names:
            weights = weights.reindex(asset_names)
        if weights.isna().any().any():
            raise ValueError(
                "Weight labels do not match scenario asset names after reindexing."
            )
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
    var = float(portfolio_var(w_matrix, R_arr, p, confidence=conf, demean=demean))
    cvar = float(portfolio_cvar(w_matrix, R_arr, p, confidence=conf, demean=demean))
    ens = compute_effective_number_scenarios(p)

    return pd.Series(
        {
            "mean": mean,
            "stdev": stdev,
            f"VaR{int(round(conf * 100))}": var,
            f"CVaR{int(round(conf * 100))}": cvar,
            "ENS": ens,
        }
    )


def horizon_report(
    weights: WeightsLike,
    invariants: ArrayLike,
    *,
    horizons: Sequence[int] = (4, 13, 26, 52),
    n_simulations: int = 5000,
    p: Optional[ProbabilityLike] = None,
    reprice=None,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Compare risk metrics across multiple projection horizons.

    For each horizon, bootstraps invariants via
    :func:`~pyvallocation.utils.projection.project_scenarios`, reprices,
    and computes :func:`performance_report`.

    Args:
        weights: Portfolio weights.
        invariants: Invariant scenarios ``(T, N)``.
        horizons: Sequence of horizon lengths (in invariant time steps).
        n_simulations: Scenarios per horizon.
        p: Optional scenario probabilities.
        reprice: Repricing callable (default ``reprice_exp``).
        confidence: Confidence level for VaR/CVaR.
        seed: Base random seed (incremented per horizon for independence).

    Returns:
        pd.DataFrame: Rows = horizons, columns = mean, stdev, VaR, CVaR, ENS.
    """
    from .projection import project_scenarios, reprice_exp as _reprice_exp

    if reprice is None:
        reprice = _reprice_exp

    default_labels = {4: "1m", 13: "3m", 26: "6m", 52: "1y"}
    rows = {}
    for i, h in enumerate(horizons):
        h_seed = (seed + i) if seed is not None else None
        scen = project_scenarios(
            invariants, investment_horizon=h, p=p,
            n_simulations=n_simulations, reprice=reprice,
        )
        report = performance_report(
            weights, scen, confidence=confidence,
        )
        label = default_labels.get(h, f"{h}w")
        rows[label] = report

    return pd.DataFrame(rows).T


def drawdown_quantile(
    weights: WeightsLike,
    invariants: ArrayLike,
    horizon: int,
    *,
    confidence: float = 0.95,
    n_paths: int = 1000,
    p: Optional[ProbabilityLike] = None,
    reprice=None,
    seed: Optional[int] = None,
) -> pd.Series:
    """Compute the maximum drawdown distribution via path simulation.

    Simulates full wealth paths, computes the maximum peak-to-trough
    drawdown of each path, and returns summary statistics.

    Args:
        weights: Portfolio weights.
        invariants: Invariant scenarios ``(T, N)``.
        horizon: Path length in invariant time steps.
        confidence: Quantile level for the drawdown statistic.
        n_paths: Number of simulated paths.
        p: Optional scenario probabilities.
        reprice: Repricing callable (default ``reprice_exp``).
        seed: Random seed.

    Returns:
        pd.Series: Keys ``max_dd_mean``, ``max_dd_median``,
        ``max_dd_{confidence}``, ``max_dd_worst``.
    """
    from .projection import simulate_paths, reprice_exp as _reprice_exp

    if reprice is None:
        reprice = _reprice_exp

    w = np.asarray(weights, dtype=float).ravel()
    paths = simulate_paths(
        invariants, horizon=horizon, n_paths=n_paths,
        p=p, reprice=reprice, seed=seed,
    )  # (n_paths, horizon, N)

    # Portfolio cumulative returns at each step
    port_cum = paths @ w  # (n_paths, horizon)
    wealth = 1.0 + port_cum  # wealth relative to $1

    # Running maximum and drawdown
    running_max = np.maximum.accumulate(wealth, axis=1)
    drawdowns = (running_max - wealth) / np.where(running_max > 0, running_max, 1.0)
    max_dd = drawdowns.max(axis=1)  # (n_paths,)

    pct = int(round(confidence * 100))
    return pd.Series({
        "max_dd_mean": float(max_dd.mean()),
        "max_dd_median": float(np.median(max_dd)),
        f"max_dd_{pct}": float(np.quantile(max_dd, confidence)),
        "max_dd_worst": float(max_dd.max()),
    })

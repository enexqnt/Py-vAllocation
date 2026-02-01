"""Shared utilities for runnable examples."""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper


ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]


def ensure_psd(cov: ArrayLike, *, jitter: float = 1e-6) -> ArrayLike:
    """Return a PSD covariance by adding minimal diagonal jitter when needed.

    Args:
        cov: Covariance matrix.
        jitter: Diagonal jitter to enforce PSD.

    Returns:
        ArrayLike: PSD-adjusted covariance.
    """
    if isinstance(cov, pd.DataFrame):
        cov_values = cov.to_numpy(dtype=float)
        min_eig = float(np.linalg.eigvalsh(cov_values).min())
        if min_eig <= 0:
            cov = cov + np.eye(len(cov)) * (abs(min_eig) + jitter)
        return cov
    cov_arr = np.asarray(cov, dtype=float)
    min_eig = float(np.linalg.eigvalsh(cov_arr).min())
    if min_eig <= 0:
        cov_arr = cov_arr + np.eye(cov_arr.shape[0]) * (abs(min_eig) + jitter)
    return cov_arr


def build_wrapper_from_scenarios(
    scenarios: ArrayLike,
    *,
    probabilities: Optional[Sequence[float]] = None,
    bounds: Optional[tuple[Optional[float], Optional[float]]] = None,
    total_weight: float = 1.0,
) -> PortfolioWrapper:
    """Create a long-only wrapper from scenario data.

    Args:
        scenarios: Scenario matrix or DataFrame.
        probabilities: Optional scenario probabilities.
        bounds: Optional box bounds for weights.
        total_weight: Sum of weights (default ``1.0``).

    Returns:
        PortfolioWrapper: Configured wrapper instance.
    """
    dist = AssetsDistribution(scenarios=scenarios, probabilities=probabilities)
    wrapper = PortfolioWrapper(dist)
    constraints = {"long_only": True, "total_weight": total_weight}
    if bounds is not None:
        constraints["bounds"] = bounds
    wrapper.set_constraints(constraints)
    return wrapper


def build_wrapper_from_moments(
    mu: ArrayLike,
    cov: ArrayLike,
    *,
    bounds: Optional[tuple[Optional[float], Optional[float]]] = None,
    total_weight: float = 1.0,
) -> PortfolioWrapper:
    """Create a long-only wrapper from mean/covariance moments.

    Args:
        mu: Mean vector.
        cov: Covariance matrix.
        bounds: Optional box bounds for weights.
        total_weight: Sum of weights (default ``1.0``).

    Returns:
        PortfolioWrapper: Configured wrapper instance.
    """
    cov_psd = ensure_psd(cov)
    dist = AssetsDistribution(mu=mu, cov=cov_psd)
    wrapper = PortfolioWrapper(dist)
    constraints = {"long_only": True, "total_weight": total_weight}
    if bounds is not None:
        constraints["bounds"] = bounds
    wrapper.set_constraints(constraints)
    return wrapper


def print_portfolio(
    title: str,
    weights: Union[pd.Series, np.ndarray],
    expected_return: float,
    risk: float,
    *,
    risk_label: str,
) -> None:
    """Pretty-print portfolio weights and metrics.

    Args:
        title: Section title.
        weights: Portfolio weights.
        expected_return: Expected return value.
        risk: Risk metric value.
        risk_label: Label for the risk metric.
    """
    print(title)
    if isinstance(weights, pd.Series):
        print(weights.round(4))
    else:
        print(np.round(np.asarray(weights, dtype=float), 4))
    print(f"Expected return: {expected_return:.4%} | {risk_label}: {risk:.4%}\n")

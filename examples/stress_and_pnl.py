"""
Stress-testing and performance summary demo.

This example shows how to evaluate a portfolio with the new stress-testing and
performance utilities:

1. Build a long-only mean-variance tangency portfolio.
2. Generate a performance report on nominal probabilities.
3. Evaluate historical half-life stress and a kernel-focused scenario tilt.
4. Apply a simple linear scenario shock and evaluate the combined impact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper
from pyvallocation.stress import (
    exp_decay_stress,
    kernel_focus_stress,
    linear_map,
    stress_test,
)
from pyvallocation.utils.performance import performance_report

DATA_PATH = Path(__file__).resolve().parents[1] / "examples" / "ETF_prices.csv"


def load_weekly_returns() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Sample data not found at {DATA_PATH}")
    prices = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True).ffill()
    weekly = prices.resample("W-FRI").last().dropna(how="all")
    returns = weekly.pct_change().dropna()
    return returns.rename(columns=lambda c: c.replace(" ", "_"))


def build_tangency_portfolio(returns: pd.DataFrame) -> pd.Series:
    dist = AssetsDistribution(scenarios=np.log1p(returns))
    wrapper = PortfolioWrapper(dist)
    wrapper.set_constraints({"long_only": True, "total_weight": 1.0})
    frontier = wrapper.mean_variance_frontier(num_portfolios=25)
    weights, *_ = frontier.get_tangency_portfolio(risk_free_rate=0.01)
    return weights


def main() -> None:
    returns = load_weekly_returns()
    weights = build_tangency_portfolio(returns)

    print("=== Nominal performance ===")
    report = performance_report(weights, returns.values, alpha=0.95)
    print(report.round(4))

    print("\n=== Half-life stress (60 weeks) ===")
    df_half_life = exp_decay_stress(weights, returns.values, half_life=60)
    print(df_half_life.round(4))

    print("\n=== Kernel focus stress on SPY drawdowns ===")
    focus = returns["SPY"].rolling(12).std(ddof=0).fillna(method="bfill")
    df_kernel = kernel_focus_stress(
        weights,
        returns.values,
        focus_series=focus.values,
        bandwidth=None,
        target=focus.values.max(),
    )
    print(df_kernel.round(4))

    print("\n=== Combined scenario shock and EP-style reweight ===")
    scale_up = linear_map(scale=1.25)
    # Ad-hoc posterior probabilities favour first decile of scenarios
    tail_weights = np.linspace(1.0, 2.0, num=returns.shape[0])
    tail_weights /= tail_weights.sum()
    df_combo = stress_test(
        weights,
        returns.values,
        stressed_probabilities=tail_weights,
        transform=scale_up,
    )
    print(df_combo.round(4))


if __name__ == "__main__":
    main()


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

import numpy as np
import pandas as pd

from data_utils import load_prices
from pyvallocation import PortfolioWrapper
from pyvallocation.stress import exp_decay_stress, kernel_focus_stress, linear_map, stress_test
from pyvallocation.utils.performance import performance_report


def load_weekly_returns() -> pd.DataFrame:
    prices = load_prices().dropna(how="all")
    weekly = prices.resample("W-FRI").last().dropna(how="all")
    returns = weekly.pct_change().dropna()
    return returns.rename(columns=lambda c: c.replace(" ", "_"))


def build_tangency_portfolio(returns: pd.DataFrame) -> pd.Series:
    wrapper = PortfolioWrapper.from_moments(returns.mean(), returns.cov())
    frontier = wrapper.variance_frontier(num_portfolios=25)
    weights, *_ = frontier.tangency(risk_free_rate=0.01)
    return weights


def main() -> None:
    returns = load_weekly_returns()
    weights = build_tangency_portfolio(returns)

    print("=== Nominal performance ===")
    report = performance_report(weights, returns.values, confidence=0.95)
    print(report.round(4))

    print("\n=== Half-life stress (60 weeks) ===")
    df_half_life = exp_decay_stress(weights, returns.values, half_life=60)
    print(df_half_life.round(4))

    print("\n=== Kernel focus stress on SPY drawdowns ===")
    focus = returns["SPY"].rolling(12).std(ddof=0).bfill()
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

"""Demonstrate the robust optimiser across a range of uncertainty budgets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper

DATA_PATH = Path(__file__).with_name("ETF_prices.csv")


def load_moments() -> tuple[pd.Series, pd.DataFrame]:
    prices = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    prices = prices.dropna(how="all")
    returns = prices.pct_change().dropna(how="any")
    mu = returns.mean()
    cov = returns.cov()
    min_eig = float(np.linalg.eigvalsh(cov.values).min())
    if min_eig <= 0:
        cov += np.eye(len(cov)) * (abs(min_eig) + 1e-6)
    return mu, cov


def main() -> None:
    mu, cov = load_moments()
    distribution = AssetsDistribution(mu=mu, cov=cov)

    wrapper = PortfolioWrapper(distribution)
    wrapper.set_constraints({"long_only": True, "total_weight": 1.0})

    frontier = wrapper.robust_lambda_frontier(num_portfolios=6, max_lambda=1.5)
    min_risk_weights, min_risk_return, min_risk_radius = frontier.get_min_risk_portfolio()
    max_return_weights, max_return_return, max_return_radius = frontier.get_max_return_portfolio()

    print("=== Robust frontier summary ===")
    print("Lambda grid:", np.linspace(0.0, 1.5, 6))
    print("Returns:", frontier.returns.round(4))
    print("Risk radii:", frontier.risks.round(4), "\n")

    print("Minimum estimation-risk portfolio")
    print(min_risk_weights.round(4))
    print(f"Nominal return: {min_risk_return:.4%} | Risk radius: {min_risk_radius:.4f}\n")

    print("Maximum nominal-return portfolio")
    print(max_return_weights.round(4))
    print(f"Nominal return: {max_return_return:.4%} | Risk radius: {max_return_radius:.4f}")


if __name__ == "__main__":
    main()

"""Solve a mean-CVaR allocation problem on a small ETF universe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper

DATA_PATH = Path(__file__).with_name("ETF_prices.csv")


def load_returns() -> pd.DataFrame:
    prices = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    prices = prices.dropna(how="all")
    returns = prices.pct_change().dropna(how="any")
    return returns


def main() -> None:
    returns = load_returns().iloc[-750:]  # recent history only
    probabilities = np.full(len(returns), 1.0 / len(returns))

    distribution = AssetsDistribution(
        scenarios=returns,
        probabilities=probabilities,
    )
    wrapper = PortfolioWrapper(distribution)
    wrapper.set_constraints({"long_only": True, "total_weight": 1.0})

    # Target a modest return in line with the median scenario outcome.
    target_return = float(np.median(returns.mean()))
    weights, expected_return, cvar = wrapper.mean_cvar_portfolio_at_return(
        return_target=target_return,
        alpha=0.05,
    )

    frontier = wrapper.mean_cvar_frontier(num_portfolios=7, alpha=0.05)
    tangency_weights, tangency_return, tangency_risk = frontier.get_tangency_portfolio(0.001)

    print("=== Mean-CVaR target portfolio ===")
    print(weights.round(4))
    print(f"Expected return: {expected_return:.4%} | 95% CVaR: {cvar:.4%}\n")

    print("=== Tangency portfolio on the CVaR frontier (rf=0.1%) ===")
    print(tangency_weights.round(4))
    print(f"Expected return: {tangency_return:.4%} | 95% CVaR: {tangency_risk:.4%}")


if __name__ == "__main__":
    main()

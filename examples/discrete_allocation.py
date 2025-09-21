"""Convert continuous weights into discrete share counts for a model portfolio."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper

DATA_PATH = Path(__file__).with_name("ETF_prices.csv")


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    prices = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    prices = prices.ffill().dropna()
    returns = prices.pct_change().dropna().iloc[-750:]
    latest_prices = prices.iloc[-1]
    return returns, latest_prices


def main() -> None:
    returns, latest_prices = load_data()

    distribution = AssetsDistribution(scenarios=returns)
    wrapper = PortfolioWrapper(distribution)
    wrapper.set_constraints({"long_only": True, "total_weight": 1.0})

    frontier = wrapper.mean_variance_frontier(num_portfolios=8)
    # Select a middle point on the frontier for demonstration.
    middle_index = frontier.weights.shape[1] // 2
    allocation = frontier.as_discrete_allocation(
        column=middle_index,
        latest_prices=latest_prices,
        total_value=25000.0,
    )

    print("=== Continuous weights ===")
    target_weights = pd.Series(frontier.weights[:, middle_index], index=frontier.asset_names)
    print(target_weights.round(4))

    print("\n=== Discrete allocation ===")
    print(allocation.as_series())
    print(f"Leftover cash: ${allocation.leftover_cash:,.2f}")
    print(f"Tracking error: {allocation.tracking_error:.4%}")


if __name__ == "__main__":
    main()

"""Convert continuous weights into discrete share counts for a model portfolio."""

from __future__ import annotations

import pandas as pd

from data_utils import load_prices
from example_utils import build_wrapper_from_scenarios


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    prices = load_prices().dropna()
    returns = prices.pct_change().dropna().iloc[-750:]
    latest_prices = prices.iloc[-1]
    return returns, latest_prices


def main() -> None:
    returns, latest_prices = load_data()

    wrapper = build_wrapper_from_scenarios(returns)

    frontier = wrapper.variance_frontier(num_portfolios=8)
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

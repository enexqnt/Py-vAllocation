"""Minimal example: build a mean-variance frontier and report key portfolios."""

from __future__ import annotations

import pandas as pd

from example_utils import build_wrapper_from_scenarios, print_portfolio


def main() -> None:
    data = pd.DataFrame(
        {
            "Equity_US": [0.021, -0.013, 0.018, 0.007, 0.011],
            "Equity_EU": [0.017, -0.009, 0.014, 0.004, 0.006],
            "Credit_US": [0.009, 0.008, 0.007, 0.006, 0.008],
            "Govt_Bonds": [0.004, 0.003, 0.005, 0.002, 0.004],
        }
    )

    wrapper = build_wrapper_from_scenarios(data)

    frontier = wrapper.variance_frontier(num_portfolios=6)
    min_risk_w, min_risk_return, min_risk_vol = frontier.get_min_risk_portfolio()
    max_return_w, max_return_return, max_return_vol = frontier.get_max_return_portfolio()

    print_portfolio(
        "Minimum-risk portfolio",
        min_risk_w,
        min_risk_return,
        min_risk_vol,
        risk_label="Volatility",
    )
    print_portfolio(
        "Maximum-return portfolio",
        max_return_w,
        max_return_return,
        max_return_vol,
        risk_label="Volatility",
    )


if __name__ == "__main__":
    main()

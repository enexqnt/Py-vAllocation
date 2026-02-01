"""Solve a mean-CVaR allocation problem on a small ETF universe."""

from __future__ import annotations

import numpy as np

from data_utils import load_returns
from example_utils import build_wrapper_from_scenarios, print_portfolio


def main() -> None:
    returns = load_returns().iloc[-750:]  # recent history only
    probabilities = np.full(len(returns), 1.0 / len(returns))
    wrapper = build_wrapper_from_scenarios(returns, probabilities=probabilities)

    # Target a modest return in line with the median scenario outcome.
    target_return = float(np.median(returns.mean()))
    weights, expected_return, cvar = wrapper.min_cvar_at_return(
        return_target=target_return,
        alpha=0.05,
    )

    frontier = wrapper.cvar_frontier(num_portfolios=7, alpha=0.05)
    tangency_weights, tangency_return, tangency_risk = frontier.get_tangency_portfolio(0.001)

    print_portfolio(
        "=== Mean-CVaR target portfolio ===",
        weights,
        expected_return,
        cvar,
        risk_label="CVaR (alpha=0.05)",
    )
    print_portfolio(
        "=== Tangency portfolio on the CVaR frontier (rf=0.1%) ===",
        tangency_weights,
        tangency_return,
        tangency_risk,
        risk_label="CVaR (alpha=0.05)",
    )


if __name__ == "__main__":
    main()

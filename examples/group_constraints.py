"""Sector allocation limits with typed Constraints."""
from __future__ import annotations

import numpy as np
from pyvallocation import PortfolioWrapper, Constraints


def main():
    mu = np.array([0.10, 0.08, 0.06, 0.04, 0.03])
    cov = np.eye(5) * 0.03 + 0.01
    np.fill_diagonal(cov, [0.05, 0.04, 0.03, 0.025, 0.02])

    # Define sector constraints
    constraints = Constraints(
        group_constraints={
            "Equity": ([0, 1], 0.30, 0.60),          # 30-60% in equities
            "Fixed Income": ([2, 3, 4], 0.30, 0.70),  # 30-70% in fixed income
        },
        bounds=(0.05, 0.40),  # 5-40% per asset
    )

    wrapper = PortfolioWrapper.from_moments(mu, cov, constraints=constraints)
    frontier = wrapper.variance_frontier(num_portfolios=8)

    print("=== Sector-Constrained Frontier ===")
    for i in range(frontier.weights.shape[1]):
        w = frontier.weights[:, i]
        eq = w[0] + w[1]
        fi = w[2] + w[3] + w[4]
        print(
            f"P{i}: Equity={eq:.1%}  FI={fi:.1%}  "
            f"Return={frontier.returns[i]:.2%}  Vol={frontier.risks[i]:.2%}"
        )


if __name__ == "__main__":
    main()

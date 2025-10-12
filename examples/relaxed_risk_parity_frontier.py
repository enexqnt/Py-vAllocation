"""Demonstrate relaxed risk parity frontiers with Py-vAllocation."""

from __future__ import annotations

import numpy as np

from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper


def main() -> None:
    mu = np.array([0.08, 0.06, 0.05, 0.03])
    cov = np.array(
        [
            [0.090, 0.040, 0.025, 0.010],
            [0.040, 0.070, 0.020, 0.015],
            [0.025, 0.020, 0.060, 0.018],
            [0.010, 0.015, 0.018, 0.045],
        ]
    )

    dist = AssetsDistribution(mu=mu, cov=cov, asset_names=["Tech", "Health", "Value", "Bonds"])
    wrapper = PortfolioWrapper(dist)
    wrapper.set_constraints({"long_only": True, "total_weight": 1.0})

    frontier = wrapper.relaxed_risk_parity_frontier(
        num_portfolios=4,
        max_multiplier=1.5,
        lambda_reg=0.25,
    )

    weights = frontier.to_frame(column_labels=[f"P{i}" for i in range(frontier.weights.shape[1])])
    print("Relaxed Risk Parity Frontier Weights")
    print(weights.round(4))

    if frontier.metadata:
        print("\nDiagnostics per frontier point:")
        for idx, meta in enumerate(frontier.metadata):
            print(
                f"  P{idx}: lambda={meta['lambda_reg']:.2f}, "
                f"multiplier={meta['target_multiplier']}, "
                f"requested={meta['requested_target']}, "
                f"effective={meta['effective_target']}, "
                f"objective={meta['objective']:.6f}"
            )


if __name__ == "__main__":
    main()

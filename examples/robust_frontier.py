"""Demonstrate the robust optimiser across a range of uncertainty budgets."""

from __future__ import annotations

import numpy as np

from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper
from data_utils import load_returns
from example_utils import ensure_psd, print_portfolio


def main() -> None:
    returns = load_returns()
    mu = returns.mean()
    cov = ensure_psd(returns.cov())

    # Robust optimiser expects uncertainty covariance of the mean.
    # Under i.i.d. sampling, Var(mean) ≈ Sigma / T.
    uncertainty_cov = cov / max(len(returns), 1)
    distribution = AssetsDistribution(mu=mu, cov=uncertainty_cov)

    wrapper = PortfolioWrapper(distribution)
    wrapper.set_constraints({"long_only": True, "total_weight": 1.0})

    lambdas = np.geomspace(0.05, 1.5, 6)
    frontier = wrapper.robust_lambda_frontier(
        num_portfolios=len(lambdas),
        lambdas=lambdas,
        return_cov=cov,
    )
    min_risk_weights, min_risk_return, min_risk_radius = frontier.get_min_risk_portfolio()
    max_return_weights, max_return_return, max_return_radius = frontier.get_max_return_portfolio()

    print("=== Robust frontier summary ===")
    print("Lambda grid:", np.round(lambdas, 3))
    print("Returns:", frontier.returns.round(4))
    print("Estimation risk radius:", frontier.risks.round(4), "\n")

    print_portfolio(
        "Minimum estimation-risk portfolio",
        min_risk_weights,
        min_risk_return,
        min_risk_radius,
        risk_label="Estimation Risk",
    )
    print_portfolio(
        "Maximum nominal-return portfolio",
        max_return_weights,
        max_return_return,
        max_return_radius,
        risk_label="Estimation Risk",
    )


if __name__ == "__main__":
    main()

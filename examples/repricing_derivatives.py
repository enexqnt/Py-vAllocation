"""Repricing workflow: stocks, bonds, and options following the Prayer framework."""
from __future__ import annotations

import numpy as np
from pyvallocation import (
    project_scenarios,
    reprice_exp,
    reprice_taylor,
    make_repricing_fn,
    PortfolioWrapper,
    AssetsDistribution,
)


def main():
    rng = np.random.default_rng(42)

    # --- Stock: invariant = log return, reprice = exp ---
    stock_invariants = rng.normal(0.0005, 0.015, size=(500, 1))  # daily log returns
    stock_scenarios = project_scenarios(
        stock_invariants, investment_horizon=21, n_simulations=2000,
        reprice=reprice_exp,
    )
    print(
        f"Stock monthly simple-return scenarios: "
        f"mean={stock_scenarios.mean():.4f}, std={stock_scenarios.std():.4f}"
    )

    # --- Bond: invariant = yield change, reprice = full pricing ---
    yield_changes = rng.normal(0.0, 0.003, size=(500, 1))  # daily yield changes
    current_yield = np.array([0.04])
    maturity = 10
    bond_price = lambda Y: 100 * np.exp(-Y * maturity)  # noqa: E731
    bond_scenarios = project_scenarios(
        yield_changes, investment_horizon=21, n_simulations=2000,
        reprice=make_repricing_fn(bond_price, current_yield),
    )
    print(
        f"Bond monthly P&L scenarios: "
        f"mean={bond_scenarios.mean():.4f}, std={bond_scenarios.std():.4f}"
    )

    # --- Option: invariant = (log-return, vol-change), reprice = delta-gamma ---
    option_invariants = rng.multivariate_normal(
        [0.0005, 0.0], [[0.015**2, 0.0], [0.0, 0.005**2]], size=500,
    )
    delta_opt, gamma_opt, vega, theta = 0.55, 0.03, 0.15, -0.02

    def option_reprice(dy):
        """Multi-factor Taylor: sum contributions across risk drivers."""
        pnl = reprice_taylor(
            dy, delta=np.array([delta_opt, vega]),
            gamma=np.array([gamma_opt, 0.0]),
            theta=theta, tau=21 / 252,
        )
        return pnl.sum(axis=1, keepdims=True)

    option_scenarios = project_scenarios(
        option_invariants, investment_horizon=21, n_simulations=2000,
        reprice=option_reprice,
    )
    print(
        f"Option monthly P&L scenarios: "
        f"mean={option_scenarios.mean():.4f}, std={option_scenarios.std():.4f}"
    )

    # --- Combine into multi-asset portfolio ---
    all_scenarios = np.column_stack([stock_scenarios, bond_scenarios, option_scenarios])
    dist = AssetsDistribution(
        scenarios=all_scenarios, asset_names=["Stock", "Bond", "Option"],
    )
    wrapper = PortfolioWrapper.from_scenarios(all_scenarios)
    frontier = wrapper.cvar_frontier(num_portfolios=8, alpha=0.05, seed=42)
    w, ret, risk = frontier.min_risk()
    print(
        f"\nMin-CVaR portfolio: Stock={w.values[0]:.1%}, "
        f"Bond={w.values[1]:.1%}, Option={w.values[2]:.1%}"
    )
    print(f"Expected return: {ret:.4f}, CVaR(95%): {risk:.4f}")


if __name__ == "__main__":
    main()

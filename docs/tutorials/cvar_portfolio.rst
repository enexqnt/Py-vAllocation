Credit Portfolio: CVaR with Scenario Weights
=============================================

This tutorial follows the ``cvar_allocation.py`` example with additional
commentary. We blend historical scenarios with exponentially-decaying weights,
target a minimum CVaR portfolio, and visualise the risk/return envelope.

1. Ingest and weight scenarios
------------------------------

.. code-block:: python

    import numpy as np
    import pandas as pd

    from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper
    from pyvallocation.probabilities import generate_exp_decay_probabilities

    prices = pd.read_csv("examples/ETF_prices.csv", index_col="Date", parse_dates=True)
    weekly_returns = prices.resample("W-FRI").last().pct_change().dropna()

    T = len(weekly_returns)
    exp_probs = generate_exp_decay_probabilities(T, half_life=max(T // 6, 8))

    distribution = AssetsDistribution(
        scenarios=weekly_returns,
        probabilities=exp_probs.reshape(-1, 1),
    )

Step 2 - Optimise for a return target under CVaR
------------------------------------------------

.. code-block:: python

    wrapper = PortfolioWrapper(distribution)
    wrapper.set_constraints({"long_only": True, "total_weight": 1.0})

    weights, exp_return, cvar = wrapper.mean_cvar_portfolio_at_return(
        return_target=0.004,
        alpha=0.05,
    )

    print("CVaR-minimising weights:")
    print(weights.sort_values(ascending=False).round(4))
    print(f"Expected return {exp_return:.2%} | CVaR {cvar:.2%}")

Step 3 - Map the entire frontier
--------------------------------

.. code-block:: python

    frontier = wrapper.mean_cvar_frontier(num_portfolios=9, alpha=0.05)

    import matplotlib.pyplot as plt
    from pyvallocation.plotting import plot_frontiers

    ax = plot_frontiers(frontier, highlight=("min_risk", "max_return"))
    ax.set_title("CVaR Frontier (alpha = 5%)")
    ax.set_xlabel("CVaR")
    ax.set_ylabel("Expected Return")
    ax.figure.tight_layout()

Step 4 - Practical considerations
---------------------------------

- **Parametric fallback:** If you only have ``mu`` and ``cov``, the wrapper
  will simulate scenarios under a multivariate normal assumption. For heavy
  tails, bring your own scenarios.
- **Stress testing:** Run ``wrapper.mean_cvar_portfolio_at_return`` across a grid
  of ``alpha`` values (e.g., 1%, 5%, 10%) to understand tail sensitivity.
- **Transaction costs:** Add ``initial_weights`` and ``proportional_costs`` to
  penalise turnover before calling the CVaR solver.
- **Reporting:** Convert ``weights`` into discrete trades using
  :func:`pyvallocation.discrete_allocation.discretize_weights` to complete the
  trade ticket lifecycle.

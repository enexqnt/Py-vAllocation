Equity/Bond Allocation with Mean-Variance Optimisation
======================================================

This tutorial reproduces the ``Example_01.ipynb`` notebook in a text-friendly
format. We construct a global equity/bond universe, apply shrinkage estimates
for stability, and generate a mean-variance frontier that is ready to ship to a
portfolio committee.

Step 1 – Load market data & scenarios
--------------------------------------

We use the ETF sample bundled with the repository. Returns are computed at a
weekly frequency and reindexed for convenience.

.. code-block:: python

    from pathlib import Path
    import pandas as pd

    data_path = Path("examples/ETF_prices.csv")
    prices = pd.read_csv(data_path, index_col="Date", parse_dates=True).ffill()
    weekly_prices = prices.resample("W-FRI").last().dropna(how="all")
    weekly_returns = weekly_prices.pct_change().dropna()

    # Always keep asset names tidy – they propagate through every output.
    weekly_returns = weekly_returns.rename(columns=lambda c: c.replace(" ", "_"))

Step 2 – Estimate robust moments
--------------------------------

High-dimensional problems need shrinkage. The helper below applies
James–Stein mean shrinkage and OAS covariance shrinkage, preserving the original
labels.

.. code-block:: python

    from pyvallocation.moments import estimate_moments

    mu_js, sigma_oas = estimate_moments(
        weekly_returns,
        mean_estimator="james_stein",
        cov_estimator="oas",
    )

Step 3 – Optimise and generate the frontier
-------------------------------------------

The :class:`pyvallocation.portfolioapi.PortfolioWrapper` handles constraints and
frontier construction. We supply the estimated moments via
:class:`pyvallocation.portfolioapi.AssetsDistribution`.

.. code-block:: python

    from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper

    distribution = AssetsDistribution(mu=mu_js, cov=sigma_oas)
    wrapper = PortfolioWrapper(distribution)
    wrapper.set_constraints({"long_only": True, "total_weight": 1.0})

    mv_frontier = wrapper.mean_variance_frontier(num_portfolios=21)
    tangency_weights, tangency_return, tangency_risk = mv_frontier.get_tangency_portfolio(
        risk_free_rate=0.01
    )

Step 4 – Visualise & interpret the curve
----------------------------------------

.. code-block:: python

    import matplotlib.pyplot as plt
    from pyvallocation.plotting import plot_frontiers

    ax = plot_frontiers(mv_frontier, highlight=("min_risk", "tangency"))
    ax.set_title("Weekly ETF Frontier (shrinkage-adjusted)")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Expected Return")
    ax.figure.tight_layout()

    print("Tangency portfolio weights (top 5):")
    print(tangency_weights.sort_values(ascending=False).head())
    print(f"Expected return {tangency_return:.2%} | Volatility {tangency_risk:.2%}")

5. Tips & next steps
--------------------

- Use :func:`pyvallocation.utils.projection.project_mean_covariance` to map
  weekly statistics to monthly or annual horizons.
- Sensitivity-test your inputs by swapping ``cov_estimator`` to ``"nls"`` or
  ``"poet"`` – no other code changes are required.
- Combine the frontier with :func:`pyvallocation.ensembles.assemble_portfolio_ensemble`
  for multi-model comparisons.
- Export ``mv_frontier.to_frame()`` to capture the entire surface for reporting
  dashboards.

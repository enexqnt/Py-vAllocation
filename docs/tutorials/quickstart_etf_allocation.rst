Comprehensive ETF Allocation Quickstart
=======================================

This end-to-end example mirrors what a practitioner would do when preparing an
allocation memo. We ingest ETF data, estimate moments with multiple shrinkage
models, incorporate a macro view, project the estimates to the investment
horizon, build optimised frontiers, ensemble them, select a final portfolio, and
produce discrete trade weights together with diagnostic plots.

Prerequisites
-------------

.. code-block:: bash

    python -m pip install "py-vallocation[robust]"

The tutorial uses ``pandas`` for data manipulation, ``numpy`` for random seeds,
``matplotlib`` for plots, and the optional shrinkage packages bundled with the
``robust`` extra.

1. Load and preprocess market data
----------------------------------

.. code-block:: python

    from pathlib import Path
    import numpy as np
    import pandas as pd

    ETF_CSV = Path("examples/ETF_prices.csv")
    if not ETF_CSV.exists():
        raise FileNotFoundError("ETF sample data missing; check the repository checkout.")

    prices = pd.read_csv(ETF_CSV, index_col="Date", parse_dates=True).ffill()
    # Weekly aggregation to focus on medium-term allocation decisions
    weekly_prices = prices.resample("W-FRI").last().dropna(how="all")
    weekly_returns = weekly_prices.pct_change().dropna()
    weekly_returns = weekly_returns.rename(columns=lambda c: c.replace(" ", "_"))

    print("Weekly sample:")
    print(weekly_returns.tail())

2. Estimate moments under multiple models
-----------------------------------------

We prepare three complementary moment estimators: shrinkage mean-variance,
robust Tyler covariance, and a Bayesian Black–Litterman posterior that encodes
a macro view ("US equities expected excess return +50bps vs developed ex-US").

.. code-block:: python

    from pyvallocation.moments import (
        estimate_moments,
        posterior_moments_black_litterman,
    )
    from pyvallocation.views import FlexibleViewsProcessor

    # Base shrinkage moments (James–Stein + OAS)
    mu_shrink, sigma_oas = estimate_moments(
        weekly_returns,
        mean_estimator="james_stein",
        cov_estimator="oas",
    )

    # Robust variant (Huber mean + Tyler covariance)
    mu_huber, sigma_tyler = estimate_moments(
        weekly_returns,
        mean_estimator="huber",
        cov_estimator="tyler",
        cov_kwargs={"shrinkage": 0.1},
    )

    # Black–Litterman macro view
    view_processor = FlexibleViewsProcessor(
        prior_mean=mu_shrink,
        prior_cov=sigma_oas,
        mean_views={"Equity_US": 0.015, ("Equity_US", "Equity_Dev_ex_US"): 0.005},
    )
    mu_bl, sigma_bl = view_processor.posterior_mean, view_processor.posterior_cov

3. Project to the investment horizon
------------------------------------

Suppose the committee targets a 1-year horizon. We project the statistics using
a realistic 52-week annualisation and convert log expectations to simple
returns.

.. code-block:: python

    from pyvallocation.utils.projection import project_mean_covariance, log2simple

    def project(mu, sigma, annualisation):
        mu_proj, sigma_proj = project_mean_covariance(mu, sigma, annualization_factor=annualisation)
        return log2simple(mu_proj, sigma_proj)

    horizon_mu = {}
    horizon_sigma = {}
    for label, (mu, sigma) in {
        "Shrinkage": (mu_shrink, sigma_oas),
        "Robust": (mu_huber, sigma_tyler),
        "BL": (mu_bl, sigma_bl),
    }.items():
        horizon_mu[label], horizon_sigma[label] = project(mu, sigma, annualisation=52)

4. Build frontiers and define ensemble specs
--------------------------------------------

We create portfolio specifications that pair each estimator with an optimiser.
Constraints enforce long-only, fully invested allocations.

.. code-block:: python

    from pyvallocation.ensembles import make_portfolio_spec, assemble_portfolio_ensemble
    from pyvallocation.portfolioapi import AssetsDistribution

    long_only = {"long_only": True, "total_weight": 1.0}

    specs = [
        make_portfolio_spec(
            name="Shrinkage_MV",
            returns=weekly_returns,
            mean_estimator="james_stein",
            cov_estimator="oas",
            projection={"annualization_factor": 52, "log_to_simple": True},
            optimiser="mean_variance",
            optimiser_kwargs={"num_portfolios": 21, "constraints": long_only},
            selector="risk_target",
            selector_kwargs={"max_risk": 0.12},
            metadata={"horizon": "1Y", "model": "Shrinkage"},
        ),
        make_portfolio_spec(
            name="Robust_RRP",
            returns=weekly_returns,
            mean_estimator="huber",
            cov_estimator="tyler",
            cov_kwargs={"shrinkage": 0.1},
            projection={"annualization_factor": 52, "log_to_simple": True},
            optimiser="rrp",
            optimiser_kwargs={
                "num_portfolios": 9,
                "max_multiplier": 1.5,
                "lambda_reg": 0.2,
                "constraints": long_only,
            },
            selector="risk_target",
            selector_kwargs={"max_risk": 0.12},
            metadata={"horizon": "1Y", "model": "Robust"},
        ),
        make_portfolio_spec(
            name="BL_MV",
            distribution=AssetsDistribution(mu=horizon_mu["BL"], cov=horizon_sigma["BL"]),
            optimiser="mean_variance",
            optimiser_kwargs={"num_portfolios": 21, "constraints": long_only},
            selector="risk_target",
            selector_kwargs={"max_risk": 0.12},
            metadata={"horizon": "1Y", "model": "Black-Litterman"},
        ),
    ]

    ensemble = assemble_portfolio_ensemble(
        specs,
        ensemble=("average", "stack"),
        stack_folds=3,
    )

    print("Selected portfolios:")
    print(ensemble.selections.round(4))

Step 5 – Plot frontiers and the stacked allocation
--------------------------------------------------

.. code-block:: python

    import matplotlib.pyplot as plt
    from pyvallocation.plotting import plot_frontiers

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, frontier in ensemble.frontiers.items():
        plot_frontiers(frontier, ax=ax, label=name)
    ax.legend()
    ax.set_title("ETF Frontier Comparison – 1Y Horizon")
    ax.figure.tight_layout()

    print("\nStacked allocation (top 5 holdings):")
    print(ensemble.stacked.sort_values(ascending=False).head())
    stacked_weights = ensemble.stacked.reindex(weekly_returns.columns, fill_value=0.0)
    portfolio_weekly = weekly_returns.dot(stacked_weights)
    annualised_return = (1.0 + portfolio_weekly.mean()) ** 52 - 1.0
    annualised_vol = portfolio_weekly.std(ddof=0) * np.sqrt(52)
    sharpe = (annualised_return - 0.01) / annualised_vol if annualised_vol > 0 else float("nan")
    print(
        f"\nStacked trailing metrics: return={annualised_return:.2%}, "
        f"vol={annualised_vol:.2%}, Sharpe≈{sharpe:.2f} (rf=1%)"
    )

Step 6 – Convert to discrete trades
-----------------------------------

Assume $10M AUM with current holdings all cash. We discretise using lot sizes of
1 share and sample prices from the latest weekly close.

.. code-block:: python

    latest_prices = weekly_prices.iloc[-1]
    from pyvallocation.discrete_allocation import discretize_weights

    allocation = discretize_weights(
        weights=ensemble.stacked,
        latest_prices=latest_prices,
        total_value=10_000_000,
    )

    print("\nDiscrete allocation (share counts):")
    shares = pd.Series(allocation.shares, dtype=int).reindex(ensemble.stacked.index, fill_value=0)
    summary = pd.DataFrame(
        {
            "Target Weight": ensemble.stacked.round(4),
            "Achieved Weight": allocation.achieved_weights.reindex(ensemble.stacked.index).round(4),
            "Shares": shares.astype(int),
            "Market Value": (shares * latest_prices).round(2),
        }
    )
    print(summary[summary["Shares"] > 0].sort_values("Market Value", ascending=False))
    print(f"Residual cash: {allocation.leftover_cash:,.2f}")
    print(f"Tracking error (RMSE): {allocation.tracking_error:.6f}")

The greedy allocator automatically detects when it cannot make further progress
with whole lots and transparently falls back to the MILP routine (when SciPy's
``milp`` is available), so even challenging discretisation problems complete
reliably without manual intervention.

Step 7 – Save artefacts for compliance
--------------------------------------

.. code-block:: python

    report = {
        "selected_weights": ensemble.selections,
        "stacked_weights": ensemble.stacked,
        "average_weights": ensemble.average,
        "metadata": ensemble.metadata,
    }
    for key, value in report.items():
        if hasattr(value, "to_csv"):
            value.to_csv(f"output/{key}.csv")

    # Persist diagnostic plots
    ax.figure.savefig("output/frontiers.png", dpi=150)

Step 8 – Optional enhancements
------------------------------

- Feed ``ensemble`` into :func:`pyvallocation.ensembles.exposure_stack_frontiers`
  with additional specs (e.g., CVaR optimiser) for stress comparisons.
- Integrate views with :class:`pyvallocation.views.FlexibleViewsProcessor` for
  scenario re-weighting beyond the simple mean adjustments shown here.
- Attach ESG or liquidity constraints via ``PortfolioWrapper.set_constraints``
  before generating frontiers.
- Monitor turnover by storing ``allocation.turnover`` when rebalancing.

This workflow runs in under a minute on a laptop and mirrors the typical steps a
buy-side allocation team follows from data ingestion to trade-ready output.

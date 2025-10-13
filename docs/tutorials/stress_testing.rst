Stress Testing & PnL Summary
============================

This tutorial complements :file:`examples/stress_and_pnl.py` and focuses on the
new stress-testing and performance-report helpers.

Set up data & tangency portfolio
--------------------------------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from pathlib import Path
   from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper
   from pyvallocation.utils.performance import performance_report

   DATA_PATH = Path("examples/ETF_prices.csv")
   prices = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True).ffill()
   weekly = prices.resample("W-FRI").last().dropna(how="all")
   returns = weekly.pct_change().dropna().rename(columns=lambda c: c.replace(" ", "_"))

   wrapper = PortfolioWrapper(AssetsDistribution(scenarios=np.log1p(returns)))
   wrapper.set_constraints({"long_only": True, "total_weight": 1.0})
   frontier = wrapper.mean_variance_frontier(num_portfolios=21)
   weights, *_ = frontier.get_tangency_portfolio(risk_free_rate=0.01)

   nominal_perf = performance_report(weights, returns.values, alpha=0.95)
   print(nominal_perf.round(4))

Half-life probability tilt
--------------------------

Historical-simulation flexible probabilities emphasise recent scenarios. The
``exp_decay_stress`` helper wraps `generate_exp_decay_probabilities` and feeds
the output into :func:`pyvallocation.stress.stress_test`.

.. code-block:: python

   from pyvallocation.stress import exp_decay_stress

   decay_df = exp_decay_stress(weights, returns.values, half_life=52)
   print(decay_df.loc["portfolio_0", ["return_stress", "ENS_stress"]].round(4))

Gaussian kernel focus
---------------------

To focus on particular regimes (e.g., high volatility), reuse the kernel
probability helper. The stress wrapper constructs ``p*`` with
``generate_gaussian_kernel_probabilities`` and keeps the rest of the pipeline
unchanged.

.. code-block:: python

   from pyvallocation.stress import kernel_focus_stress

   vol_feature = returns["SPY"].rolling(12).std(ddof=0).fillna(method="bfill")
   kernel_df = kernel_focus_stress(
       weights,
       returns.values,
       focus_series=vol_feature.values,
       target=vol_feature.max(),
   )
   print(kernel_df.round(4))

Linear scenario shocks
----------------------

Combine probability tilts with scenario transforms via :func:`linear_map`. The
callable can inject mean shifts, scaling, or matrix mappings before the metrics
are recomputed.

.. code-block:: python

   from pyvallocation.stress import linear_map, stress_test

   vol_up = linear_map(scale=1.25)
   combo_df = stress_test(
       weights,
       returns.values,
       stressed_probabilities=np.linspace(1.0, 2.0, num=returns.shape[0]) / np.linspace(1.0, 2.0, num=returns.shape[0]).sum(),
       transform=vol_up,
   )
   print(combo_df.round(4))

PnL vectors for custom analytics
--------------------------------

If you need raw scenario P&L vectors for bespoke reporting, use
:func:`pyvallocation.utils.performance.scenario_pnl`. Results stay aligned with
pandas indices and can be archived or charted directly.

.. code-block:: python

   from pyvallocation.utils.performance import scenario_pnl

   pnl_series = scenario_pnl(weights, returns)
   pnl_series.plot(title="Scenario P&L")

Next steps
----------

- Feed posterior probabilities from :class:`pyvallocation.views.FlexibleViewsProcessor`
  into :func:`entropy_pooling_stress`.
- Use :func:`performance_report` as a quick numerical summary in notebooks or
  dashboards after running a stress campaign.

Getting Started
===============

*Py-vAllocation* ships with a compact API, extensive docstrings, and
production-oriented tutorials. This page shows how to install the package,
exercise the core wrapper, and run the end-to-end ETF quickstart.

Installation
------------

.. code-block:: bash

   python -m pip install py-vallocation

The base install pulls in ``numpy``, ``pandas``, ``scipy`` and ``cvxopt``. If
you depend on nonlinear shrinkage, POET, or the graphical lasso extras, install
the optional bundle:

.. code-block:: bash

   python -m pip install "py-vallocation[robust]"

Verify the install
------------------

The snippet below ingests scenario data into
:class:`pyvallocation.portfolioapi.AssetsDistribution`, wires it to a
:class:`pyvallocation.portfolioapi.PortfolioWrapper`, and computes a compact
mean-variance frontier.

.. code-block:: python

   import pandas as pd
   from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper

   scenarios = pd.DataFrame(
       {
           "Equity_US": [0.021, -0.013, 0.018, 0.007, 0.011],
           "Equity_EU": [0.017, -0.009, 0.014, 0.004, 0.006],
           "Credit_US": [0.009, 0.008, 0.007, 0.006, 0.008],
           "Govt_Bonds": [0.004, 0.003, 0.005, 0.002, 0.004],
       }
   )

   wrapper = PortfolioWrapper.from_scenarios(scenarios)

   frontier = wrapper.variance_frontier(num_portfolios=5)
   weights, expected_return, risk = frontier.min_risk()

   print(weights.round(4))
   print(f"Expected return: {expected_return:.4%} | Volatility: {risk:.4%}")

Run the ETF quickstart
----------------------

For a full workflow--from data ingestion to trade lots--run the ETF quickstart
script. See the :doc:`tutorials/notebooks/ETF_Multi_Asset_Walkthrough` notebook
for the full walkthrough with output and plots.

.. code-block:: bash

   python examples/quickstart_etf_allocation.py

The tutorial explains each stage, referencing
:func:`pyvallocation.moments.estimate_moments`,
:class:`pyvallocation.views.FlexibleViewsProcessor`,
:func:`pyvallocation.ensembles.assemble_portfolio_ensemble`, and
:func:`pyvallocation.discrete_allocation.discretize_weights`.

From prices (simplest entry point)
-----------------------------------

For users with raw price data, ``from_prices`` computes simple returns and
builds the distribution in one line:

.. code-block:: python

   from pyvallocation import PortfolioWrapper

   wrapper = PortfolioWrapper.from_prices(price_df)
   frontier = wrapper.variance_frontier()
   w, ret, risk = frontier.tangency(risk_free_rate=0.04)

From invariants (Meucci Prayer pipeline)
-----------------------------------------

For log-return invariants (or yield changes, vol changes), ``from_invariants``
chains projection (P3) and repricing (P4) into P&L scenarios for optimisation:

.. code-block:: python

   import numpy as np
   from pyvallocation import PortfolioWrapper

   log_rets = np.log(prices / prices.shift(1)).dropna()
   wrapper = PortfolioWrapper.from_invariants(log_rets, horizon=52, seed=42)
   frontier = wrapper.cvar_frontier(alpha=0.05)

For mixed portfolios (stocks + bonds + options), specify per-instrument repricing:

.. code-block:: python

   from pyvallocation import reprice_exp, reprice_taylor

   wrapper = PortfolioWrapper.from_invariants(
       invariants_df,  # DataFrame: equity log-return, yield change, vol change
       reprice={
           "SPY":  reprice_exp,
           "TLT":  (["yield_10y"], lambda dy: reprice_taylor(dy, delta=-17, gamma=200)),
           "Call": (["equity_lr", "iv_chg"], my_greeks_fn),
       },
       horizon=52, seed=42,
   )

Views with helpers
------------------

Express views in plain terms with ``at_least``, ``at_most``, ``between``:

.. code-block:: python

   from pyvallocation import FlexibleViewsProcessor, at_least, at_most, between

   ep = FlexibleViewsProcessor(
       prior_risk_drivers=invariants_df,
       mean_views={"SPY": at_least(0.05), ("SPY", "TLT"): at_least(0.02)},
       vol_views={"TLT": between(0.08, 0.15)},
       rank_mean=["SPY", "TLT", "GLD"],   # E[SPY] >= E[TLT] >= E[GLD]
   )

   # Pass tilted probabilities into the Prayer pipeline
   wrapper = PortfolioWrapper.from_invariants(
       ep.get_scenarios(),
       p=ep.get_posterior_probabilities(),
       reprice=reprice_exp, horizon=52, seed=42,
   )

Invariant-level stress testing
-------------------------------

Test portfolio resilience to risk-driver shocks (not just P&L shocks):

.. code-block:: python

   from pyvallocation import stress_invariants, at_least

   report = stress_invariants(
       invariants_df, weights=optimal_weights,
       reprice={"SPY": reprice_exp, "TLT": (["yield_10y"], bond_fn)},
       stress_views={"yield_10y": at_least(0.005)},  # yields up 50bp
       horizon=52, seed=42,
   )
   print(report)   # nominal vs stressed: return, vol, VaR, CVaR

Relaxed risk parity
-------------------

Gambeta & Kwon's relaxed risk parity solver is exposed through the same
interface. It solves for the benchmark risk parity weights then sweeps relaxed
targets sized by multiplier ``m``.

.. code-block:: python

   import numpy as np
   from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper

   mu = np.array([0.08, 0.06, 0.05, 0.03])
   cov = np.array(
       [
           [0.090, 0.040, 0.025, 0.010],
           [0.040, 0.070, 0.020, 0.015],
           [0.025, 0.020, 0.060, 0.018],
           [0.010, 0.015, 0.018, 0.045],
       ]
   )

   wrapper = PortfolioWrapper.from_moments(mu, cov)

   frontier = wrapper.relaxed_risk_parity_frontier(
       num_portfolios=4,
       max_multiplier=1.5,
       lambda_reg=0.25,
   )
   print(frontier.to_frame().round(4))

Metadata attached to ``frontier.metadata`` documents the multiplier, target,
and objective value per point--ideal inputs for dashboarding or plotting.

Where to go next
----------------

- Browse :doc:`tutorials/index` for narrative walkthroughs.
- Run additional examples from :mod:`pyvallocation.examples`:

  * ``mean_variance_frontier.py`` - classical efficient frontier summary.
  * ``cvar_allocation.py`` - minimum-CVaR portfolio and tangency allocation.
  * ``robust_frontier.py`` - Meucci-style robust tau-frontier.
  * ``relaxed_risk_parity_frontier.py`` - relaxed risk parity diagnostics.
  * ``discrete_allocation.py`` - map continuous weights to share counts.
  * ``portfolio_ensembles.py`` - blend multiple frontiers into a single allocation.
  * ``stress_and_pnl.py`` - probability tilts, linear shocks, and performance summaries.

- Dive into the API reference starting at :doc:`pyvallocation.portfolioapi`;
  docstrings are synchronised with the published documentation.

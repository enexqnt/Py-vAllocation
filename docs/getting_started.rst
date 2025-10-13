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

   dist = AssetsDistribution(scenarios=scenarios)
   wrapper = PortfolioWrapper(dist)
   wrapper.set_constraints({"long_only": True, "total_weight": 1.0})

   frontier = wrapper.mean_variance_frontier(num_portfolios=5)
   weights, expected_return, risk = frontier.get_min_risk_portfolio()

   print(weights.round(4))
   print(f"Expected return: {expected_return:.4%} | Volatility: {risk:.4%}")

Run the ETF quickstart
----------------------

For a full workflow--from data ingestion to trade lots--run the ETF quickstart
script. It mirrors :doc:`tutorials/quickstart_etf_allocation` and writes CSVs
and plots to ``output/``.

.. code-block:: bash

   python examples/quickstart_etf_allocation.py

The tutorial explains each stage, referencing
:func:`pyvallocation.moments.estimate_moments`,
:class:`pyvallocation.views.FlexibleViewsProcessor`,
:func:`pyvallocation.ensembles.assemble_portfolio_ensemble`, and
:func:`pyvallocation.discrete_allocation.discretize_weights`.

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

   dist = AssetsDistribution(mu=mu, cov=cov)
   wrapper = PortfolioWrapper(dist)
   wrapper.set_constraints({"long_only": True, "total_weight": 1.0})

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

- Dive into the API reference starting at :doc:`pyvallocation.portfolioapi`;
  docstrings are synchronised with the published documentation.

Welcome to Py-vAllocation
=========================

*Py-vAllocation* is a modular toolkit for single-period portfolio construction.
It exposes a consistent Python API for mean-variance, mean-CVaR and robust
optimisation, integrates view-based Bayesian updates, and ships with portfolio
ensembling utilities so research portfolios can become investable allocations.

Key capabilities
----------------

- **Multiple risk models, one wrapper** - switch between mean-variance,
  CVaR, and robust formulations without changing how you declare constraints or
  transaction costs.
- **Investor views made actionable** - entropy pooling and Black-Litterman
  helpers transform qualitative views into posterior distributions.
- **Scenario-aware statistics** - shrinkage estimators, Bayesian updates and
  scenario bootstrapping reduce estimation error out of the box.
- **Portfolio ensembling** - average exposures, stack frontiers, and discretise
  weights to bridge the gap between optimisation outputs and tradeable lists.
- **Stress testing made easy** - probability tilts, linear shocks, and PnL summaries
  reuse the same scenario machinery to keep analyses consistent.

Where to start
--------------

- :doc:`getting_started` for installation and a minimal efficient frontier
  example.
- :doc:`tutorials/quickstart_etf_allocation` for the ETF allocation walk-through
  that mirrors the runnable quickstart script.
- :doc:`tutorials/index` for guided walkthroughs of the main workflows
  (mean-variance, CVaR, stress testing, and portfolio ensembling).
- :doc:`tutorials/examples_overview` for a catalog of runnable scripts and notebooks.
- The `examples/` directory offers runnable scripts that mirror the tutorials -
  try ``python examples/stress_and_pnl.py`` for probability tilts and performance summaries.

Install & quickstart
--------------------

.. code-block:: bash

   # Clone and install the library (full instructions in :doc:`getting_started`)
   git clone https://github.com/enexqnt/py-vallocation.git
   cd py-vallocation
   python -m pip install -e .[robust]

.. code-block:: python

   import numpy as np
   from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper

   wrapper = PortfolioWrapper(AssetsDistribution(scenarios=np.random.normal(0, 0.01, size=(252, 4))))
   wrapper.set_constraints({"long_only": True, "total_weight": 1.0})
   frontier = wrapper.mean_variance_frontier(num_portfolios=11)
   weights, ret, risk = frontier.get_tangency_portfolio(risk_free_rate=0.01)
   print(weights.round(3))

.. toctree::
   :maxdepth: 1
   :caption: Tutorials & Examples

   getting_started
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   pyvallocation
   pyvallocation.portfolioapi
   pyvallocation.optimization
   pyvallocation.views
   pyvallocation.bayesian
   pyvallocation.probabilities
   pyvallocation.moments
   pyvallocation.ensembles
   pyvallocation.utils

.. toctree::
   :maxdepth: 1
   :caption: Background

   bibliography
   releases

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

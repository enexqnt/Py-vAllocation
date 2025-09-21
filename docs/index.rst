Welcome to Py-vAllocation
=========================

*Py-vAllocation* is a modular toolkit for single-period portfolio construction.
It exposes a consistent Python API for mean-variance, mean-CVaR and robust
optimisation, integrates view-based Bayesian updates, and ships with portfolio
ensembling utilities so research portfolios can become investable allocations.

Key capabilities
----------------

- **Multiple risk models, one wrapper** – switch between mean-variance,
  CVaR, and robust formulations without changing how you declare constraints or
  transaction costs.
- **Investor views made actionable** – entropy pooling and Black–Litterman
  helpers transform qualitative views into posterior distributions.
- **Scenario-aware statistics** – shrinkage estimators, Bayesian updates and
  scenario bootstrapping reduce estimation error out of the box.
- **Portfolio ensembling** – average exposures, stack frontiers, and discretise
  weights to bridge the gap between optimisation outputs and tradeable lists.

Where to start
--------------

- :doc:`getting_started` for installation and a minimal efficient frontier
  example.
- :doc:`tutorials/index` for guided walkthroughs of the main workflows
  (mean-variance, CVaR, and portfolio ensembling).
- The `examples/` directory offers runnable scripts that mirror the tutorials –
  try ``python examples/portfolio_ensembles.py`` for a quick demonstration.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

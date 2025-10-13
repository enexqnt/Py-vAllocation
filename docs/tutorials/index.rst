Hands-On Tutorials
==================

The notebooks bundled with the project have been distilled into narrative,
production-focused walkthroughs. Each tutorial starts from raw or lightly
processed market data, shows how to estimate statistical moments, and finishes
with optimisation and reporting steps that mirror real-world workflows.

Pick the storyline that best matches your use case:

.. toctree::
   :maxdepth: 2

   quickstart_etf_allocation
   mean_variance_frontier
   cvar_portfolio
   relaxed_risk_parity
   portfolio_ensembles
   stress_testing

Examples & notebooks
--------------------

For a catalog of runnable scripts and rendered notebooks, see:

.. toctree::
   :maxdepth: 1

   examples_overview

Tips for working through the tutorials
--------------------------------------

- **Keep notebooks handy.** The ``examples/`` scripts and notebooks share the
  same logic and often include richer visualisations.
- **Reuse helper APIs.** Functions like
  :func:`pyvallocation.moments.estimate_moments` and
  :func:`pyvallocation.ensembles.assemble_portfolio_ensemble` are designed to
  snap into larger research pipelines without additional plumbing.
- **Bring your own data.** All tutorials operate on pandas objects. Swap the
  CSV loader with your own data source and the rest of the workflow will remain
  unchanged.

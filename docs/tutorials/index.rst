Tutorials
=========

Interactive Jupyter notebooks with live output, tables, and plots.
Each tutorial starts from raw market data and walks through a complete
portfolio construction workflow.

.. toctree::
   :maxdepth: 1

   notebooks/ETF_Multi_Asset_Walkthrough
   notebooks/Stress_Testing_Comprehensive
   notebooks/Mean_Variance
   notebooks/CVaR_Frontier
   notebooks/Budget_Risk_Parity
   notebooks/Group_Constraints
   notebooks/Repricing_Derivatives
   notebooks/Portfolio_Ensembles
   notebooks/Stress_Testing
   notebooks/Bayesian
   notebooks/Flexible_Views
   notebooks/Simple_views_on_mean
   notebooks/Example_01

Runnable scripts
----------------

The ``examples/`` directory contains standalone Python scripts.

.. toctree::
   :maxdepth: 1

   examples_overview

Tips
----

- **Bring your own data.** All tutorials operate on pandas objects. Swap the
  CSV loader with your own data source and the rest of the workflow remains
  unchanged.
- **Reuse helper APIs.** Functions like
  :func:`pyvallocation.moments.estimate_moments` and
  :func:`pyvallocation.ensembles.assemble_portfolio_ensemble` snap into
  larger research pipelines without additional plumbing.

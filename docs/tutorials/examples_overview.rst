Examples & Notebooks
====================

This page lists the runnable examples and companion notebooks. Each example is
designed to be short, reproducible, and easy to adapt to your workflow.

How to run
----------

.. code-block:: bash

   # from the repository root
   python -m pip install -e .[robust]
   python examples/quickstart_etf_allocation.py

Quickstarts
-----------

- :file:`examples/quickstart_etf_allocation.py` - end-to-end ETF allocation
  (moments -> frontiers -> ensemble -> discrete trades). Outputs plots and CSVs
  under :file:`output/` and is documented in :doc:`quickstart_etf_allocation`.

Frontiers & risk models
-----------------------

- :file:`examples/mean_variance_frontier.py` - classical mean-variance frontier.
- :file:`examples/cvar_allocation.py` - CVaR frontier (:math:`\alpha = 5\%`).
- :file:`examples/robust_frontier.py` - robust (Meucci) :math:`\lambda`-frontier.
- :file:`examples/relaxed_risk_parity_frontier.py` - relaxed risk parity diagnostic sweep.

Utilities
---------

- :file:`examples/discrete_allocation.py` - convert continuous weights into
  lot-sized share counts.
- :file:`examples/portfolio_ensembles.py` - blend multiple model selections via
  stacking/averaging to a single allocation.
- :file:`examples/stress_and_pnl.py` - probability tilts, linear shocks, and performance summaries.

Notebooks
---------

Rendered notebooks live under :file:`docs/tutorials/notebooks/`:

- :file:`tutorials/notebooks/Example_01.ipynb`
- :file:`tutorials/notebooks/Simple_views_on_mean.ipynb`
- :file:`tutorials/notebooks/Bayesian.ipynb`
- :file:`tutorials/notebooks/Flexible_Views.ipynb`

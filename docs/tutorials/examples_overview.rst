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
  under :file:`output/`.

Frontiers & risk models
-----------------------

- :file:`examples/mean_variance_frontier.py` - classical mean-variance frontier.
- :file:`examples/cvar_allocation.py` - CVaR frontier (:math:`\alpha = 5\%`).
- :file:`examples/robust_frontier.py` - robust (Meucci) :math:`\lambda`-frontier.
- :file:`examples/relaxed_risk_parity_frontier.py` - relaxed risk parity diagnostic sweep.
- :file:`examples/budget_risk_parity.py` - custom risk budgets (50/20/20/10) vs. ERC.
- :file:`examples/group_constraints.py` - sector allocation limits with typed ``Constraints``.
- :file:`examples/repricing_derivatives.py` - stocks + bonds + options via the Prayer repricing chain.

Utilities
---------

- :file:`examples/discrete_allocation.py` - convert continuous weights into
  lot-sized share counts.
- :file:`examples/portfolio_ensembles.py` - blend multiple model selections via
  stacking/averaging to a single allocation.
- :file:`examples/stress_and_pnl.py` - probability tilts, linear shocks, and performance summaries.

Notebooks (with output and plots)
----------------------------------

.. toctree::
   :maxdepth: 1

   notebooks/Example_01
   notebooks/Simple_views_on_mean
   notebooks/Bayesian
   notebooks/Flexible_Views
   notebooks/Mean_Variance
   notebooks/CVaR_Frontier
   notebooks/Group_Constraints
   notebooks/Budget_Risk_Parity
   notebooks/Repricing_Derivatives
   notebooks/Stress_Testing
   notebooks/Portfolio_Ensembles
   notebooks/ETF_Multi_Asset_Walkthrough

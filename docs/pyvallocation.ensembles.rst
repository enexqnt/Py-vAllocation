Portfolio Ensembling
====================

Ensembling utilities in :mod:`pyvallocation.ensembles` blend multiple model
outputs into tradeable allocations. They are designed to accept either raw
NumPy arrays or :class:`~pyvallocation.portfolioapi.PortfolioFrontier`
instances, so research portfolios can be piped into production workflows with
minimal glue code.

.. important::

   The exposure stacking routines adapt the GPL-3 implementation released by
   the `fortitudo.tech <https://github.com/fortitudo-tech/fortitudo.tech>`_
   project. If you build on top of them, please keep the original attribution in
   downstream documentation or research notes.

At a glance
-----------

- :func:`pyvallocation.ensembles.average_exposures` averages a stack of sample
  portfolios (uniformly or using custom weights).
- :func:`pyvallocation.ensembles.exposure_stacking` solves the quadratic
  programme popularised by Vorobets :cite:p:`vorobets2024derivatives` to
  concentrate risk on common factors while cancelling idiosyncratic bets.
  This implementation adapts the GPL-3 licensed routine released in the
  `fortitudo.tech <https://github.com/fortitudo-tech/fortitudo.tech>`_
  project and is credited accordingly.
- :func:`pyvallocation.ensembles.assemble_portfolio_ensemble` orchestrates
  frontier sampling, averaging, stacking, and optional selectors. The function
  underpins the :doc:`tutorials/portfolio_ensembles` walkthrough.

Quick start
-----------

The helpers operate on column-organised samples. Start with simple NumPy arrays
to get a feel for the APIs:

.. code-block:: python

   import numpy as np
   from pyvallocation.ensembles import average_exposures, exposure_stacking

   # Two sample portfolios across two assets
   samples = np.array([[0.6, 0.3],
                       [0.4, 0.7]])

   avg = average_exposures(samples)
   stacked = exposure_stacking(samples, L=2)

   print(avg)      # -> [0.45 0.55]
   print(stacked)  # -> exposure stacking output with damped idiosyncratic bets

Workflow summary
----------------

End-to-end ensemble construction typically follows these steps:

1. **Generate candidate portfolios.** Optimise frontiers or run bespoke models
   to obtain a set of column-organised sample weights.
2. **Select representatives.** Use methods such as
   :meth:`pyvallocation.portfolioapi.PortfolioFrontier.portfolio_at_risk_target`
   or :func:`pyvallocation.ensembles.make_portfolio_spec` to standardise inputs.
3. **Blend exposures.** Apply :func:`average_exposures` for a linear average or
   :func:`exposure_stacking` to damp idiosyncratic bets while keeping the common
   factor structure.
4. **Report and trade.** The outputs are pandas-aware vectors, so they can be
   fed straight into stress testing, discrete allocation, or attribution.

When you work with pre-built frontiers the API stays consistent:

.. code-block:: python

   from pyvallocation.portfolioapi import PortfolioWrapper, AssetsDistribution
   from pyvallocation.ensembles import average_frontiers, exposure_stack_frontiers

   wrapper = PortfolioWrapper(AssetsDistribution(scenarios=returns))
   frontier = wrapper.mean_variance_frontier(num_portfolios=21)
   another = wrapper.mean_cvar_frontier(num_portfolios=21)

   avg_portfolio = average_frontiers([frontier, another])
   stacked_portfolio = exposure_stack_frontiers([frontier, another], L=3)

   avg_portfolio.plot.bar(title="Average ensemble weights")

Tips
----

- The stacking depth ``L`` controls how tightly exposures are shrunk. Larger
  values yield smoother allocations but require more sample portfolios (``L``
  cannot exceed the number of samples).
- Exposure stacking assumes portfolios are long-only and sum to one. If your
  research stack permits leverage, normalise samples first.
- :func:`assemble_portfolio_ensemble` can mix averaging and stacking in a single
  call. See :doc:`tutorials/portfolio_ensembles` for an end-to-end example.
- Every helper preserves pandas indices when they are present so the output can
  flow straight into downstream reporting.
- Solver options can be forwarded via ``solver_options`` when you need to tweak
  CVXOPT tolerances or iteration limits.

Troubleshooting
---------------

- **Shape mismatches.** Ensure inputs broadcast to ``(n_assets, n_samples)``.
  Use ``DataFrame.T`` or ``np.column_stack`` to align your sample set.
- **Missing labels.** When averaging/stacking Series with different indices the
  helpers will reindex and raise on missing entries—double-check asset names.
- **Solver errors.** Exposure stacking relies on CVXOPT. Pass
  ``solver_options={'feastol': 1e-7}`` (or similar) for noisy inputs, and verify
  that no column contains NaNs or violates the long-only assumption.

Reference
---------

.. automodule:: pyvallocation.ensembles
   :members:
   :undoc-members:
   :show-inheritance:

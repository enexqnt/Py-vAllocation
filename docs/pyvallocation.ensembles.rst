Portfolio Ensembling
====================

Ensembling utilities in :mod:`pyvallocation.ensembles` blend model outputs into
tradeable allocations. Core helpers include:

- :func:`pyvallocation.ensembles.average_exposures` - arithmetic averaging of
  portfolio weights (uniform or custom weights).
- :func:`pyvallocation.ensembles.exposure_stacking` - quadratic programme that
  stacks exposures while damping idiosyncratic bets.
- :func:`pyvallocation.ensembles.assemble_portfolio_ensemble` - orchestrates
  optimiser runs, applies selectors, and returns the combined result used
  throughout the tutorials.

.. automodule:: pyvallocation.ensembles
   :members:
   :undoc-members:
   :show-inheritance:

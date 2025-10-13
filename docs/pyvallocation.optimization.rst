Portfolio Optimization
======================

Convex optimisation wrappers live in :mod:`pyvallocation.optimization`. They
share a thin interface so frontiers can be swapped without changing constraint
plumbing:

- :class:`pyvallocation.optimization.MeanVariance` - quadratic risk/return
  trade-offs with optional quadratic turnover costs.
- :class:`pyvallocation.optimization.MeanCVaR` - linear-program CVaR frontiers
  with proportional costs.
- :class:`pyvallocation.optimization.RelaxedRiskParity` - Gambeta & Kwon's
  relaxed risk parity SOCP implementation.
- :class:`pyvallocation.optimization.RobustOptimizer` - Meucci-style robust
  optimiser with chance-constraint and penalised variants.

.. automodule:: pyvallocation.optimization
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Related Modules
---------------

- :doc:`pyvallocation.portfolioapi`: 
- :doc:`pyvallocation.views`: 
- :doc:`pyvallocation.moments`: 

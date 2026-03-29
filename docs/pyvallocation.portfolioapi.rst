Portfolio API
=============

The :mod:`pyvallocation.portfolioapi` module provides the public entry points
used across the tutorials and examples:

- :class:`pyvallocation.portfolioapi.AssetsDistribution` stores parametric or
  scenario-based return descriptions while preserving labels.
- :class:`pyvallocation.portfolioapi.PortfolioWrapper` exposes efficient
  frontiers (mean-variance, CVaR, relaxed risk parity, robust optimisation) with
  a uniform interface for constraints, turnover costs, and selectors.
- :class:`pyvallocation.portfolioapi.PortfolioFrontier` encapsulates solved
  frontiers and offers convenience selectors such as
  :meth:`~pyvallocation.portfolioapi.PortfolioFrontier.tangency`
  and :meth:`~pyvallocation.portfolioapi.PortfolioFrontier.at_risk`.

All helpers preserve pandas alignment and cooperate with the ensembling and
discrete allocation utilities.

.. automodule:: pyvallocation.portfolioapi
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:




- :doc:`pyvallocation.optimization`: For advanced portfolio optimization
- :doc:`pyvallocation.views`: For applying investment views
- :doc:`pyvallocation.moments`: For statistical analysis

Portfolio Views
================

Investment view tooling resides in :mod:`pyvallocation.views`. It covers both
Bayesian updates and entropy-based scenario reweighting:

- :class:`pyvallocation.views.BlackLittermanProcessor` - classical
  Black-Litterman equality views with Idzorek confidences.
- :class:`pyvallocation.views.FlexibleViewsProcessor` - entropy pooling with
  inequalities on means, volatilities, correlations, and higher moments.
- :func:`pyvallocation.views.entropy_pooling` - low-level solver returning
  reweighted scenario probabilities.

.. automodule:: pyvallocation.views
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:


Related Modules
---------------

- :doc:`pyvallocation.optimization`: For portfolio optimization
- :doc:`pyvallocation.portfolioapi`: For portfolio management
- :doc:`pyvallocation.bayesian`: For probabilistic view modeling

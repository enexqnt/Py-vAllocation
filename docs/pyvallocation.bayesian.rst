.. _bayesian:

pyvallocation.bayesian module
=============================

This module provides functionalities related to Bayesian statistics, particularly
for handling Normal–Inverse–Wishart (NIW) distributions. It includes tools for
robust Cholesky decomposition, chi-square quantile computation, and managing
NIW posterior parameters for asset allocation.

.. automodule:: pyvallocation.bayesian
   :show-inheritance:
   :undoc-members:

Functions
---------

.. autofunction:: _cholesky_pd

.. autofunction:: chi2_quantile

Classes
-------

.. autoclass:: NIWParams
   :members:
   :show-inheritance:

.. autoclass:: NIWPosterior
   :members:
   :show-inheritance:

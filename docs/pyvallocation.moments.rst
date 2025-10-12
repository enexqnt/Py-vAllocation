Statistical Moments Module
==========================

Robust, shrinkage-aware estimators for the first two moments of asset returns
reside in :mod:`pyvallocation.moments`. The functions below preserve pandas
indices when provided and are designed to mix seamlessly with the portfolio
API, Bayesian views, and optimisation engines. All algorithms are extensively
unit-tested (see ``tests/test_moments_shrinkage.py``) to guarantee robustness,
PSD outputs, and consistent labelling.

Quick Start
-----------

.. code-block:: python
   :caption: Moment estimation pipeline

   import pandas as pd
   from pyvallocation.moments import (
       estimate_moments,
       shrink_covariance_nls,
       shrink_mean_james_stein,
   )

   returns = pd.read_csv("returns.csv", index_col=0, parse_dates=True)
   mu_js, sigma_nls = estimate_moments(
       returns,
       mean_estimator="james_stein",
       cov_estimator="nls",
   )

   # Alternatively call estimators directly
   sigma_nls_direct = shrink_covariance_nls(returns)
   mu_js_direct = shrink_mean_james_stein(
       returns.mean(),
       sigma_nls_direct,
       T=len(returns),
   )

Mean Estimators
---------------

.. currentmodule:: pyvallocation.moments

.. autofunction:: estimate_sample_moments
.. autofunction:: shrink_mean_jorion
.. autofunction:: shrink_mean_james_stein
.. autofunction:: robust_mean_huber
.. autofunction:: robust_mean_median_of_means

Covariance Estimators
---------------------

.. autofunction:: shrink_covariance_ledoit_wolf
.. autofunction:: shrink_covariance_oas
.. autofunction:: shrink_covariance_nls
.. autofunction:: factor_covariance_poet
.. autofunction:: robust_covariance_tyler
.. autofunction:: sparse_precision_glasso

   Implements an ADMM-based solver with cross-validated penalty selection.

Bayesian Posterior Adapters
---------------------------

.. autofunction:: posterior_moments_black_litterman
.. autofunction:: posterior_moments_niw

Composite Factory
-----------------

.. autofunction:: estimate_moments

Related Modules
---------------

- :doc:`pyvallocation.probabilities`: Probabilities and scenario weights.
- :doc:`pyvallocation.portfolioapi`: Portfolio wrappers consuming the estimators.
- :doc:`pyvallocation.optimization`: Optimisers making use of the moments.

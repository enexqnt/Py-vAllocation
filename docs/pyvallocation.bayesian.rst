pyvallocation.bayesian module
=============================

Bayesian utilities complement the shrinkage estimators by providing posterior
updates for both means and covariances. Highlights include:

- :func:`pyvallocation.bayesian.posterior_moments_black_litterman` - closed-form
  Black-Litterman posterior moments given pick matrices and confidences.
- :class:`pyvallocation.bayesian.NormalInverseWishartPosterior` - container for
  Normal-Inverse-Wishart updates used in the quickstart.
- Quantile wrappers for chi-square distributions used when sizing Bayesian
  uncertainty sets.

.. automodule:: pyvallocation.bayesian
   :members:
   :show-inheritance:

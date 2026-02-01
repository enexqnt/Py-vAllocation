pyvallocation.bayesian module
=============================

Bayesian utilities complement the shrinkage estimators by providing posterior
updates for both means and covariances. Highlights include:

- :class:`pyvallocation.bayesian.NIWPosterior` - Normal-Inverse-Wishart updates
  used across the examples and tests.
- :class:`pyvallocation.bayesian.RobustBayesPosterior` - convenient wrapper that
  exposes mean-uncertainty covariances for robust optimisation.
- Quantile wrappers for chi-square distributions used when sizing Bayesian
  uncertainty sets.

Robust-Bayesian uncertainty
---------------------------

The NIW posterior implies a closed-form covariance of the mean :math:`S_{\\mu}`
(see :cite:p:`meucci2005robust`):

.. math::

   S_{\\mu} = \\frac{\\nu_1}{T_1 (\\nu_1 - 2)} \\Sigma_1.

Use :class:`pyvallocation.bayesian.RobustBayesPosterior` to access ``S_mu`` and
its horizon-scaled variants (log or simple returns) for robust optimisation.

.. automodule:: pyvallocation.bayesian
   :members:
   :show-inheritance:

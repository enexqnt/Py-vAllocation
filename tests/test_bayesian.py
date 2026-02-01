import numpy as np
import pytest

from pyvallocation.bayesian import _cholesky_pd, RobustBayesPosterior
from pyvallocation.moments import posterior_moments_niw_with_uncertainty


def test_cholesky_pd_escalates_jitter_until_positive_definite():
    nearly_singular = np.array(
        [[0.09, 0.12], [0.12, 0.16 - 5e-5]],
        dtype=float,
    )

    with pytest.warns(RuntimeWarning):
        chol = _cholesky_pd(nearly_singular, jitter=1e-8)

    reconstructed = chol @ chol.T
    eigvals = np.linalg.eigvalsh(reconstructed)

    assert np.all(eigvals > 0)
    assert np.allclose(chol, np.tril(chol))


def test_robust_bayes_posterior_uncertainty_helpers():
    prior_mu = np.array([0.01, 0.02])
    prior_sigma = np.array([[0.10, 0.02], [0.02, 0.08]])
    sample_mu = np.array([0.012, 0.018])
    sample_sigma = np.array([[0.09, 0.01], [0.01, 0.07]])

    rb = RobustBayesPosterior.from_niw(
        prior_mu=prior_mu,
        prior_sigma=prior_sigma,
        t0=5,
        nu0=5,
        sample_mu=sample_mu,
        sample_sigma=sample_sigma,
        n_obs=120,
    )
    cov_simple = rb.mean_uncertainty_cov_simple(annualization_factor=12)
    assert cov_simple.shape == (2, 2)

    mu_ce, sigma_ce, s_mu = posterior_moments_niw_with_uncertainty(
        prior_mu=prior_mu,
        prior_sigma=prior_sigma,
        t0=5,
        nu0=5,
        sample_mu=sample_mu,
        sample_sigma=sample_sigma,
        n_obs=120,
    )
    assert np.asarray(mu_ce).shape == (2,)
    assert np.asarray(sigma_ce).shape == (2, 2)
    assert np.asarray(s_mu).shape == (2, 2)

import pytest
import numpy as np
from pyvallocation.views import entropy_pooling, _dual_objective, FlexibleViewsProcessor, BlackLittermanProcessor

def test_entropy_pooling_no_constraints():
    # Uniform prior, no additional constraints: posterior equals prior
    p = np.array([[0.2], [0.3], [0.5]])
    A = np.ones((1, 3))
    b = np.array([[1.0]])
    q = entropy_pooling(p, A, b)
    assert q.shape == p.shape
    np.testing.assert_allclose(q, p, atol=1e-6)

def test_entropy_pooling_invalid_method():
    p = np.array([[0.5], [0.5]])
    A = np.ones((1, 2))
    b = np.array([[1.0]])
    with pytest.raises(ValueError):
        entropy_pooling(p, A, b, method="ABC")

def test_flexible_views_processor_no_views():
    # No views: posterior probabilities equal uniform prior
    R = np.array([[0.1, 0.2], [0.2, 0.1], [0.15, 0.15]])
    p0 = np.ones((3, 1)) / 3
    proc = FlexibleViewsProcessor(prior_returns=R, prior_probabilities=p0)
    q = proc.get_posterior_probabilities()
    assert q.shape == p0.shape
    np.testing.assert_allclose(q, p0, atol=1e-6)
    mu_post, cov_post = proc.get_posterior()
    # Posterior moments equal sample moments
    expected_mu = (R.T @ p0).flatten()
    expected_cov = ((R - expected_mu) * p0).T @ (R - expected_mu)
    np.testing.assert_allclose(mu_post, expected_mu, atol=1e-6)
    np.testing.assert_allclose(cov_post, expected_cov, atol=1e-6)

def test_blacklitterman_prior_mean_only():
    # No views: posterior equals prior
    mu = np.array([0.01, 0.02])
    cov = np.array([[0.1, 0.01], [0.01, 0.2]])
    bl = BlackLittermanProcessor(prior_mean=mu, prior_cov=cov)
    mu_post, cov_post = bl.get_posterior()
    np.testing.assert_allclose(mu_post, mu, atol=1e-8)
    np.testing.assert_allclose(cov_post, cov, atol=1e-8)

def test_blacklitterman_market_weights():
    # Derive pi from market_weights
    cov = np.eye(2)
    w = np.array([0.5, 0.5])
    bl = BlackLittermanProcessor(prior_cov=cov, market_weights=w, risk_aversion=2.0)
    # π = δ Σ w = 2 * I * w = [1,1]
    expected_pi = np.array([1.0, 1.0])
    # internal pi stored as column vector
    mu_post, _ = bl.get_posterior()
    np.testing.assert_allclose(mu_post, expected_pi, atol=1e-8)

def test_parse_view_and_asset_idx():
    from pyvallocation.views import FlexibleViewsProcessor
    # initialize with prior_mean and prior_cov to avoid scenario dimension mismatch
    prior_mean = np.array([0.1, 0.2])
    prior_cov = np.eye(2)
    proc = FlexibleViewsProcessor(prior_mean=prior_mean, prior_cov=prior_cov)
    # _parse_view tests
    op, tgt = proc._parse_view(5.0)
    assert op == "==" and tgt == 5.0
    op, tgt = proc._parse_view((">=", 3.0))
    assert op == ">=" and tgt == 3.0
    # _asset_idx with numeric and string key
    assert proc._asset_idx(1) == 1
    with pytest.raises(ValueError):
        proc._asset_idx("X")
def test_blacklitterman_mean_views_vector():
    mu = np.array([0.01, 0.02])
    cov = np.array([[0.1, 0.0], [0.0, 0.1]])
    views_vec = np.array([0.03, 0.04])
    bl_vec = BlackLittermanProcessor(prior_mean=mu, prior_cov=cov, mean_views=views_vec)
    bl_dict = BlackLittermanProcessor(prior_mean=mu, prior_cov=cov, mean_views={0: 0.03, 1: 0.04})
    mu_vec, cov_vec = bl_vec.get_posterior()
    mu_dict, cov_dict = bl_dict.get_posterior()
    np.testing.assert_allclose(mu_vec, mu_dict, atol=1e-8)
    np.testing.assert_allclose(cov_vec, cov_dict, atol=1e-8)

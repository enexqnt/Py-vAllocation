import pytest
import numpy as np
import pandas as pd
from pyvallocation import moments

def test_estimate_sample_moments_basic():
    R = np.array([[0.01, 0.02],
                  [0.03, 0.04],
                  [0.05, 0.06]])
    p = np.array([1/3, 1/3, 1/3])
    mu, S = moments.estimate_sample_moments(R, p)
    expected_mu = np.mean(R, axis=0)
    np.testing.assert_allclose(mu, expected_mu)
    X = R - expected_mu
    expected_S = (X.T @ X) / 3
    np.testing.assert_allclose(S, expected_S, atol=1e-10)

def test_estimate_sample_moments_invalid_weights():
    R = np.random.randn(5, 2)
    p = np.array([0.5, 0.4, 0.05, 0, 0])  # sum != 1
    with pytest.raises(ValueError):
        moments.estimate_sample_moments(R, p)

def test_shrink_mean_jorion_invalid():
    mu = np.array([0.01, 0.02])
    S = np.array([[0.1, 0.02],
                  [0.02, 0.08]])
    with pytest.raises(ValueError):
        moments.shrink_mean_jorion(mu, S, 5)

def test_shrink_mean_jorion_basic_n3():
    # Basic shrink for N=3
    mu = np.array([0.01, 0.02, 0.03])
    S = np.eye(3) * 0.05
    T = 10
    shrunk = moments.shrink_mean_jorion(mu, S, T)
    assert shrunk.shape == mu.shape
    assert np.all(shrunk >= mu.min())
    assert np.all(shrunk <= mu.max())

def test_shrink_covariance_ledoit_wolf_identity():
    R = np.random.randn(10, 3)
    S_hat = np.cov(R, rowvar=False)
    Sigma = moments.shrink_covariance_ledoit_wolf(R, S_hat, target="identity")
    np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(Sigma)
    assert np.all(eigvals >= -1e-8)

def test_shrink_covariance_ledoit_wolf_constant_correlation():
    R = np.random.randn(10, 3)
    S_hat = np.cov(R, rowvar=False)
    Sigma = moments.shrink_covariance_ledoit_wolf(R, S_hat, target="constant_correlation")
    np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(Sigma)
    assert np.all(eigvals >= -1e-8)

def test_shrink_covariance_ledoit_wolf_invalid_target():
    R = np.random.randn(10, 3)
    S_hat = np.cov(R, rowvar=False)
    with pytest.raises(ValueError):
        moments.shrink_covariance_ledoit_wolf(R, S_hat, target="unsupported")

def test_estimate_sample_moments_with_pandas():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    p = pd.Series([0.2, 0.3, 0.5], index=df.index)
    mu, S = moments.estimate_sample_moments(df, p)
    assert isinstance(mu, pd.Series)
    assert list(mu.index) == list(df.columns)
    assert isinstance(S, pd.DataFrame)
    assert list(S.index) == list(df.columns)
    expected_mu = df.values.T @ p.values
    np.testing.assert_allclose(mu.values, expected_mu, atol=1e-10)

def test_shrink_mean_jorion_no_shrink():
    mu = np.array([0.05, 0.05, 0.05])
    S = np.eye(3) * 0.1
    T = 1000
    shrunk = moments.shrink_mean_jorion(mu, S, T)
    np.testing.assert_allclose(shrunk, mu, atol=1e-8)

def test_shrink_covariance_ledoit_wolf_constant_correlation_properties():
    R = np.random.randn(50, 4)
    S_hat = np.cov(R, rowvar=False)
    Sigma = moments.shrink_covariance_ledoit_wolf(R, S_hat, target="constant_correlation")
    # diagonal entries preserved
    np.testing.assert_allclose(np.diag(Sigma), np.diag(S_hat), atol=1e-8)
    # result is positive semi-definite
    eigvals = np.linalg.eigvalsh(Sigma)
    assert np.all(eigvals >= -1e-8)

import numpy as np
import pandas as pd

from pyvallocation.moments import (
    estimate_moments,
    factor_covariance_poet,
    robust_covariance_tyler,
    robust_mean_huber,
    robust_mean_median_of_means,
    shrink_covariance_nls,
    shrink_covariance_oas,
    shrink_mean_james_stein,
    sparse_precision_glasso,
)


def _toy_returns(T: int = 252, N: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.standard_t(df=5, size=(T, N)) * 0.01
    dates = pd.date_range("2020-01-01", periods=T, freq="B")
    cols = [f"A{i}" for i in range(N)]
    return pd.DataFrame(data, index=dates, columns=cols)


def test_shrink_covariance_oas_psd_and_shape():
    R = _toy_returns()
    Sigma = shrink_covariance_oas(R)
    assert isinstance(Sigma, pd.DataFrame)
    np.linalg.cholesky(Sigma.values + 1e-9 * np.eye(Sigma.shape[0]))


def test_shrink_covariance_nls_improves_conditioning():
    R = _toy_returns()
    Sigma_sample = np.cov(R.values, rowvar=False, ddof=1)
    Sigma_nls = shrink_covariance_nls(R)
    cond_sample = np.linalg.cond(Sigma_sample)
    cond_nls = np.linalg.cond(Sigma_nls.values)
    assert cond_nls <= cond_sample * 1.05


def test_factor_covariance_poet_psd():
    R = _toy_returns(T=128, N=6)
    Sigma = factor_covariance_poet(R)
    assert isinstance(Sigma, pd.DataFrame)
    eigvals = np.linalg.eigvalsh(Sigma.values)
    assert np.all(eigvals >= -1e-6)


def test_robust_covariance_tyler_trace_positive():
    R = _toy_returns(T=200, N=8)
    Sigma = robust_covariance_tyler(R, shrinkage=0.1)
    assert np.trace(Sigma.values) > 0.0


def test_sparse_precision_glasso_inverts():
    R = _toy_returns(T=320, N=8)
    Sigma, Theta = sparse_precision_glasso(R, return_precision=True)
    Theta_vals = Theta.values if isinstance(Theta, pd.DataFrame) else Theta
    Sigma_vals = Sigma.values if isinstance(Sigma, pd.DataFrame) else Sigma
    identity = Theta_vals @ Sigma_vals
    assert np.allclose(identity, np.eye(identity.shape[0]), atol=1e-3)


def test_shrink_mean_james_stein_between_target_and_sample():
    R = _toy_returns(T=260, N=6)
    mu_sample = R.mean()
    Sigma = np.cov(R.values, rowvar=False, ddof=1)
    mu_js = shrink_mean_james_stein(mu_sample, Sigma, T=R.shape[0])
    gm = np.full_like(mu_sample.values, mu_sample.mean())
    d_js = np.linalg.norm(mu_js.values - gm)
    d_sample = np.linalg.norm(mu_sample.values - gm)
    assert d_js <= d_sample + 1e-12


def test_robust_means_defined():
    R = _toy_returns()
    mu_huber = robust_mean_huber(R)
    mu_mom = robust_mean_median_of_means(R, random_state=42)
    assert mu_huber.shape == mu_mom.shape == (R.shape[1],)


def test_estimate_moments_factory_routes():
    R = _toy_returns()
    mu, Sigma = estimate_moments(
        R,
        mean_estimator="james_stein",
        cov_estimator="oas",
    )
    assert isinstance(mu, pd.Series)
    assert isinstance(Sigma, pd.DataFrame)
    assert mu.shape[0] == Sigma.shape[0] == Sigma.shape[1] == R.shape[1]


def test_estimate_moments_pandas_roundtrip():
    R = _toy_returns(T=64, N=5)
    weights = pd.Series(np.full(len(R), 1.0 / len(R)), index=R.index)
    mu, Sigma = estimate_moments(
        R,
        p=weights,
        mean_estimator="sample",
        cov_estimator="sample",
    )
    assert isinstance(mu, pd.Series)
    assert isinstance(Sigma, pd.DataFrame)
    assert list(mu.index) == list(R.columns)
    assert list(Sigma.index) == list(R.columns)
    assert list(Sigma.columns) == list(R.columns)

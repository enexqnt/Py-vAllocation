import pytest
import numpy as np
import pyvallocation.optimization as optimization
from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper
from pyvallocation.utils.functions import portfolio_cvar

def test_assets_distribution_parametric():
    mu = np.array([0.05, 0.1])
    cov = np.array([[0.01, 0.002], [0.002, 0.02]])
    dist = AssetsDistribution(mu=mu, cov=cov)
    assert hasattr(dist, 'N') and dist.N == 2
    assert np.allclose(dist.mu, mu)
    assert np.allclose(dist.cov, cov)

def test_assets_distribution_scenarios():
    scenarios = np.array([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]])
    probs = np.array([0.2, 0.3, 0.5])
    dist = AssetsDistribution(scenarios=scenarios, probabilities=probs)
    assert hasattr(dist, 'T') and dist.T == 3
    assert hasattr(dist, 'N') and dist.N == 2
    assert np.allclose(dist.scenarios, scenarios)
    assert np.allclose(dist.probabilities, probs)

def test_assets_distribution_invalid():
    with pytest.raises(ValueError):
        AssetsDistribution()

def test_portfoliowrapper_variance():
    mu = np.array([0.1, 0.2])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    dist = AssetsDistribution(mu=mu, cov=cov)
    wrapper = PortfolioWrapper(dist, num_portfolios=5)
    wrapper.set_constraints()
    wrapper.initialize_optimizer('Variance')
    assert isinstance(wrapper.optimizer, optimization.MeanVariance)
    w = wrapper.optimizer.efficient_portfolio()
    assert w.shape == (2, 1)
    assert np.isclose(np.sum(w), 1.0)

def test_portfoliowrapper_cvar():
    scenarios = np.random.randn(50, 3)
    probs = np.ones(50) / 50
    dist = AssetsDistribution(scenarios=scenarios, probabilities=probs)
    wrapper = PortfolioWrapper(dist, num_portfolios=4, alpha=0.1)
    wrapper.set_constraints()
    wrapper.initialize_optimizer('CVaR')
    assert isinstance(wrapper.optimizer, optimization.MeanCVaR)
    w = wrapper.optimizer.efficient_portfolio()
    assert w.shape == (3, 1)
    assert np.isclose(np.sum(w), 1.0)

def test_set_efficient_frontier_and_minrisk():
    mu = np.array([0.1, 0.2])
    cov = np.array([[0.01, 0.0], [0.0, 0.04]])
    dist = AssetsDistribution(mu=mu, cov=cov)
    wrapper = PortfolioWrapper(dist, num_portfolios=3)
    wrapper.set_constraints()
    wrapper.initialize_optimizer('Variance')
    wrapper.set_efficient_frontier()
    ef = wrapper.efficient_frontier
    assert ef.shape == (2, 3)
    minw = wrapper.get_minrisk_portfolio()
    assert minw.shape == (2,)
    assert np.isclose(np.sum(minw), 1.0)


def test_wrapper_with_tcosts():
    mu = np.array([0.05, 0.1])
    cov = np.array([[0.02, 0.005], [0.005, 0.03]])
    dist = AssetsDistribution(mu=mu, cov=cov)
    wrapper = PortfolioWrapper(dist)
    wrapper.set_constraints()
    wrapper.initialize_optimizer(
        'Variance',
        tcost_lambda=np.array([0.1, 0.2]),
        prev_weights=np.array([0.6, 0.4]),
    )
    w = wrapper.optimizer.efficient_portfolio()
    assert w.shape == (2, 1)
    assert np.isclose(np.sum(w), 1.0)

def test_portfolio_cvar_function():
    ef = np.array([[0.5, 0.5], [0.5, 0.5]])
    scenarios = np.array([[0.1, -0.1], [0.2, -0.2], [0.0, 0.0]])
    probs = np.array([0.3, 0.3, 0.4]).reshape(-1,1)
    alpha = 0.5
    cvar = portfolio_cvar(ef, scenarios, probs, alpha)
    assert isinstance(cvar, np.ndarray)
    assert cvar.shape == (1, 2)

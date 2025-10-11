import numpy as np
import pandas as pd
import pytest

from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper


def test_assets_distribution_normalises_probabilities_and_estimates_moments():
    scenarios = np.array(
        [
            [0.01, 0.02],
            [0.03, 0.01],
            [0.02, -0.01],
        ]
    )
    raw_probabilities = np.array([[0.2], [0.3], [0.8]])  # sums to 1.3
    expected_probs = raw_probabilities.ravel() / raw_probabilities.sum()

    dist = AssetsDistribution(scenarios=scenarios, probabilities=raw_probabilities)

    np.testing.assert_allclose(dist.probabilities, expected_probs)
    np.testing.assert_allclose(dist.mu, scenarios.T @ expected_probs)
    assert dist.N == 2
    assert dist.T == scenarios.shape[0]


def test_assets_distribution_preserves_pandas_asset_names():
    mu = pd.Series([0.01, 0.015], index=["AAA", "BBB"])
    cov = pd.DataFrame(
        [[0.1, 0.02], [0.02, 0.08]],
        index=mu.index,
        columns=mu.index,
    )
    scenarios = pd.DataFrame(
        [[0.0, 0.01], [0.02, -0.01], [0.01, 0.0]],
        columns=mu.index,
    )

    dist = AssetsDistribution(mu=mu, cov=cov, scenarios=scenarios)

    assert dist.asset_names == list(mu.index)
    np.testing.assert_allclose(dist.mu, mu.to_numpy(dtype=float))
    np.testing.assert_allclose(dist.cov, cov.to_numpy(dtype=float))


def _build_long_only_wrapper() -> PortfolioWrapper:
    mu = pd.Series([0.01, 0.015], index=["AAA", "BBB"])
    cov = pd.DataFrame(
        [[0.1, 0.02], [0.02, 0.08]],
        index=mu.index,
        columns=mu.index,
    )
    wrapper = PortfolioWrapper(AssetsDistribution(mu=mu, cov=cov))
    wrapper.set_constraints({"long_only": True, "total_weight": 1.0})
    return wrapper


def test_robust_lambda_frontier_custom_grid_preserves_names():
    wrapper = _build_long_only_wrapper()
    lambdas = [0.0, 0.25, 0.75]
    frontier = wrapper.robust_lambda_frontier(lambdas=lambdas)

    assert frontier.weights.shape == (2, len(lambdas))
    assert frontier.asset_names == ["AAA", "BBB"]

    weights, _, _ = frontier.get_min_risk_portfolio()
    pd.testing.assert_index_equal(weights.index, pd.Index(["AAA", "BBB"]))


def test_robust_lambda_frontier_handles_single_portfolio_request():
    wrapper = _build_long_only_wrapper()
    frontier = wrapper.robust_lambda_frontier(num_portfolios=1, max_lambda=0.6)

    assert frontier.weights.shape == (2, 1)
    assert frontier.returns.shape == (1,)
    assert frontier.risks.shape == (1,)


def test_robust_lambda_frontier_rejects_negative_lambdas():
    wrapper = _build_long_only_wrapper()
    with pytest.raises(ValueError):
        wrapper.robust_lambda_frontier(lambdas=[-0.1, 0.2])

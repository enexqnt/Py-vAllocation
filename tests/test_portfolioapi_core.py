import numpy as np
import pandas as pd

from pyvallocation.portfolioapi import AssetsDistribution


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

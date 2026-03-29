import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Shared data dimensions
# ---------------------------------------------------------------------------
N_ASSETS = 4
T_SCENARIOS = 60


@pytest.fixture
def rng():
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_mu():
    """Mean return vector (N,)."""
    return np.array([0.08, 0.06, 0.04, 0.03])


@pytest.fixture
def sample_cov():
    """PSD covariance matrix (N, N)."""
    cov = np.array([
        [0.040, 0.010, 0.005, 0.002],
        [0.010, 0.030, 0.008, 0.004],
        [0.005, 0.008, 0.025, 0.006],
        [0.002, 0.004, 0.006, 0.020],
    ])
    return cov


@pytest.fixture
def sample_returns(rng, sample_mu, sample_cov):
    """Scenario matrix (T, N) drawn from multivariate normal."""
    return rng.multivariate_normal(sample_mu, sample_cov, T_SCENARIOS)


@pytest.fixture
def sample_returns_df(sample_returns):
    """Scenario DataFrame with asset labels."""
    return pd.DataFrame(
        sample_returns,
        columns=["Equity", "Credit", "Govt", "Gold"],
    )


@pytest.fixture
def sample_weights():
    """Equal-weight vector (N,)."""
    return np.array([0.25, 0.25, 0.25, 0.25])


@pytest.fixture
def sample_weights_series(sample_weights):
    """Equal-weight Series with asset labels."""
    return pd.Series(
        sample_weights,
        index=["Equity", "Credit", "Govt", "Gold"],
        name="EqualWeight",
    )


@pytest.fixture
def uniform_probs():
    """Uniform probability vector (T,)."""
    return np.full(T_SCENARIOS, 1.0 / T_SCENARIOS)

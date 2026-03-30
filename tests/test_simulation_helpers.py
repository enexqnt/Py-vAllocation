"""Tests for simulate_paths, horizon_report, drawdown_quantile."""
import numpy as np
import pytest
from pyvallocation import simulate_paths, horizon_report, drawdown_quantile, reprice_exp


@pytest.fixture
def log_returns():
    rng = np.random.default_rng(42)
    return rng.normal(0.001, 0.02, (200, 3))


@pytest.fixture
def equal_weights():
    return np.array([1/3, 1/3, 1/3])


class TestSimulatePaths:
    def test_shape(self, log_returns):
        paths = simulate_paths(log_returns, horizon=10, n_paths=50, seed=1)
        assert paths.shape == (50, 10, 3)

    def test_with_reprice(self, log_returns):
        paths = simulate_paths(log_returns, horizon=10, n_paths=50, reprice=reprice_exp, seed=1)
        assert paths.shape == (50, 10, 3)
        assert np.all(np.isfinite(paths))

    def test_1d_input(self):
        x = np.random.default_rng(42).normal(0, 0.01, 100)
        paths = simulate_paths(x, horizon=5, n_paths=20, seed=1)
        assert paths.shape == (20, 5, 1)

    def test_variance_increases_with_step(self, log_returns):
        paths = simulate_paths(log_returns, horizon=20, n_paths=500, seed=1)
        var_early = paths[:, 2, :].var(axis=0).mean()
        var_late = paths[:, -1, :].var(axis=0).mean()
        assert var_late > var_early


class TestHorizonReport:
    def test_shape(self, log_returns, equal_weights):
        df = horizon_report(equal_weights, log_returns, horizons=[4, 13], n_simulations=500, seed=1)
        assert df.shape[0] == 2
        assert "mean" in df.columns

    def test_default_labels(self, log_returns, equal_weights):
        df = horizon_report(equal_weights, log_returns, horizons=[4, 13, 26, 52], n_simulations=500, seed=1)
        assert list(df.index) == ["1m", "3m", "6m", "1y"]

    def test_risk_increases_with_horizon(self, log_returns, equal_weights):
        df = horizon_report(equal_weights, log_returns, horizons=[4, 26], n_simulations=2000, seed=1)
        assert df.loc["6m", "stdev"] > df.loc["1m", "stdev"]


class TestDrawdownQuantile:
    def test_bounds(self, log_returns, equal_weights):
        result = drawdown_quantile(equal_weights, log_returns, horizon=20, n_paths=200, seed=1)
        assert 0 <= result["max_dd_mean"] <= 1
        assert 0 <= result["max_dd_95"] <= 1

    def test_longer_horizon_worse(self, log_returns, equal_weights):
        short = drawdown_quantile(equal_weights, log_returns, horizon=5, n_paths=500, seed=1)
        long = drawdown_quantile(equal_weights, log_returns, horizon=50, n_paths=500, seed=1)
        assert long["max_dd_95"] >= short["max_dd_95"]

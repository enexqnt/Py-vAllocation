"""Comprehensive tests for pyvallocation utility modules."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from pyvallocation.utils.constraints import build_G_h_A_b
from pyvallocation.utils.data_helpers import (
    numpy_weights_to_pandas_series,
    pandas_to_numpy_returns,
)
from pyvallocation.utils.functions import (
    portfolio_cvar,
    portfolio_var,
    portfolio_variance,
    portfolio_volatility,
)
from pyvallocation.utils.validation import (
    check_non_negativity,
    check_weights_sum_to_one,
    ensure_psd_matrix,
    is_psd,
)
from pyvallocation.utils.weights import (
    ensure_samples_matrix,
    normalize_weights,
)

# ── conftest constants ──────────────────────────────────────────────
N_ASSETS = 4
T_SCENARIOS = 60


# ====================================================================
# constraints.py
# ====================================================================


class TestBuildLongOnly:
    """build_G_h_A_b with long_only=True should produce -I rows."""

    def test_build_long_only(self):
        G, h, A, b = build_G_h_A_b(N_ASSETS, long_only=True, total_weight=None)
        assert G is not None
        np.testing.assert_array_equal(G, -np.eye(N_ASSETS))
        np.testing.assert_array_equal(h, np.zeros(N_ASSETS))
        # No equality constraint when total_weight is None
        assert A is None
        assert b is None


class TestBuildTotalWeight:
    """build_G_h_A_b with total_weight should produce a ones-row equality."""

    @pytest.mark.filterwarnings("ignore:No position bounds")
    def test_build_total_weight(self):
        tw = 1.0
        G, h, A, b = build_G_h_A_b(N_ASSETS, long_only=False, total_weight=tw)
        assert A is not None and b is not None
        np.testing.assert_array_equal(A, np.ones((1, N_ASSETS)))
        np.testing.assert_array_equal(b, np.array([tw]))

    @pytest.mark.filterwarnings("ignore:No position bounds")
    def test_build_total_weight_fractional(self):
        tw = 0.5
        _, _, A, b = build_G_h_A_b(N_ASSETS, long_only=False, total_weight=tw)
        assert b is not None
        assert float(b[0]) == pytest.approx(tw)


class TestBuildBounds:
    """Box bounds — single tuple and per-asset list."""

    def test_build_bounds_single_tuple(self):
        lb, ub = 0.1, 0.4
        G, h, _, _ = build_G_h_A_b(
            N_ASSETS, long_only=False, total_weight=None, bounds=(lb, ub)
        )
        assert G is not None and h is not None
        # lower: -I row, h = -lb; upper: +I row, h = ub
        # Total: 2 * N_ASSETS rows
        assert G.shape == (2 * N_ASSETS, N_ASSETS)
        # Lower-bound block: -I with h = -lb
        np.testing.assert_array_equal(G[:N_ASSETS], -np.eye(N_ASSETS))
        np.testing.assert_array_almost_equal(h[:N_ASSETS], [-lb] * N_ASSETS)
        # Upper-bound block: +I with h = ub
        np.testing.assert_array_equal(G[N_ASSETS:], np.eye(N_ASSETS))
        np.testing.assert_array_almost_equal(h[N_ASSETS:], [ub] * N_ASSETS)

    def test_build_bounds_per_asset(self):
        per_asset = [(0.0, 0.3), (0.05, 0.5), (0.0, 0.25), (0.1, 0.4)]
        G, h, _, _ = build_G_h_A_b(
            N_ASSETS, long_only=False, total_weight=None, bounds=per_asset
        )
        assert G is not None and h is not None
        # Each asset contributes a lower-bound row and an upper-bound row
        assert G.shape[0] == 2 * N_ASSETS
        # Spot-check asset 1 lower bound: row should have -1 at index 1
        # Rows are generated in pairs per asset: lower then upper
        for idx, (lb, ub) in enumerate(per_asset):
            row_lb = 2 * idx
            row_ub = 2 * idx + 1
            assert G[row_lb, idx] == -1.0
            assert float(h[row_lb]) == pytest.approx(-lb)
            assert G[row_ub, idx] == 1.0
            assert float(h[row_ub]) == pytest.approx(ub)


class TestBuildRelativeBounds:
    """Relative-bound constraints: w_i - w_j <= k."""

    def test_build_relative_bounds_additive(self):
        k = 0.15
        G, h, _, _ = build_G_h_A_b(
            N_ASSETS,
            long_only=False,
            total_weight=None,
            relative_bounds=[(0, 1, k)],
        )
        assert G is not None and h is not None
        # Single row: coefficient +1 at i=0, -1 at j=1
        assert G.shape == (1, N_ASSETS)
        assert G[0, 0] == 1.0
        assert G[0, 1] == -1.0
        assert float(h[0]) == pytest.approx(k)

    def test_relative_bounds_i_equals_j_raises(self):
        with pytest.raises(ValueError, match="i and j must differ"):
            build_G_h_A_b(
                N_ASSETS,
                long_only=False,
                total_weight=None,
                relative_bounds=[(2, 2, 0.1)],
            )


class TestTotalWeightZeroWarns:
    """total_weight=0 should log a warning."""

    def test_total_weight_zero_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="pyvallocation.utils.constraints"):
            build_G_h_A_b(N_ASSETS, long_only=True, total_weight=0)
        assert any(
            "total_weight=0" in rec.message for rec in caplog.records
        ), "Expected warning about total_weight=0"


class TestNoConstraintsReturnsNone:
    """long_only=False, total_weight=None -> all None."""

    def test_no_constraints_returns_none(self):
        with pytest.warns(UserWarning, match="unbounded"):
            G, h, A, b = build_G_h_A_b(
                N_ASSETS, long_only=False, total_weight=None
            )
        assert G is None
        assert h is None
        assert A is None
        assert b is None


# ====================================================================
# data_helpers.py
# ====================================================================


class TestPandasToNumpyReturns:
    """Price DataFrame -> return arrays."""

    @pytest.fixture()
    def price_df(self):
        """Small price DataFrame that grows monotonically."""
        rng = np.random.default_rng(99)
        dates = pd.date_range("2020-01-01", periods=11, freq="B")
        prices = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, (11, 3)), axis=0)
        return pd.DataFrame(prices, index=dates, columns=["A", "B", "C"])

    def test_pandas_to_numpy_returns_simple(self, price_df):
        ret = pandas_to_numpy_returns(price_df, return_calculation_method="simple")
        assert ret.shape == (10, 3)  # T-1 rows
        # Simple returns: (P_t / P_{t-1}) - 1
        expected = price_df.pct_change(fill_method=None).iloc[1:].to_numpy()
        np.testing.assert_array_almost_equal(ret, expected)

    def test_pandas_to_numpy_returns_log(self, price_df):
        ret = pandas_to_numpy_returns(price_df, return_calculation_method="log")
        assert ret.shape == (10, 3)
        # Log returns: ln(P_t / P_{t-1})
        expected = np.log(price_df / price_df.shift(1)).iloc[1:].to_numpy()
        np.testing.assert_array_almost_equal(ret, expected)


class TestNumpyWeightsToPandasSeries:
    """Weight vector -> labelled Series."""

    def test_numpy_weights_to_pandas_series(self, sample_weights):
        names = ["Equity", "Credit", "Govt", "Gold"]
        s = numpy_weights_to_pandas_series(sample_weights, names)
        assert isinstance(s, pd.Series)
        assert list(s.index) == names
        np.testing.assert_array_equal(s.values, sample_weights)


# ====================================================================
# validation.py
# ====================================================================


class TestIsPsd:
    """is_psd positive and negative cases."""

    def test_is_psd_positive(self):
        assert bool(is_psd(np.eye(4))) is True

    def test_is_psd_negative(self):
        mat = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: -1, 3
        assert bool(is_psd(mat)) is False


class TestEnsurePsd:
    """ensure_psd_matrix fixes near-singular matrices."""

    def test_ensure_psd_adds_jitter(self):
        # Construct a symmetric matrix that is *not* PSD
        mat = np.array([[1.0, 2.0], [2.0, 1.0]])
        assert bool(is_psd(mat)) is False
        fixed = ensure_psd_matrix(mat)
        assert bool(is_psd(fixed)) is True


class TestCheckWeightsSumToOne:
    """check_weights_sum_to_one: valid and invalid."""

    def test_valid(self, sample_weights):
        assert bool(check_weights_sum_to_one(sample_weights)) is True

    def test_invalid(self):
        bad = np.array([0.5, 0.5, 0.5, 0.5])  # sums to 2
        assert bool(check_weights_sum_to_one(bad)) is False


class TestCheckNonNegativity:
    """check_non_negativity: valid and invalid."""

    def test_valid(self, sample_weights):
        assert bool(check_non_negativity(sample_weights)) is True

    def test_invalid(self):
        bad = np.array([0.5, -0.1, 0.3, 0.3])
        assert bool(check_non_negativity(bad)) is False


# ====================================================================
# functions.py
# ====================================================================


class TestPortfolioVariance:
    """portfolio_variance for single and multiple portfolios."""

    def test_portfolio_variance_single(self, sample_weights, sample_cov):
        var = portfolio_variance(sample_weights, sample_cov)
        expected = sample_weights @ sample_cov @ sample_weights
        assert isinstance(var, float)
        assert var == pytest.approx(expected)

    def test_portfolio_variance_multiple(self, sample_cov):
        # Two columns -> array of 2 variances
        w_mat = np.column_stack(
            [np.array([1, 0, 0, 0]), np.array([0.25, 0.25, 0.25, 0.25])]
        )
        result = portfolio_variance(w_mat, sample_cov)
        assert result.shape == (2,)
        assert result[0] == pytest.approx(sample_cov[0, 0])


class TestPortfolioVolatility:
    """portfolio_volatility = sqrt(variance)."""

    def test_portfolio_volatility(self, sample_weights, sample_cov):
        vol = portfolio_volatility(sample_weights, sample_cov)
        var = portfolio_variance(sample_weights, sample_cov)
        assert vol == pytest.approx(np.sqrt(var))


class TestPortfolioCVaR:
    """portfolio_cvar defaults to confidence=0.95."""

    def test_portfolio_cvar_confidence(self, sample_weights, sample_returns):
        cvar = portfolio_cvar(sample_weights, sample_returns, confidence=0.95)
        # CVaR should be a finite scalar for a single portfolio
        assert np.isfinite(cvar)

    def test_portfolio_cvar_default_confidence(self, sample_weights, sample_returns):
        cvar_default = portfolio_cvar(sample_weights, sample_returns)
        cvar_explicit = portfolio_cvar(
            sample_weights, sample_returns, confidence=0.95
        )
        assert cvar_default == pytest.approx(cvar_explicit)


class TestPortfolioVaR:
    """portfolio_var defaults to confidence=0.95."""

    def test_portfolio_var_confidence(self, sample_weights, sample_returns):
        var = portfolio_var(sample_weights, sample_returns, confidence=0.95)
        assert np.isfinite(var)

    def test_portfolio_var_default_confidence(self, sample_weights, sample_returns):
        var_default = portfolio_var(sample_weights, sample_returns)
        var_explicit = portfolio_var(
            sample_weights, sample_returns, confidence=0.95
        )
        assert var_default == pytest.approx(var_explicit)


class TestDemeanDoesNotMutateInput:
    """Verify that demean=True does not mutate the original return array."""

    def test_demean_does_not_mutate_input(self, sample_weights, sample_returns):
        R_copy = sample_returns.copy()
        portfolio_cvar(sample_weights, sample_returns, demean=True)
        np.testing.assert_array_equal(sample_returns, R_copy)


# ====================================================================
# weights.py
# ====================================================================


class TestEnsureSamplesMatrix:
    """ensure_samples_matrix shape and label handling."""

    def test_ensure_samples_matrix_1d(self, sample_weights):
        mat, names, snames = ensure_samples_matrix(sample_weights)
        assert mat.shape == (N_ASSETS, 1)
        assert names is None
        assert snames is None

    def test_ensure_samples_matrix_dataframe(self, sample_weights_series):
        df = pd.DataFrame(
            {
                "p1": sample_weights_series,
                "p2": sample_weights_series * 2,
            }
        )
        mat, asset_names, sample_names = ensure_samples_matrix(df)
        assert mat.shape == (N_ASSETS, 2)
        assert asset_names == list(df.index)
        assert sample_names == ["p1", "p2"]

    def test_ensure_samples_matrix_3d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            ensure_samples_matrix(np.zeros((2, 3, 4)))


class TestNormalizeWeights:
    """normalize_weights sums to 1 and rejects negative sums."""

    def test_normalize_weights_sums_to_one(self):
        raw = np.array([2.0, 3.0, 5.0])
        result = normalize_weights(raw, num_samples=3)
        assert result.sum() == pytest.approx(1.0)
        np.testing.assert_array_almost_equal(result, raw / raw.sum())

    def test_normalize_weights_negative_raises(self):
        raw = np.array([-1.0, -2.0, -3.0])
        with pytest.raises(ValueError, match="positive finite"):
            normalize_weights(raw, num_samples=3)

    def test_normalize_weights_none_uniform(self):
        result = normalize_weights(None, num_samples=5)
        assert result.shape == (5,)
        assert result.sum() == pytest.approx(1.0)
        np.testing.assert_array_almost_equal(result, np.full(5, 0.2))

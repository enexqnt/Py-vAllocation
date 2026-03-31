"""Tests for v0.6.0: Meucci Prayer pipeline, return-type tracking, bug fixes."""

import logging

import numpy as np
import pandas as pd
import pytest

from pyvallocation import PortfolioWrapper, compose_repricers
from pyvallocation.utils.projection import (
    log2simple,
    project_scenarios,
    reprice_exp,
    reprice_taylor,
    convert_scenarios_compound_to_simple,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def price_df():
    """Synthetic price DataFrame (101 rows, 3 assets)."""
    rng = np.random.default_rng(7)
    log_rets = rng.multivariate_normal(
        [0.0005, 0.0003, 0.0001],
        np.diag([0.0002, 0.0001, 0.00005]),
        100,
    )
    prices = 100 * np.exp(np.cumsum(log_rets, axis=0))
    prices = np.vstack([np.full(3, 100.0), prices])
    return pd.DataFrame(prices, columns=["SPY", "TLT", "GLD"])


@pytest.fixture
def log_returns(price_df):
    return np.log(price_df / price_df.shift(1)).iloc[1:]


@pytest.fixture
def simple_mu():
    return np.array([0.08, 0.06, 0.04])


@pytest.fixture
def simple_cov():
    return np.array([
        [0.040, 0.010, 0.005],
        [0.010, 0.030, 0.008],
        [0.005, 0.008, 0.025],
    ])


# ---------------------------------------------------------------------------
# Phase 3A: from_prices
# ---------------------------------------------------------------------------

class TestFromPrices:
    def test_creates_wrapper(self, price_df):
        wrapper = PortfolioWrapper.from_prices(price_df)
        assert wrapper.dist.N == 3
        assert wrapper.dist.scenarios is not None
        assert wrapper.dist.scenarios.shape[0] == 100  # T+1 prices → T returns

    def test_returns_match_pct_change(self, price_df):
        wrapper = PortfolioWrapper.from_prices(price_df)
        expected = price_df.pct_change().iloc[1:].to_numpy()
        np.testing.assert_allclose(wrapper.dist.scenarios, expected, atol=1e-12)

    def test_variance_frontier(self, price_df):
        wrapper = PortfolioWrapper.from_prices(price_df)
        f = wrapper.variance_frontier(num_portfolios=5)
        assert f.weights.shape == (3, 5)

    def test_numpy_array_input(self, price_df):
        wrapper = PortfolioWrapper.from_prices(price_df.to_numpy())
        assert wrapper.dist.N == 3


# ---------------------------------------------------------------------------
# Phase 3B: from_invariants
# ---------------------------------------------------------------------------

class TestFromInvariants:
    def test_default_reprice_exp(self, log_returns):
        wrapper = PortfolioWrapper.from_invariants(
            log_returns, horizon=2, n_simulations=500, seed=42,
        )
        assert wrapper.dist.scenarios is not None
        assert wrapper.dist.scenarios.shape == (500, 3)

    def test_with_dict_reprice(self, log_returns):
        wrapper = PortfolioWrapper.from_invariants(
            log_returns,
            reprice={"SPY": reprice_exp, "TLT": reprice_exp, "GLD": reprice_exp},
            horizon=2, n_simulations=500, seed=42,
        )
        assert wrapper.dist.N == 3

    def test_factor_model_custom_repricer(self):
        rng = np.random.default_rng(99)
        factors = rng.standard_normal((200, 2))  # 2 factors, 200 obs

        def repricer(projected):
            f1, f2 = projected[:, 0], projected[:, 1]
            return np.column_stack([np.exp(f1) - 1, -4 * f2 + 10 * f2**2])

        wrapper = PortfolioWrapper.from_invariants(
            factors, reprice=repricer, horizon=4, n_simulations=300, seed=1,
        )
        assert wrapper.dist.N == 2  # 2 instruments from 2 factors
        assert wrapper.dist.scenarios.shape == (300, 2)

    def test_dict_reprice_requires_dataframe(self):
        with pytest.raises(TypeError, match="Dict reprice requires DataFrame"):
            PortfolioWrapper.from_invariants(
                np.zeros((10, 2)),
                reprice={"a": reprice_exp, "b": reprice_exp},
            )

    def test_seed_reproducibility(self, log_returns):
        w1 = PortfolioWrapper.from_invariants(log_returns, horizon=2, n_simulations=100, seed=42)
        w2 = PortfolioWrapper.from_invariants(log_returns, horizon=2, n_simulations=100, seed=42)
        np.testing.assert_array_equal(w1.dist.scenarios, w2.dist.scenarios)


# ---------------------------------------------------------------------------
# Phase 3C: compose_repricers
# ---------------------------------------------------------------------------

class TestComposeRepricers:
    def test_routes_correctly(self):
        dy = np.array([[0.01, -0.02], [0.03, 0.01]])
        combined, names = compose_repricers(
            {"Stock": reprice_exp, "Bond": lambda x: -4.5 * x},
            invariant_columns=["Stock", "Bond"],
        )
        assert names == ["Stock", "Bond"]
        result = combined(dy)
        np.testing.assert_allclose(result[:, 0], np.exp(dy[:, 0]) - 1)
        np.testing.assert_allclose(result[:, 1], -4.5 * dy[:, 1])

    def test_missing_key_raises(self):
        with pytest.raises(KeyError, match="no matching invariant column"):
            compose_repricers({"A": reprice_exp, "B": reprice_exp}, invariant_columns=["A"])


# ---------------------------------------------------------------------------
# Phase 3D: project_scenarios seed
# ---------------------------------------------------------------------------

class TestProjectScenariosSeed:
    def test_seed_reproducible(self):
        R = np.random.default_rng(0).standard_normal((50, 3))
        s1 = project_scenarios(R, investment_horizon=3, n_simulations=100, seed=42)
        s2 = project_scenarios(R, investment_horizon=3, n_simulations=100, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self):
        R = np.random.default_rng(0).standard_normal((50, 3))
        s1 = project_scenarios(R, investment_horizon=3, n_simulations=100, seed=42)
        s2 = project_scenarios(R, investment_horizon=3, n_simulations=100, seed=99)
        assert not np.array_equal(s1, s2)


# ---------------------------------------------------------------------------
# Phase 2: return_type conversion
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_from_moments_log_converts(self, simple_mu, simple_cov):
        # Compute log moments from simple
        from pyvallocation.utils.projection import simple2log
        mu_log, cov_log = simple2log(simple_mu, simple_cov)

        wrapper = PortfolioWrapper.from_moments(mu_log, cov_log, return_type="log")
        # Should round-trip back to simple (approximately)
        np.testing.assert_allclose(wrapper.dist.mu, simple_mu, atol=1e-10)
        np.testing.assert_allclose(
            np.asarray(wrapper.dist.cov, dtype=float),
            simple_cov, atol=1e-6,
        )

    def test_from_scenarios_log_converts(self):
        rng = np.random.default_rng(42)
        log_scen = rng.standard_normal((50, 3)) * 0.01

        wrapper = PortfolioWrapper.from_scenarios(log_scen, return_type="log")
        expected = np.exp(log_scen) - 1.0
        np.testing.assert_allclose(wrapper.dist.scenarios, expected, atol=1e-12)

    def test_from_moments_simple_passthrough(self, simple_mu, simple_cov):
        wrapper = PortfolioWrapper.from_moments(simple_mu, simple_cov)
        np.testing.assert_allclose(wrapper.dist.mu, simple_mu)


# ---------------------------------------------------------------------------
# Phase 1: Robust posterior fixes
# ---------------------------------------------------------------------------

class TestRobustPosterior:
    @pytest.fixture
    def posterior(self):
        from pyvallocation.bayesian import RobustBayesPosterior
        rng = np.random.default_rng(42)
        N = 3
        mu0 = np.zeros(N)
        sigma0 = np.eye(N) * 0.04
        sample_mu = rng.standard_normal(N) * 0.01
        sample_sigma = np.eye(N) * 0.03
        return RobustBayesPosterior.from_niw(
            prior_mu=mu0, prior_sigma=sigma0, t0=5, nu0=10,
            sample_mu=sample_mu, sample_sigma=sample_sigma, n_obs=60,
        )

    def test_stores_sigma_ce(self, posterior):
        wrapper = PortfolioWrapper.from_robust_posterior(posterior)
        sigma_ce = np.asarray(posterior.sigma, dtype=float)
        np.testing.assert_allclose(
            np.asarray(wrapper.dist.cov, dtype=float),
            sigma_ce, atol=1e-12,
        )

    def test_stores_s_mu_separately(self, posterior):
        wrapper = PortfolioWrapper.from_robust_posterior(posterior)
        s_mu = np.asarray(posterior.s_mu, dtype=float)
        np.testing.assert_allclose(wrapper._uncertainty_cov, s_mu, atol=1e-12)

    def test_both_frontiers_work(self, posterior):
        wrapper = PortfolioWrapper.from_robust_posterior(posterior)
        mv = wrapper.variance_frontier(num_portfolios=3)
        robust = wrapper.robust_lambda_frontier(num_portfolios=3)
        assert mv.weights.shape[1] == 3
        assert robust.weights.shape[1] == 3

    def test_robust_frontier_has_vol_overlay(self, posterior):
        wrapper = PortfolioWrapper.from_robust_posterior(posterior)
        f = wrapper.robust_lambda_frontier(num_portfolios=3)
        assert "Volatility" in f.alternate_risks

    def test_report_mu_cov_converts(self, posterior):
        wrapper = PortfolioWrapper.from_robust_posterior(posterior)
        mu_r, cov_r = wrapper._report_mu_cov()
        mu_expected, cov_expected = log2simple(wrapper.dist.mu, wrapper.dist.cov)
        np.testing.assert_allclose(mu_r, mu_expected, atol=1e-12)
        np.testing.assert_allclose(cov_r, cov_expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Phase 4A: EP non-sequential linearization fix
# ---------------------------------------------------------------------------

class TestEPNonSequentialFix:
    def test_mean_plus_vol_linearization(self):
        """Non-sequential EP with mean + vol views should linearize at updated mean."""
        from pyvallocation.views import FlexibleViewsProcessor
        rng = np.random.default_rng(42)
        R = pd.DataFrame(
            rng.multivariate_normal([0.0, 0.0], [[0.01, 0.002], [0.002, 0.01]], 3000),
            columns=["A", "B"],
        )

        target_vol = 0.08
        processor = FlexibleViewsProcessor(
            prior_risk_drivers=R,
            mean_views={"A": ("==", 0.05)},
            vol_views={"B": ("==", target_vol)},
            sequential=False,
        )
        mu_post, cov_post = processor.get_posterior()
        cov_arr = np.asarray(cov_post, dtype=float)
        achieved_vol = np.sqrt(cov_arr[1, 1])
        # Should be close to target; before fix it used prior linearization
        assert abs(achieved_vol - target_vol) < 0.02, f"Achieved vol {achieved_vol:.4f} far from target {target_vol}"


# ---------------------------------------------------------------------------
# Phase 4D: CVaR convexity opt-out
# ---------------------------------------------------------------------------

class TestCVaRConvexity:
    def test_enforce_convexity_false(self, simple_mu, simple_cov):
        rng = np.random.default_rng(42)
        scen = rng.multivariate_normal(simple_mu, simple_cov, 200)
        wrapper = PortfolioWrapper.from_scenarios(scen)
        f = wrapper.cvar_frontier(num_portfolios=5, seed=1, enforce_convexity=False)
        assert f.weights.shape[1] == 5

    def test_enforce_convexity_default_true(self, simple_mu, simple_cov):
        rng = np.random.default_rng(42)
        scen = rng.multivariate_normal(simple_mu, simple_cov, 200)
        wrapper = PortfolioWrapper.from_scenarios(scen)
        f = wrapper.cvar_frontier(num_portfolios=5, seed=1)
        # Risks should be monotone non-decreasing
        assert np.all(np.diff(f.risks) >= -1e-12)


# ---------------------------------------------------------------------------
# K→N compose_repricers (multi-driver instruments)
# ---------------------------------------------------------------------------

class TestComposeRepricersKN:
    def test_multi_driver_instrument(self):
        """2-driver instrument receives (n_sim, 2) array."""
        rng = np.random.default_rng(42)
        dy = rng.standard_normal((100, 3))  # 3 invariant columns

        def option_fn(factors):
            # factors is (n_sim, 2): underlying + vol
            return 0.5 * (np.exp(factors[:, 0]) - 1) + 0.3 * factors[:, 1]

        fn, names = compose_repricers(
            {"Stock": (["stock_lr"], reprice_exp),
             "Call": (["stock_lr", "iv_chg"], option_fn)},
            invariant_columns=["stock_lr", "bond_dy", "iv_chg"],
        )
        assert names == ["Stock", "Call"]
        result = fn(dy)
        assert result.shape == (100, 2)
        # Stock uses column 0, Call uses columns 0 and 2
        np.testing.assert_allclose(result[:, 0], np.exp(dy[:, 0]) - 1)
        expected_call = 0.5 * (np.exp(dy[:, 0]) - 1) + 0.3 * dy[:, 2]
        np.testing.assert_allclose(result[:, 1], expected_call)

    def test_named_single_driver(self):
        """Single driver with name different from instrument."""
        dy = np.array([[0.01, -0.005], [0.02, 0.003]])
        fn, names = compose_repricers(
            {"TLT": (["yield_10y"], lambda x: -7.0 * x)},
            invariant_columns=["equity_lr", "yield_10y"],
        )
        assert names == ["TLT"]
        result = fn(dy)
        assert result.shape == (2, 1)
        np.testing.assert_allclose(result[:, 0], -7.0 * dy[:, 1])

    def test_k4_to_n3_full_pipeline(self):
        """4 invariants → 3 instruments end-to-end via from_invariants."""
        rng = np.random.default_rng(42)
        invariants = pd.DataFrame(
            rng.multivariate_normal([0.001, -0.0001, 0.0005, 0.0],
                                    np.diag([0.0004, 0.00001, 0.0003, 0.00005]), 200),
            columns=["eq_lr", "yield_10y", "aapl_lr", "aapl_iv"],
        )

        def call_fn(factors):
            ds = np.exp(factors[:, 0]) - 1  # delta * underlying simple return
            dv = factors[:, 1]               # vega * vol change
            return 0.6 * ds + 0.3 * dv

        wrapper = PortfolioWrapper.from_invariants(
            invariants,
            reprice={
                "SPY": (["eq_lr"], reprice_exp),
                "TLT": (["yield_10y"], lambda dy: reprice_taylor(dy, delta=-17, gamma=200)),
                "AAPL_Call": (["aapl_lr", "aapl_iv"], call_fn),
            },
            horizon=12, n_simulations=2000, seed=42,
        )
        assert wrapper.dist.N == 3
        assert wrapper.dist.asset_names == ["SPY", "TLT", "AAPL_Call"]
        assert wrapper.dist.scenarios.shape == (2000, 3)
        # Should be able to run a frontier on 3 instruments
        f = wrapper.variance_frontier(num_portfolios=3)
        assert f.weights.shape == (3, 3)

    def test_backward_compat_1to1(self):
        """Old 1-to-1 dict API still works via from_invariants."""
        rng = np.random.default_rng(42)
        invariants = pd.DataFrame(
            rng.standard_normal((100, 2)) * 0.01,
            columns=["A", "B"],
        )
        wrapper = PortfolioWrapper.from_invariants(
            invariants,
            reprice={"A": reprice_exp, "B": reprice_exp},
            horizon=2, n_simulations=200, seed=1,
        )
        assert wrapper.dist.N == 2
        assert wrapper.dist.asset_names == ["A", "B"]


# ---------------------------------------------------------------------------
# get_scenarios
# ---------------------------------------------------------------------------

class TestGetScenarios:
    def test_from_risk_drivers(self):
        scenarios = pd.DataFrame(
            np.random.default_rng(42).standard_normal((50, 2)) * 0.1,
            columns=["X", "Y"],
        )
        from pyvallocation.views import FlexibleViewsProcessor
        fvp = FlexibleViewsProcessor(prior_risk_drivers=scenarios, mean_views={"X": 0.05})
        retrieved = fvp.get_scenarios()
        assert isinstance(retrieved, pd.DataFrame)
        assert list(retrieved.columns) == ["X", "Y"]
        assert retrieved.shape == (50, 2)

    def test_from_moments_synthesis(self):
        from pyvallocation.views import FlexibleViewsProcessor
        fvp = FlexibleViewsProcessor(
            prior_mean=np.array([0.05, 0.03]),
            prior_cov=np.diag([0.04, 0.02]),
            num_scenarios=500,
            mean_views={"0": 0.06},
        )
        retrieved = fvp.get_scenarios()
        assert isinstance(retrieved, np.ndarray)
        assert retrieved.shape == (500, 2)


# ---------------------------------------------------------------------------
# stress_invariants
# ---------------------------------------------------------------------------

class TestStressInvariants:
    @pytest.fixture
    def invariants(self):
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            rng.multivariate_normal([0.001, -0.0001], np.diag([0.0004, 0.00001]), 300),
            columns=["equity_lr", "yield_10y"],
        )

    def test_basic_no_stress(self, invariants):
        """Without stress_views, returns nominal metrics only."""
        from pyvallocation import stress_invariants
        report = stress_invariants(
            invariants,
            weights=np.array([0.6, 0.4]),
            reprice={"Equity": (["equity_lr"], reprice_exp),
                     "Bond": (["yield_10y"], lambda dy: reprice_taylor(dy, delta=-7, gamma=50))},
            horizon=12, n_simulations=1000, seed=42,
        )
        assert isinstance(report, pd.DataFrame)
        assert "return_nom" in report.columns

    def test_stress_yield_shock(self, invariants):
        """Yields up should hurt the bond, worsen portfolio."""
        from pyvallocation import stress_invariants, at_least
        report = stress_invariants(
            invariants,
            weights=np.array([0.3, 0.7]),  # heavy bond allocation
            reprice={"Equity": (["equity_lr"], reprice_exp),
                     "Bond": (["yield_10y"], lambda dy: reprice_taylor(dy, delta=-7, gamma=50))},
            stress_views={"yield_10y": at_least(0.005)},  # yields up by 50bp
            horizon=12, n_simulations=1000, seed=42,
        )
        assert "return_stressed" in report.columns
        # Stressed return should be lower than nominal (yields up hurts bond-heavy portfolio)
        assert report["return_stressed"].iloc[0] < report["return_nom"].iloc[0]

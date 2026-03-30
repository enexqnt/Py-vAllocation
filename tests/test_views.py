"""Tests for pyvallocation.views — FlexibleViewsProcessor, BlackLittermanProcessor, entropy_pooling."""

import numpy as np
import pandas as pd
import pytest

from pyvallocation.views import (
    BlackLittermanProcessor,
    FlexibleViewsProcessor,
    entropy_pooling,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ASSETS = ["Equity", "Credit", "Govt", "Gold"]


def _posterior_mean_from_probs(R, q):
    """Weighted mean from scenario matrix and probability column vector."""
    return (R.T @ q).flatten()


def _posterior_cov_from_probs(R, q):
    """Weighted covariance from scenario matrix and probability column vector."""
    return np.cov(R.T, aweights=q.flatten(), bias=True)


# =========================================================================
# FlexibleViewsProcessor
# =========================================================================


class TestFVPMeanViewEquality:
    """test_fvp_mean_view_equality — mean view on one asset shifts posterior."""

    def test_mean_shifts_toward_target(self, sample_returns_df):
        target = 0.15
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=sample_returns_df,
            mean_views={"Equity": target},
        )
        mu_post, _ = fvp.get_posterior()
        # Posterior mean for Equity should be close to the target
        assert abs(mu_post["Equity"] - target) < 0.02


class TestFVPMeanViewInequality:
    """test_fvp_mean_view_inequality — inequality view (>=)."""

    def test_mean_respects_lower_bound(self, sample_returns_df):
        bound = 0.10
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=sample_returns_df,
            mean_views={"Credit": (">=", bound)},
        )
        mu_post, _ = fvp.get_posterior()
        assert mu_post["Credit"] >= bound - 1e-6


class TestFVPVolView:
    """test_fvp_vol_view — volatility view moves posterior vol toward target."""

    def test_vol_shifts_toward_target(self, sample_returns_df):
        target_vol = 0.25
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=sample_returns_df,
            vol_views={"Equity": target_vol},
        )
        _, cov_post = fvp.get_posterior()
        post_vol = np.sqrt(cov_post.loc["Equity", "Equity"])
        prior_vol = np.sqrt(np.cov(sample_returns_df["Equity"].values, bias=True))
        # Posterior vol should be closer to target than the prior vol was
        assert abs(post_vol - target_vol) < abs(float(prior_vol) - target_vol)


class TestFVPCorrView:
    """test_fvp_corr_view — correlation view between two assets."""

    def test_corr_shifts_toward_target(self, sample_returns_df):
        target_corr = 0.80
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=sample_returns_df,
            corr_views={("Equity", "Credit"): target_corr},
        )
        _, cov_post = fvp.get_posterior()
        vol_eq = np.sqrt(cov_post.loc["Equity", "Equity"])
        vol_cr = np.sqrt(cov_post.loc["Credit", "Credit"])
        post_corr = cov_post.loc["Equity", "Credit"] / (vol_eq * vol_cr)
        # Should move toward the target
        prior_cov = np.cov(sample_returns_df.values.T, bias=True)
        prior_corr = prior_cov[0, 1] / (np.sqrt(prior_cov[0, 0]) * np.sqrt(prior_cov[1, 1]))
        assert abs(post_corr - target_corr) < abs(prior_corr - target_corr)


class TestFVPSkewView:
    """test_fvp_skew_view — skewness view on one asset."""

    def test_skew_view_runs(self, sample_returns_df):
        target_skew = -0.5
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=sample_returns_df,
            skew_views={"Equity": target_skew},
        )
        q = fvp.get_posterior_probabilities()
        assert q.shape == (len(sample_returns_df), 1)
        assert np.isclose(q.sum(), 1.0, atol=1e-6)


class TestFVPCVaRView:
    """test_fvp_cvar_view — CVaR view via recursive EP."""

    def test_cvar_view_runs(self, sample_mu, sample_cov):
        # Build scenarios and compute prior CVaR to set a feasible target
        rng = np.random.default_rng(7)
        S = 1000
        R = rng.multivariate_normal(sample_mu, sample_cov, S)
        gamma = 0.10
        x = R[:, 0]
        n_tail = int(gamma * S)
        prior_cvar = np.sort(x)[:n_tail].mean()
        # Target slightly worse (more negative) than prior, so it is feasible
        target_cvar = prior_cvar - 0.01

        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=R,
            cvar_views={"0": (target_cvar, gamma)},
        )
        q = fvp.get_posterior_probabilities()
        assert q.shape == (S, 1)
        assert np.isclose(q.sum(), 1.0, atol=1e-6)


class TestFVPSequentialVsSimultaneous:
    """test_fvp_sequential_vs_simultaneous — both modes produce valid probabilities."""

    @pytest.fixture
    def views(self):
        return dict(
            mean_views={"Equity": 0.10},
            vol_views={"Credit": 0.20},
        )

    def test_simultaneous(self, sample_returns_df, views):
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=sample_returns_df,
            sequential=False,
            **views,
        )
        q = fvp.get_posterior_probabilities()
        assert q.shape[0] == len(sample_returns_df)
        assert np.all(q >= 0)
        assert np.isclose(q.sum(), 1.0, atol=1e-6)

    def test_sequential(self, sample_returns_df, views):
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=sample_returns_df,
            sequential=True,
            **views,
        )
        q = fvp.get_posterior_probabilities()
        assert q.shape[0] == len(sample_returns_df)
        assert np.all(q >= 0)
        assert np.isclose(q.sum(), 1.0, atol=1e-6)


class TestFVPPosteriorProbsSumToOne:
    """test_fvp_posterior_probabilities_sum_to_one."""

    def test_sum_to_one(self, sample_returns_df):
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=sample_returns_df,
            mean_views={"Equity": 0.10, "Gold": 0.05},
            vol_views={"Credit": 0.18},
        )
        q = fvp.get_posterior_probabilities()
        assert np.isclose(q.sum(), 1.0, atol=1e-6)


class TestFVPPosteriorMomentsShape:
    """test_fvp_posterior_moments_shape — get_posterior returns correct shapes."""

    def test_shapes(self, sample_returns_df):
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=sample_returns_df,
            mean_views={"Equity": 0.10},
        )
        mu, cov = fvp.get_posterior()
        n = sample_returns_df.shape[1]
        assert mu.shape == (n,)
        assert cov.shape == (n, n)


class TestFVPFromScenarios:
    """test_fvp_from_scenarios — construct from scenario matrix (ndarray)."""

    def test_from_ndarray(self, sample_returns):
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=sample_returns,
            mean_views={"0": 0.10},
        )
        mu, cov = fvp.get_posterior()
        assert isinstance(mu, np.ndarray)
        assert isinstance(cov, np.ndarray)


class TestFVPFromMoments:
    """test_fvp_from_moments — construct from mean/cov (parametric mode)."""

    def test_from_moments(self, sample_mu, sample_cov):
        fvp = FlexibleViewsProcessor(
            prior_mean=sample_mu,
            prior_cov=sample_cov,
            num_scenarios=500,
            random_state=99,
            mean_views={"0": 0.12},
        )
        mu, cov = fvp.get_posterior()
        assert mu.shape == (4,)
        assert cov.shape == (4, 4)


class TestFVPInvalidVolTarget:
    """test_fvp_invalid_vol_target_raises — vol target <= 0 raises ValueError."""

    def test_zero_vol_target(self, sample_returns_df):
        with pytest.raises(ValueError, match="positive"):
            FlexibleViewsProcessor(
                prior_risk_drivers=sample_returns_df,
                vol_views={"Equity": 0.0},
            )

    def test_negative_vol_target(self, sample_returns_df):
        with pytest.raises(ValueError, match="positive"):
            FlexibleViewsProcessor(
                prior_risk_drivers=sample_returns_df,
                vol_views={"Equity": -0.10},
            )


class TestFVPInvalidCorrTarget:
    """test_fvp_invalid_corr_target_raises — |corr| > 1 raises ValueError."""

    def test_corr_above_one(self, sample_returns_df):
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            FlexibleViewsProcessor(
                prior_risk_drivers=sample_returns_df,
                corr_views={("Equity", "Credit"): 1.5},
            )

    def test_corr_below_minus_one(self, sample_returns_df):
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            FlexibleViewsProcessor(
                prior_risk_drivers=sample_returns_df,
                corr_views={("Equity", "Credit"): -1.5},
            )


class TestFVPNearZeroVarianceSkew:
    """test_fvp_near_zero_variance_skew_raises — skew on near-zero-variance asset raises."""

    def test_near_zero_variance(self):
        # Create a scenario matrix where one column has near-zero variance
        rng = np.random.default_rng(123)
        R = rng.standard_normal((60, 4))
        R[:, 2] = 1e-15  # near-zero variance for column 2
        df = pd.DataFrame(R, columns=ASSETS)
        with pytest.raises(ValueError, match="near-zero"):
            FlexibleViewsProcessor(
                prior_risk_drivers=df,
                skew_views={"Govt": 0.5},
            )


# =========================================================================
# BlackLittermanProcessor
# =========================================================================


class TestBLPAbsoluteView:
    """test_blp_absolute_view — single absolute view shifts posterior."""

    def test_posterior_shifts(self, sample_mu, sample_cov):
        cov_df = pd.DataFrame(sample_cov, index=ASSETS, columns=ASSETS)
        target = 0.12
        bl = BlackLittermanProcessor(
            prior_cov=cov_df,
            prior_mean=sample_mu,
            mean_views={"Equity": target},
        )
        mu_post, _ = bl.get_posterior()
        # Posterior for Equity should shift toward target relative to prior
        prior_equity = sample_mu[0]
        assert abs(mu_post["Equity"] - target) < abs(prior_equity - target)


class TestBLPRelativeView:
    """test_blp_relative_view — relative view (A outperforms B)."""

    def test_spread(self, sample_mu, sample_cov):
        cov_df = pd.DataFrame(sample_cov, index=ASSETS, columns=ASSETS)
        # Prior spread is sample_mu[0] - sample_mu[3] = 0.08 - 0.03 = 0.05
        # Use a different target so posterior actually moves
        spread_target = 0.10
        bl = BlackLittermanProcessor(
            prior_cov=cov_df,
            prior_mean=sample_mu,
            mean_views={("Equity", "Gold"): spread_target},
        )
        mu_post, _ = bl.get_posterior()
        post_spread = mu_post["Equity"] - mu_post["Gold"]
        prior_spread = sample_mu[0] - sample_mu[3]
        # Posterior spread should be closer to target than prior spread
        assert abs(post_spread - spread_target) < abs(prior_spread - spread_target)


class TestBLPIdzorekConfidence:
    """test_blp_idzorek_confidence — omega='idzorek' with view_confidences."""

    def test_idzorek_runs(self, sample_mu, sample_cov):
        cov_df = pd.DataFrame(sample_cov, index=ASSETS, columns=ASSETS)
        bl = BlackLittermanProcessor(
            prior_cov=cov_df,
            prior_mean=sample_mu,
            mean_views={"Equity": 0.12, "Credit": 0.08},
            view_confidences={"Equity": 0.80, "Credit": 0.50},
            omega="idzorek",
        )
        mu_post, cov_post = bl.get_posterior()
        assert mu_post.shape == (4,)
        assert cov_post.shape == (4, 4)


class TestBLPDefaultOmega:
    """test_blp_default_omega — omega=None uses He-Litterman diagonal."""

    def test_default_omega(self, sample_mu, sample_cov):
        cov_df = pd.DataFrame(sample_cov, index=ASSETS, columns=ASSETS)
        bl = BlackLittermanProcessor(
            prior_cov=cov_df,
            prior_mean=sample_mu,
            mean_views={"Equity": 0.12},
            omega=None,
        )
        mu_post, _ = bl.get_posterior()
        assert mu_post.shape == (4,)


class TestBLPNoViews:
    """test_blp_no_views — no views returns prior."""

    def test_no_views_returns_prior(self, sample_mu, sample_cov):
        cov_df = pd.DataFrame(sample_cov, index=ASSETS, columns=ASSETS)
        bl = BlackLittermanProcessor(
            prior_cov=cov_df,
            prior_mean=sample_mu,
        )
        mu_post, cov_post = bl.get_posterior()
        np.testing.assert_allclose(mu_post.values, sample_mu, atol=1e-10)
        np.testing.assert_allclose(cov_post.values, sample_cov, atol=1e-10)


class TestBLPPosteriorCovPSD:
    """test_blp_posterior_cov_psd — posterior covariance is PSD."""

    def test_psd(self, sample_mu, sample_cov):
        cov_df = pd.DataFrame(sample_cov, index=ASSETS, columns=ASSETS)
        bl = BlackLittermanProcessor(
            prior_cov=cov_df,
            prior_mean=sample_mu,
            mean_views={"Equity": 0.12, "Credit": 0.08},
        )
        _, cov_post = bl.get_posterior()
        eigenvalues = np.linalg.eigvalsh(cov_post.values)
        assert np.all(eigenvalues >= -1e-10), f"Negative eigenvalue: {eigenvalues.min()}"


# =========================================================================
# entropy_pooling (standalone)
# =========================================================================


class TestEntropyPoolingEqualityConstraint:
    """test_entropy_pooling_equality_constraint — simple mean constraint."""

    def test_equality(self, sample_returns, uniform_probs):
        R = sample_returns
        S, N = R.shape
        target_mean_0 = 0.12

        # Equality: E[R_0] = target  =>  sum(q_j * R_j_0) = target
        # Plus normalization: sum(q) = 1
        A = np.vstack([R[:, 0], np.ones(S)])
        b = np.array([target_mean_0, 1.0])

        q = entropy_pooling(uniform_probs, A, b)
        post_mean_0 = (R[:, 0] @ q).item()
        assert abs(post_mean_0 - target_mean_0) < 1e-4


class TestEntropyPoolingInequalityConstraint:
    """test_entropy_pooling_inequality_constraint."""

    def test_inequality(self, sample_returns, uniform_probs):
        R = sample_returns
        S = R.shape[0]
        bound = 0.12

        # Normalization equality
        A = np.ones((1, S))
        b = np.array([1.0])

        # Inequality: E[R_0] <= bound  =>  sum(q * R_0) <= bound
        G = R[:, 0].reshape(1, -1)
        h = np.array([bound])

        q = entropy_pooling(uniform_probs, A, b, G, h)
        post_mean_0 = (R[:, 0] @ q).item()
        assert post_mean_0 <= bound + 1e-4


class TestEntropyPoolingSumsToOne:
    """test_entropy_pooling_posterior_sums_to_one."""

    def test_normalization(self, sample_returns, uniform_probs):
        R = sample_returns
        S = R.shape[0]
        target = 0.10

        A = np.vstack([R[:, 0], np.ones(S)])
        b = np.array([target, 1.0])

        q = entropy_pooling(uniform_probs, A, b)
        assert np.isclose(q.sum(), 1.0, atol=1e-6)

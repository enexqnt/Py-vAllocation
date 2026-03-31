"""Tests for enhanced FlexibleViewsProcessor: helpers, range, rank, quantile, var views."""

import numpy as np
import pandas as pd
import pytest

from pyvallocation.views import (
    FlexibleViewsProcessor,
    at_least,
    at_most,
    between,
    above,
    below,
)


@pytest.fixture
def scenarios():
    """3-asset scenario DataFrame (500 obs)."""
    rng = np.random.default_rng(42)
    R = rng.multivariate_normal(
        [0.01, 0.005, 0.003],
        [[0.02, 0.005, 0.001],
         [0.005, 0.01, 0.002],
         [0.001, 0.002, 0.008]],
        500,
    )
    return pd.DataFrame(R, columns=["A", "B", "C"])


# ---------------------------------------------------------------------------
# View helpers
# ---------------------------------------------------------------------------

class TestViewHelpers:
    def test_at_least(self):
        assert at_least(0.05) == (">=", 0.05)

    def test_at_most(self):
        assert at_most(0.20) == ("<=", 0.20)

    def test_between(self):
        assert between(0.05, 0.10) == ("between", 0.05, 0.10)

    def test_above(self):
        assert above(0.0) == (">", 0.0)

    def test_below(self):
        assert below(0.0) == ("<", 0.0)

    def test_helpers_with_processor(self, scenarios):
        """Helpers are valid view values for FlexibleViewsProcessor."""
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=scenarios,
            mean_views={"A": at_least(0.02), "B": at_most(0.01)},
            vol_views={"C": at_most(0.12)},
        )
        q = fvp.get_posterior_probabilities()
        assert np.isclose(q.sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Range views (between)
# ---------------------------------------------------------------------------

class TestRangeViews:
    def test_mean_between(self, scenarios):
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=scenarios,
            mean_views={"A": between(0.005, 0.02)},
        )
        mu_post = np.asarray(fvp.posterior_returns)
        assert mu_post[0] >= 0.005 - 1e-4
        assert mu_post[0] <= 0.02 + 1e-4

    def test_vol_between(self, scenarios):
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=scenarios,
            vol_views={"A": between(0.10, 0.18)},
        )
        cov_post = np.asarray(fvp.posterior_cov, dtype=float)
        vol_a = np.sqrt(cov_post[0, 0])
        assert vol_a >= 0.10 - 0.005
        assert vol_a <= 0.18 + 0.005

    def test_corr_between(self, scenarios):
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=scenarios,
            corr_views={("A", "B"): between(-0.1, 0.3)},
        )
        cov_post = np.asarray(fvp.posterior_cov, dtype=float)
        corr_ab = cov_post[0, 1] / np.sqrt(cov_post[0, 0] * cov_post[1, 1])
        assert corr_ab >= -0.1 - 0.05
        assert corr_ab <= 0.3 + 0.05

    def test_between_bad_order_raises(self, scenarios):
        with pytest.raises(ValueError, match="lower bound exceeds upper"):
            FlexibleViewsProcessor(
                prior_risk_drivers=scenarios,
                mean_views={"A": between(0.10, 0.01)},
            )


# ---------------------------------------------------------------------------
# Rank views
# ---------------------------------------------------------------------------

class TestRankViews:
    def test_rank_mean_basic(self, scenarios):
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=scenarios,
            rank_mean=["A", "B", "C"],  # E[A] >= E[B] >= E[C]
        )
        mu = np.asarray(fvp.posterior_returns)
        assert mu[0] >= mu[1] - 1e-4
        assert mu[1] >= mu[2] - 1e-4

    def test_rank_mean_two_assets(self, scenarios):
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=scenarios,
            rank_mean=["C", "A"],  # E[C] >= E[A]
        )
        mu = np.asarray(fvp.posterior_returns)
        assert mu[2] >= mu[0] - 1e-4  # C >= A

    def test_rank_mean_too_short(self, scenarios):
        with pytest.raises(ValueError, match="at least 2 assets"):
            FlexibleViewsProcessor(
                prior_risk_drivers=scenarios,
                rank_mean=["A"],
            )

    def test_rank_mean_combines_with_other_views(self, scenarios):
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=scenarios,
            rank_mean=["A", "B"],
            vol_views={"C": at_most(0.10)},
        )
        mu = np.asarray(fvp.posterior_returns)
        assert mu[0] >= mu[1] - 1e-4


# ---------------------------------------------------------------------------
# Variance views
# ---------------------------------------------------------------------------

class TestVarViews:
    def test_var_equality(self, scenarios):
        target_var = 0.015
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=scenarios,
            var_views={"A": target_var},
        )
        cov_post = np.asarray(fvp.posterior_cov, dtype=float)
        assert abs(cov_post[0, 0] - target_var) < 0.003

    def test_var_inequality(self, scenarios):
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=scenarios,
            var_views={"A": at_most(0.01)},
        )
        cov_post = np.asarray(fvp.posterior_cov, dtype=float)
        assert cov_post[0, 0] <= 0.01 + 0.002

    def test_var_between(self, scenarios):
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=scenarios,
            var_views={"B": between(0.005, 0.015)},
        )
        cov_post = np.asarray(fvp.posterior_cov, dtype=float)
        assert cov_post[1, 1] >= 0.005 - 0.002
        assert cov_post[1, 1] <= 0.015 + 0.002

    def test_var_negative_raises(self, scenarios):
        with pytest.raises(ValueError, match="non-negative"):
            FlexibleViewsProcessor(
                prior_risk_drivers=scenarios,
                var_views={"A": -0.01},
            )


# ---------------------------------------------------------------------------
# Quantile (VaR) views
# ---------------------------------------------------------------------------

class TestQuantileViews:
    def test_quantile_equality(self, scenarios):
        """P(R_A <= level) = alpha."""
        level = -0.10
        alpha = 0.10
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=scenarios,
            quantile_views={"A": (level, alpha)},
        )
        q = fvp.get_posterior_probabilities().flatten()
        R_a = scenarios["A"].values
        achieved_prob = (q * (R_a <= level)).sum()
        assert abs(achieved_prob - alpha) < 0.02

    def test_quantile_inequality(self, scenarios):
        """P(R_A <= level) <= alpha (tail probability capped)."""
        level = -0.05
        alpha_cap = 0.08
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=scenarios,
            quantile_views={"A": (level, at_most(alpha_cap))},
        )
        q = fvp.get_posterior_probabilities().flatten()
        R_a = scenarios["A"].values
        achieved_prob = (q * (R_a <= level)).sum()
        assert achieved_prob <= alpha_cap + 0.01

    def test_quantile_bad_probability(self, scenarios):
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            FlexibleViewsProcessor(
                prior_risk_drivers=scenarios,
                quantile_views={"A": (-0.10, 1.5)},
            )


# ---------------------------------------------------------------------------
# Integration: mixing all view types
# ---------------------------------------------------------------------------

class TestMixedViews:
    def test_all_view_types_together(self, scenarios):
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=scenarios,
            mean_views={"A": at_least(0.01)},
            vol_views={"B": at_most(0.12)},
            var_views={"C": at_most(0.01)},
            corr_views={("A", "B"): between(-0.2, 0.5)},
            rank_mean=["A", "B"],
            sequential=True,
        )
        q = fvp.get_posterior_probabilities()
        assert np.isclose(q.sum(), 1.0, atol=1e-6)
        mu = np.asarray(fvp.posterior_returns)
        assert mu[0] >= 0.01 - 1e-3
        assert mu[0] >= mu[1] - 1e-3

    def test_quantile_with_mean_views(self, scenarios):
        fvp = FlexibleViewsProcessor(
            prior_risk_drivers=scenarios,
            mean_views={"A": at_least(0.005)},
            quantile_views={"A": (-0.15, at_most(0.05))},
        )
        q = fvp.get_posterior_probabilities()
        assert np.isclose(q.sum(), 1.0, atol=1e-6)

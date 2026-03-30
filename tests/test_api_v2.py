"""Tests for v0.5.0 API: factories, typed constraints, per-method overrides."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyvallocation import (
    AssetsDistribution,
    Constraints,
    InfeasibleOptimizationError,
    PortfolioFrontier,
    PortfolioWrapper,
    TransactionCosts,
)


# ── Shared helpers ─────────────────────────────────────────────────
N = 4
MU = np.array([0.08, 0.06, 0.04, 0.03])
COV = np.array([
    [0.040, 0.010, 0.005, 0.002],
    [0.010, 0.030, 0.008, 0.004],
    [0.005, 0.008, 0.025, 0.006],
    [0.002, 0.004, 0.006, 0.020],
])


@pytest.fixture()
def scenarios_df():
    rng = np.random.default_rng(42)
    data = rng.multivariate_normal(MU, COV, 80)
    return pd.DataFrame(data, columns=["Equity", "Credit", "Govt", "Gold"])


# ====================================================================
# Factory method tests
# ====================================================================


class TestFromMoments:
    """PortfolioWrapper.from_moments factory."""

    def test_from_moments_default_constraints(self):
        wrapper = PortfolioWrapper.from_moments(MU, COV)
        frontier = wrapper.variance_frontier(num_portfolios=5)
        assert isinstance(frontier, PortfolioFrontier)
        assert frontier.weights.shape[0] == N
        assert frontier.weights.shape[1] == 5
        # Default long-only => all weights >= 0 (within tolerance)
        assert np.all(frontier.weights >= -1e-6)

    def test_from_moments_with_constraints_object(self):
        c = Constraints(bounds=(0, 0.5))
        wrapper = PortfolioWrapper.from_moments(MU, COV, constraints=c)
        frontier = wrapper.variance_frontier(num_portfolios=5)
        assert isinstance(frontier, PortfolioFrontier)
        # Check upper bound respected
        assert np.all(frontier.weights <= 0.5 + 1e-6)

    def test_from_moments_with_dict_constraints(self):
        d = {"long_only": True, "total_weight": 1.0, "bounds": (0.0, 0.4)}
        wrapper = PortfolioWrapper.from_moments(MU, COV, constraints=d)
        frontier = wrapper.variance_frontier(num_portfolios=5)
        assert isinstance(frontier, PortfolioFrontier)
        # weights bounded at 0.4
        assert np.all(frontier.weights <= 0.4 + 1e-6)


class TestFromScenarios:
    """PortfolioWrapper.from_scenarios factory."""

    def test_from_scenarios_default(self, scenarios_df):
        wrapper = PortfolioWrapper.from_scenarios(scenarios_df)
        frontier = wrapper.variance_frontier(num_portfolios=5)
        assert isinstance(frontier, PortfolioFrontier)
        assert frontier.weights.shape[0] == N

    def test_from_scenarios_with_probabilities(self, scenarios_df):
        probs = np.ones(len(scenarios_df)) / len(scenarios_df)
        wrapper = PortfolioWrapper.from_scenarios(scenarios_df, probabilities=probs)
        frontier = wrapper.variance_frontier(num_portfolios=5)
        assert isinstance(frontier, PortfolioFrontier)
        assert frontier.weights.shape[1] == 5


class TestFromRobustPosterior:
    """PortfolioWrapper.from_robust_posterior factory."""

    def test_from_robust_posterior(self):
        from pyvallocation.bayesian import RobustBayesPosterior

        rng = np.random.default_rng(99)
        prior_mu = MU
        prior_sigma = COV
        # Generate sample data
        sample_data = rng.multivariate_normal(MU, COV, 60)
        sample_mu = sample_data.mean(axis=0)
        sample_sigma = np.cov(sample_data, rowvar=False, bias=False)

        posterior = RobustBayesPosterior.from_niw(
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            t0=10,
            nu0=10,
            sample_mu=sample_mu,
            sample_sigma=sample_sigma,
            n_obs=60,
        )

        wrapper = PortfolioWrapper.from_robust_posterior(posterior)
        frontier = wrapper.robust_lambda_frontier(num_portfolios=5)
        assert isinstance(frontier, PortfolioFrontier)
        assert frontier.weights.shape[0] == N
        assert frontier.weights.shape[1] == 5


# ====================================================================
# Transaction costs
# ====================================================================


class TestTransactionCosts:
    """TransactionCosts flow into frontier methods."""

    def test_transaction_costs_mv(self):
        iw = np.array([0.25, 0.25, 0.25, 0.25])
        mic = np.array([0.001, 0.001, 0.001, 0.001])
        tc = TransactionCosts(initial_weights=iw, market_impact_costs=mic)
        wrapper = PortfolioWrapper.from_moments(MU, COV, costs=tc)
        frontier = wrapper.variance_frontier(num_portfolios=5)
        assert isinstance(frontier, PortfolioFrontier)
        assert frontier.weights.shape[1] == 5

    def test_transaction_costs_cvar(self, scenarios_df):
        iw = np.array([0.25, 0.25, 0.25, 0.25])
        pc = np.array([0.002, 0.002, 0.002, 0.002])
        tc = TransactionCosts(initial_weights=iw, proportional_costs=pc)
        wrapper = PortfolioWrapper.from_scenarios(scenarios_df, costs=tc)
        frontier = wrapper.cvar_frontier(num_portfolios=5, seed=42)
        assert isinstance(frontier, PortfolioFrontier)
        assert frontier.weights.shape[1] == 5


# ====================================================================
# Per-method overrides
# ====================================================================


class TestPerMethodOverrides:
    """constraints= and costs= kwargs on frontier methods override wrapper state."""

    def test_per_method_constraints_override(self):
        # Wrapper uses defaults (long-only, sum=1)
        wrapper = PortfolioWrapper.from_moments(MU, COV)
        # Override with tight upper bound at method level
        c = Constraints(bounds=(0, 0.35))
        frontier = wrapper.variance_frontier(num_portfolios=5, constraints=c)
        assert np.all(frontier.weights <= 0.35 + 1e-6)

    def test_per_method_costs_override(self):
        wrapper = PortfolioWrapper.from_moments(MU, COV)
        iw = np.array([0.25, 0.25, 0.25, 0.25])
        mic = np.array([0.001, 0.001, 0.001, 0.001])
        tc = TransactionCosts(initial_weights=iw, market_impact_costs=mic)
        # Pass costs at method level (wrapper has no costs set)
        frontier = wrapper.variance_frontier(num_portfolios=5, costs=tc)
        assert isinstance(frontier, PortfolioFrontier)
        assert frontier.weights.shape[1] == 5


# ====================================================================
# Budget risk parity
# ====================================================================


class TestBudgetRiskParity:
    """risk_budgets parameter in relaxed_risk_parity methods."""

    def test_budget_risk_parity_custom(self):
        wrapper = PortfolioWrapper.from_moments(MU, COV)
        budgets = np.array([0.5, 0.2, 0.2, 0.1])

        w_custom, ret_custom, vol_custom, diag_custom = (
            wrapper.relaxed_risk_parity_portfolio_with_diagnostics(
                lambda_reg=0.0,
                target_multiplier=None,
                risk_budgets=budgets,
            )
        )

        w_erc, ret_erc, vol_erc, diag_erc = (
            wrapper.relaxed_risk_parity_portfolio_with_diagnostics(
                lambda_reg=0.0,
                target_multiplier=None,
            )
        )

        # Custom budgets should produce different allocations vs ERC
        assert not np.allclose(w_custom.values, w_erc.values, atol=1e-4)

    def test_budget_risk_parity_normalizes(self):
        wrapper = PortfolioWrapper.from_moments(MU, COV)
        # Non-unit budgets that should auto-normalize
        budgets = np.array([5.0, 2.0, 2.0, 1.0])
        w, ret, vol, diag = (
            wrapper.relaxed_risk_parity_portfolio_with_diagnostics(
                lambda_reg=0.0,
                target_multiplier=None,
                risk_budgets=budgets,
            )
        )
        # Should not raise - auto-normalization handles it
        assert np.isfinite(ret)
        assert np.isfinite(vol)
        assert np.all(np.isfinite(w.values))


# ====================================================================
# PortfolioFrontier validation
# ====================================================================


class TestPortfolioFrontierValidation:
    """Shape and content validation on PortfolioFrontier."""

    def test_frontier_returns_and_risks_shape(self):
        wrapper = PortfolioWrapper.from_moments(MU, COV)
        frontier = wrapper.variance_frontier(num_portfolios=7)
        assert frontier.returns.shape == (7,)
        assert frontier.risks.shape == (7,)
        assert frontier.risk_measure == "Volatility"

    def test_frontier_min_risk_returns_tuple(self):
        wrapper = PortfolioWrapper.from_moments(MU, COV)
        frontier = wrapper.variance_frontier(num_portfolios=10)
        w, ret, risk = frontier.min_risk()
        assert isinstance(w, pd.Series)
        assert np.isfinite(ret)
        assert np.isfinite(risk)
        assert risk <= np.max(frontier.risks) + 1e-10

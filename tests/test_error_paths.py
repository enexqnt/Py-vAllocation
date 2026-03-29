"""Tests for error paths and exception handling across the library."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyvallocation import (
    AssetsDistribution,
    InfeasibleOptimizationError,
    PortfolioWrapper,
)
from pyvallocation.bayesian import NIWPosterior
from pyvallocation.moments import _labels
from pyvallocation.utils.projection import (
    convert_scenarios_simple_to_compound,
    simple2log,
)

# ── conftest constants ──────────────────────────────────────────────
N_ASSETS = 4
T_SCENARIOS = 60


# ====================================================================
# Infeasible optimisation errors on PortfolioFrontier targets
# ====================================================================


def _make_mv_frontier(sample_mu, sample_cov, num_portfolios=10):
    """Helper to build a small mean-variance frontier."""
    dist = AssetsDistribution(mu=sample_mu, cov=sample_cov)
    wrapper = PortfolioWrapper(dist)
    wrapper.set_constraints({"long_only": True, "total_weight": 1.0})
    return wrapper.variance_frontier(num_portfolios=num_portfolios)


class TestInfeasibleOptimizationErrorMinVariance:
    """Pass an impossibly high return target to MeanVariance._solve_target.

    CVXOPT may raise either InfeasibleOptimizationError (our wrapper) or
    a low-level ValueError/RuntimeError depending on internal solver state.
    We accept any of these as evidence that the infeasible problem is rejected.
    """

    def test_infeasible_optimization_error_on_min_variance(
        self, sample_mu, sample_cov
    ):
        from pyvallocation.optimization import MeanVariance
        from pyvallocation.utils.constraints import build_G_h_A_b

        G, h, A, b = build_G_h_A_b(N_ASSETS, long_only=True, total_weight=1.0)
        optimizer = MeanVariance(sample_mu, sample_cov, G, h, A, b)
        # Demand a return that exceeds the maximum possible return
        impossible_target = float(np.max(sample_mu)) + 100.0
        with pytest.raises((InfeasibleOptimizationError, RuntimeError, ValueError)):
            optimizer.efficient_portfolio(impossible_target)


class TestInfeasibleOnRiskTarget:
    """frontier.portfolio_at_risk_target with impossibly small max_risk."""

    def test_infeasible_optimization_error_on_risk_target(
        self, sample_mu, sample_cov
    ):
        frontier = _make_mv_frontier(sample_mu, sample_cov)
        impossible_max_risk = -999.0  # negative risk is impossible
        with pytest.raises(InfeasibleOptimizationError):
            frontier.portfolio_at_risk_target(impossible_max_risk)


class TestInfeasibleOnReturnTarget:
    """frontier.portfolio_at_return_target with impossibly high min_return."""

    def test_infeasible_optimization_error_on_return_target(
        self, sample_mu, sample_cov
    ):
        frontier = _make_mv_frontier(sample_mu, sample_cov)
        impossible_min_return = float(np.max(frontier.returns)) + 100.0
        with pytest.raises(InfeasibleOptimizationError):
            frontier.portfolio_at_return_target(impossible_min_return)


# ====================================================================
# NIW double-update raises RuntimeError
# ====================================================================


class TestNIWDoubleUpdateRaises:
    """Calling NIWPosterior.update() twice should raise RuntimeError."""

    def test_niw_double_update_raises(self, sample_mu, sample_cov):
        niw = NIWPosterior(
            prior_mu=sample_mu,
            prior_sigma=sample_cov,
            t0=30,
            nu0=30,
        )
        niw.update(sample_mu=sample_mu, sample_sigma=sample_cov, n_obs=T_SCENARIOS)
        with pytest.raises(RuntimeError, match="already computed"):
            niw.update(
                sample_mu=sample_mu, sample_sigma=sample_cov, n_obs=T_SCENARIOS
            )


# ====================================================================
# _labels alignment mismatch
# ====================================================================


class TestLabelsAlignmentMismatchRaises:
    """_labels with mismatched pandas inputs should raise ValueError."""

    def test_labels_alignment_mismatch_raises(self):
        s1 = pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"])
        s2 = pd.Series([4.0, 5.0, 6.0], index=["X", "Y", "Z"])
        with pytest.raises(ValueError, match="label orderings differ"):
            _labels(s1, s2)


# ====================================================================
# simple2log with returns <= -1
# ====================================================================


class TestSimple2LogBelowMinus1:
    """simple2log should raise ValueError when 1+mu <= 0."""

    def test_simple2log_returns_below_minus1_raises(self):
        bad_mu = np.array([-1.5, 0.05, 0.03])
        cov = np.eye(3) * 0.01
        with pytest.raises(ValueError, match="must be > -1"):
            simple2log(bad_mu, cov)


# ====================================================================
# convert_scenarios_simple_to_compound with scenarios <= -1
# ====================================================================


class TestConvertScenariosSimpleToCompoundRaises:
    """convert_scenarios_simple_to_compound should reject returns <= -1."""

    def test_convert_scenarios_simple_to_compound_raises(self):
        bad_scenarios = np.array([[-1.0, 0.01], [0.02, -1.5]])
        with pytest.raises(ValueError, match="must be > -1"):
            convert_scenarios_simple_to_compound(bad_scenarios)

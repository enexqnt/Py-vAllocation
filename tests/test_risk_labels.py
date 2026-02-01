import numpy as np
import pandas as pd

from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper


def _basic_wrapper_with_scenarios(T: int = 200, N: int = 4) -> PortfolioWrapper:
    rng = np.random.default_rng(0)
    scenarios = rng.normal(0.001, 0.02, size=(T, N))
    mu = scenarios.mean(axis=0)
    cov = np.cov(scenarios, rowvar=False)
    dist = AssetsDistribution(mu=mu, cov=cov, scenarios=scenarios)
    port = PortfolioWrapper(dist)
    port.set_constraints({"long_only": True, "total_weight": 1.0})
    return port


def test_mean_cvar_frontier_exposes_volatility_overlay():
    port = _basic_wrapper_with_scenarios()
    frontier = port.cvar_frontier(num_portfolios=8, alpha=0.05)

    assert "Volatility" in frontier.alternate_risks
    assert "CVaR (raw, alpha=0.05)" in frontier.alternate_risks
    vol = frontier.alternate_risks["Volatility"]
    assert vol.shape == frontier.risks.shape

    w, r, risk = frontier.portfolio_at_risk_target(
        max_risk=float(np.max(vol)), risk_label="Volatility"
    )
    assert isinstance(risk, float)
    assert w.shape[0] == port.dist.N


def test_mean_variance_frontier_exposes_cvar_overlay_when_scenarios_present():
    port = _basic_wrapper_with_scenarios()
    frontier = port.variance_frontier(num_portfolios=6)

    cvar_label = "CVaR (alpha=0.05)"
    assert cvar_label in frontier.alternate_risks
    cvar_vals = frontier.alternate_risks[cvar_label]
    assert cvar_vals.shape == frontier.risks.shape

    w, _, risk = frontier.portfolio_at_risk_target(
        max_risk=float(np.max(cvar_vals)), risk_label=cvar_label
    )
    assert isinstance(risk, float)
    assert w.shape[0] == port.dist.N


def test_robust_frontier_includes_volatility_overlay():
    port = _basic_wrapper_with_scenarios()
    frontier = port.robust_lambda_frontier(num_portfolios=4, max_lambda=0.8)

    assert "Volatility" in frontier.alternate_risks
    alt = frontier.alternate_risks["Volatility"]
    assert alt.shape == frontier.risks.shape

    # available_risk_measures should list primary + alternates
    measures = frontier.available_risk_measures()
    assert frontier.risk_measure in measures
    assert "Volatility" in measures


def test_risk_percentile_selection_matches_index():
    port = _basic_wrapper_with_scenarios()
    frontier = port.variance_frontier(num_portfolios=5)

    idx = frontier.index_at_risk_percentile(0.5, risk_label="Volatility")
    w, _, _ = frontier.portfolio_at_risk_percentile(0.5, risk_label="Volatility")
    assert np.allclose(w.values, frontier.weights[:, idx])

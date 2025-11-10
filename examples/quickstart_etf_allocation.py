"""
Concise ETF allocation quickstart.

The workflow stays end-to-end (data → views → frontiers → ensemble → trades) while
keeping the finance steps explicit and fixing earlier modelling slips:
- Black-Litterman risk aversion is rescaled for weekly inputs.
- Entropy-pooling volatility caps respect realised volatility.
- Every optimisation selects the best portfolio below a 12% annualised-vol cap.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyvallocation.discrete_allocation import discretize_weights
from pyvallocation.ensembles import assemble_portfolio_ensemble, make_portfolio_spec
from pyvallocation.moments import estimate_moments, posterior_moments_niw
from pyvallocation.plotting import plot_frontiers_grid
from pyvallocation.portfolioapi import AssetsDistribution, PortfolioFrontier
from pyvallocation.utils.projection import (
    convert_scenarios_compound_to_simple,
    log2simple,
    project_mean_covariance,
    project_scenarios,
)
from pyvallocation.views import BlackLittermanProcessor, FlexibleViewsProcessor

DATA_PATH = ROOT / "examples" / "ETF_prices.csv"
OUTPUT_DIR = ROOT / "output"
VOL_TARGET = 0.12  # annualised volatility target for selection
RISK_FREE = 0.01


def load_weekly_returns() -> Tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True).ffill()
    weekly_prices = prices.resample("W-FRI").last().dropna(how="all")
    weekly_returns = np.log(weekly_prices).diff().dropna()
    rename = lambda c: c.replace(" ", "_")
    return weekly_returns.rename(columns=rename), weekly_prices.rename(columns=rename)


def project_one_year(mu: pd.Series, sigma: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    mu_h, sigma_h = project_mean_covariance(mu, sigma, annualization_factor=52)
    return log2simple(mu_h, sigma_h)


def scaled_black_litterman(mu_prior: pd.Series, sigma_prior: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    views = {"SPY": 0.0003, ("SPY", "TLT"): 0.0001}  # modest weekly tilts
    processor = BlackLittermanProcessor(
        prior_cov=sigma_prior,
        prior_mean=mu_prior,
        mean_views=views,
        risk_aversion=2.5 / 52.0,  # scale for weekly units
        tau=0.05,
    )
    mu_bl, sigma_bl = processor.get_posterior()
    return (
        pd.Series(mu_bl, index=mu_prior.index),
        pd.DataFrame(sigma_bl, index=mu_prior.index, columns=mu_prior.index),
    )


def entropy_pool_posterior(weekly_returns: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame, np.ndarray]:
    ann_to_week = lambda x: x / np.sqrt(52.0)
    sample_vol = weekly_returns.std(ddof=0)
    mean_views = {
        ("SPY", "TLT"): (">=", 0.0005),
        ("SPY", "GLD"): (">=", 0.0002),
        "GLD": (">=", 0.0001),
        "DBC": (">=", 0.00005),
    }
    vol_caps = {
        asset: ("<=", max(ann_to_week(target), sample_vol.get(asset, 0.0) * 1.05))
        for asset, target in {"SPY": 0.23, "TLT": 0.18, "GLD": 0.22, "DBC": 0.28}.items()
    }
    processor = FlexibleViewsProcessor(
        prior_returns=weekly_returns,
        mean_views=mean_views,
        vol_views=vol_caps,
        corr_views={("SPY", "TLT"): ("<=", -0.10)},
        sequential=True,
        random_state=42,
    )
    mu_ep, sigma_ep = processor.get_posterior()
    q_ep = processor.get_posterior_probabilities().reshape(-1)
    columns = weekly_returns.columns
    return (
        pd.Series(mu_ep, index=columns),
        pd.DataFrame(sigma_ep, index=columns, columns=columns),
        q_ep,
    )


def annual_scenarios(weekly_returns: pd.DataFrame, probs: np.ndarray, *, horizon_weeks: int = 52, n_sim: int = 8000) -> pd.DataFrame:
    sums = project_scenarios(
        weekly_returns.values,
        investment_horizon=horizon_weeks,
        p=probs,
        n_simulations=n_sim,
    )
    return pd.DataFrame(convert_scenarios_compound_to_simple(sums), columns=weekly_returns.columns)


def vol_capped_selector(cov: pd.DataFrame, max_vol: float) -> Callable[[PortfolioFrontier], pd.Series]:
    cov = cov.copy()

    def _select(frontier: PortfolioFrontier) -> pd.Series:
        names = frontier.asset_names or list(cov.index)
        sigma = cov.loc[names, names].to_numpy(dtype=float)
        weights = np.asarray(frontier.weights, dtype=float)
        vols = np.sqrt(np.sum((weights.T @ sigma) * weights.T, axis=1))
        idx_feasible = np.where(vols <= max_vol + 1e-6)[0]
        if idx_feasible.size:
            best = idx_feasible[np.argmax(frontier.returns[idx_feasible])]
        else:
            best = int(np.argmin(vols))
        return pd.Series(weights[:, best], index=names, name=f"Max return <= {max_vol:.0%} vol")

    return _select


def enforce_cvar_convexity(frontier: PortfolioFrontier) -> PortfolioFrontier:
    if "CVaR" not in frontier.risk_measure:
        return frontier
    returns = np.asarray(frontier.returns, float)
    risks = np.asarray(frontier.risks, float)
    order = np.argsort(returns)
    r_sorted, cvar_sorted = returns[order], risks[order]
    idx_stack: list[int] = []
    for i, x in enumerate(r_sorted):
        while len(idx_stack) >= 2:
            i1, i2 = idx_stack[-2], idx_stack[-1]
            cross = (r_sorted[i2] - r_sorted[i1]) * (cvar_sorted[i] - cvar_sorted[i1]) - (
                cvar_sorted[i2] - cvar_sorted[i1]
            ) * (x - r_sorted[i1])
            if cross <= 1e-10:
                idx_stack.pop()
            else:
                break
        idx_stack.append(i)
    keep = order[idx_stack]
    if keep.size == returns.size:
        return frontier
    return PortfolioFrontier(
        weights=frontier.weights[:, keep],
        returns=returns[keep],
        risks=risks[keep],
        risk_measure=frontier.risk_measure,
        asset_names=frontier.asset_names,
    )


def build_specs(
    weekly_returns: pd.DataFrame,
    annual_covariances: Dict[str, pd.DataFrame],
    annual_mu: Dict[str, pd.Series],
    ep_cov_1y: pd.DataFrame,
    ep_mu_1y: pd.Series,
    ep_scenarios: pd.DataFrame,
) -> Sequence:
    long_only = {"long_only": True, "total_weight": 1.0, "bounds": (None, 0.6)}
    projection = {"annualization_factor": 52, "log_to_simple": True}
    common_mv = {"num_portfolios": 31, "constraints": long_only}

    spec_defs = [
        {
            "name": "Shrinkage_MV",
            "returns": weekly_returns,
            "mean_estimator": "james_stein",
            "cov_estimator": "oas",
            "selector": vol_capped_selector(annual_covariances["Shrinkage"], VOL_TARGET),
            "optimiser": "mean_variance",
            "optimiser_kwargs": common_mv,
            "projection": projection,
            "metadata": {"model": "Shrinkage", "horizon": "1Y"},
        },
        {
            "name": "NLS_MV",
            "returns": weekly_returns,
            "mean_estimator": "james_stein",
            "cov_estimator": "nls",
            "selector": vol_capped_selector(annual_covariances["NLS"], VOL_TARGET),
            "optimiser": "mean_variance",
            "optimiser_kwargs": common_mv,
            "projection": projection,
            "metadata": {"model": "Shrinkage (NLS)", "horizon": "1Y"},
        },
        {
            "name": "Robust_RRP",
            "returns": weekly_returns,
            "mean_estimator": "huber",
            "cov_estimator": "tyler",
            "cov_kwargs": {"shrinkage": 0.1},
            "selector": vol_capped_selector(annual_covariances["Robust"], VOL_TARGET),
            "optimiser": "rrp",
            "optimiser_kwargs": {"num_portfolios": 11, "max_multiplier": 1.4, "lambda_reg": 0.2, "constraints": long_only},
            "projection": projection,
            "metadata": {"model": "Robust (Huber+Tyler)", "horizon": "1Y"},
        },
        {
            "name": "Robust_Bayes_MV",
            "distribution": AssetsDistribution(mu=annual_mu["RobustBayes"], cov=annual_covariances["RobustBayes"]),
            "selector": vol_capped_selector(annual_covariances["RobustBayes"], VOL_TARGET),
            "optimiser": "mean_variance",
            "optimiser_kwargs": common_mv,
            "metadata": {"model": "Robust-Bayes (NIW)", "horizon": "1Y"},
        },
        {
            "name": "BL_MV",
            "distribution": AssetsDistribution(mu=annual_mu["BL"], cov=annual_covariances["BL"]),
            "selector": vol_capped_selector(annual_covariances["BL"], VOL_TARGET),
            "optimiser": "mean_variance",
            "optimiser_kwargs": common_mv,
            "metadata": {"model": "Black-Litterman", "horizon": "1Y"},
        },
        {
            "name": "EP_MV",
            "distribution": AssetsDistribution(mu=ep_mu_1y, cov=ep_cov_1y),
            "selector": vol_capped_selector(ep_cov_1y, VOL_TARGET),
            "optimiser": "mean_variance",
            "optimiser_kwargs": common_mv,
            "metadata": {"model": "Entropy Pooling", "horizon": "1Y"},
        },
        {
            "name": "EP_CVaR",
            "distribution": AssetsDistribution(scenarios=ep_scenarios),
            "use_scenarios": True,
            "selector": vol_capped_selector(ep_cov_1y, VOL_TARGET),
            "optimiser": "cvar",
            "optimiser_kwargs": {"num_portfolios": 21, "alpha": 0.05, "constraints": long_only},
            "metadata": {"model": "CVaR (EP scenarios)", "horizon": "1Y"},
        },
    ]

    return [make_portfolio_spec(**spec) for spec in spec_defs]


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Quickstart ETF allocation (concise)")

    def step(msg: str) -> None:
        elapsed = time.perf_counter() - start
        print(f"[{elapsed:6.2f}s] {msg}")

    step("Loading weekly data...")
    weekly_returns, weekly_prices = load_weekly_returns()

    step("Estimating base and robust moments...")
    mu_js, cov_oas = estimate_moments(weekly_returns, mean_estimator="james_stein", cov_estimator="oas")
    mu_nls, cov_nls = estimate_moments(weekly_returns, mean_estimator="james_stein", cov_estimator="nls")
    mu_huber, cov_tyler = estimate_moments(
        weekly_returns,
        mean_estimator="huber",
        cov_estimator="tyler",
        cov_kwargs={"shrinkage": 0.1},
    )
    mu_rb, cov_rb = posterior_moments_niw(
        prior_mu=mu_js,
        prior_sigma=cov_oas,
        t0=8,
        nu0=max(len(mu_js) + 2, 6),
        sample_mu=mu_huber,
        sample_sigma=cov_tyler,
        n_obs=weekly_returns.shape[0],
    )

    step("Applying macro views (BL & EP)...")
    mu_bl, cov_bl = scaled_black_litterman(mu_js, cov_oas)
    mu_ep, cov_ep, q_ep = entropy_pool_posterior(weekly_returns)
    ep_mu_1y, ep_cov_1y = project_one_year(mu_ep, cov_ep)

    step("Projecting all moments to 1-year simple returns...")
    annual_mu: Dict[str, pd.Series] = {}
    annual_cov: Dict[str, pd.DataFrame] = {}
    for label, (mu, cov) in {
        "Shrinkage": (mu_js, cov_oas),
        "NLS": (mu_nls, cov_nls),
        "Robust": (mu_huber, cov_tyler),
        "RobustBayes": (mu_rb, cov_rb),
        "BL": (mu_bl, cov_bl),
    }.items():
        mu_1y, cov_1y = project_one_year(mu, cov)
        annual_mu[label] = mu_1y
        annual_cov[label] = cov_1y

    step("Building annual scenarios for CVaR...")
    ep_scenarios = annual_scenarios(weekly_returns, q_ep)

    step("Configuring optimisation specs...")
    specs = build_specs(
        weekly_returns=weekly_returns,
        annual_covariances=annual_cov,
        annual_mu=annual_mu,
        ep_cov_1y=ep_cov_1y,
        ep_mu_1y=ep_mu_1y,
        ep_scenarios=ep_scenarios,
    )

    step("Solving frontiers and assembling ensembles...")
    ensemble = assemble_portfolio_ensemble(specs, ensemble=("average", "stack"), stack_folds=3)
    ensemble.frontiers = {k: enforce_cvar_convexity(v) for k, v in ensemble.frontiers.items()}

    selected = ensemble.selections.round(4)
    print("\nRepresentative portfolios (vol-capped):")
    print(selected)

    stacked = ensemble.stacked
    if stacked is None:
        raise RuntimeError("Stacked ensemble unavailable.")
    stacked = stacked.rename("Stacked")
    print("\nStacked allocation (top 5):")
    print(stacked.sort_values(ascending=False).head())

    step("Evaluating trailing metrics...")
    weekly_simple = weekly_prices.pct_change().dropna()
    aligned = stacked.reindex(weekly_simple.columns, fill_value=0.0)
    portfolio_weekly = weekly_simple.dot(aligned)
    annual_return = (1 + portfolio_weekly.mean()) ** 52 - 1
    annual_vol = portfolio_weekly.std(ddof=0) * np.sqrt(52)
    sharpe = (annual_return - RISK_FREE) / annual_vol if annual_vol > 0 else np.nan
    print(f"\nStacked trailing metrics: return={annual_return:.2%}, vol={annual_vol:.2%}, Sharpe≈{sharpe:.2f} (rf={RISK_FREE:.0%})")

    step("Drawing combined frontier plot...")
    fig, _ = plot_frontiers_grid(
        ensemble.frontiers,
        by=lambda _label, frontier: "CVaR (alpha=5%)" if "CVaR" in (frontier.risk_measure or "") else "Volatility",
        highlight=("min_risk", "max_return"),
        risk_free_rate=RISK_FREE,
        legend=True,
        figsize=(13, 5),
    )
    fig.savefig(OUTPUT_DIR / "frontiers.png", dpi=150)
    plt.close(fig)

    step("Converting to discrete trades...")
    allocation = discretize_weights(
        weights=stacked,
        latest_prices=weekly_prices.iloc[-1],
        total_value=10_000_000,
    )
    trade_summary = pd.DataFrame(
        {
            "Target Weight": stacked.round(4),
            "Achieved Weight": allocation.achieved_weights.reindex(stacked.index).round(4),
            "Shares": pd.Series(allocation.shares, dtype=int),
            "Market Value": pd.Series(allocation.shares) * weekly_prices.iloc[-1],
        }
    )
    print("\nDiscrete allocation (>0 shares):")
    print(trade_summary[trade_summary["Shares"] > 0].sort_values("Market Value", ascending=False))
    print(f"Residual cash: {allocation.leftover_cash:,.2f} | Tracking error RMSE: {allocation.tracking_error:.6f}")

    step("Persisting key artefacts...")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    selected.to_csv(OUTPUT_DIR / "selected_weights.csv")
    stacked.to_csv(OUTPUT_DIR / "stacked_weights.csv")
    if ensemble.average is not None:
        ensemble.average.to_csv(OUTPUT_DIR / "average_weights.csv")

    step("Stress-testing with adverse flexible views...")
    stress_processor = FlexibleViewsProcessor(
        prior_returns=weekly_returns,
        mean_views={"SPY": ("<=", -0.0002)},
        corr_views={("SPY", "TLT"): (">=", -0.02)},
        sequential=True,
        random_state=7,
    )
    mu_stress, sigma_stress = stress_processor.get_posterior()
    mu_stress_1y, sigma_stress_1y = project_one_year(
        pd.Series(mu_stress, index=weekly_returns.columns),
        pd.DataFrame(sigma_stress, index=weekly_returns.columns, columns=weekly_returns.columns),
    )
    stress_cov = sigma_stress_1y.reindex(stacked.index, axis=0).reindex(stacked.index, axis=1)
    stress_return = stacked.values @ mu_stress_1y.reindex(stacked.index).values
    stress_vol = np.sqrt(stacked.values @ stress_cov.to_numpy(dtype=float) @ stacked.values)
    print(f"Stress-case (weaker hedges) annual return={stress_return:.2%}, vol={stress_vol:.2%}")

    step("Quickstart completed.")


if __name__ == "__main__":
    main()

"""
Comprehensive ETF allocation quickstart.

This script mirrors ``docs/tutorials/quickstart_etf_allocation.rst`` and executes an end-to-end
workflow:

1. Load ETF prices and compute weekly returns.
2. Estimate moments via shrinkage, robust methods, and a Black-Litterman view.
3. Project the statistics to a 1-year horizon.
4. Build optimised frontiers for each specification.
5. Assemble stacked/average ensemble portfolios.
6. Generate plots and convert the final allocation to discrete share counts.

Run with:

    python examples/quickstart_etf_allocation.py

Outputs are written to the ``output/`` directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyvallocation.ensembles import assemble_portfolio_ensemble, make_portfolio_spec
from pyvallocation.moments import estimate_moments, posterior_moments_niw
from pyvallocation.portfolioapi import AssetsDistribution, PortfolioFrontier
from pyvallocation.utils.projection import (
    log2simple,
    project_mean_covariance,
    project_scenarios,
    convert_scenarios_compound_to_simple,
)
from pyvallocation.views import BlackLittermanProcessor, FlexibleViewsProcessor

OUTPUT_DIR = Path("output")
DATA_PATH = ROOT_DIR / "examples" / "ETF_prices.csv"


def load_weekly_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Sample ETF data not found at {DATA_PATH}. "
            "Ensure the repository data files are available."
        )
    prices = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True).ffill()
    weekly_prices = prices.resample("W-FRI").last().dropna(how="all")
    # Use log returns for estimation consistency; convert later for reporting
    weekly_returns = np.log(weekly_prices).diff().dropna()
    weekly_returns = weekly_returns.rename(columns=lambda c: c.replace(" ", "_"))
    weekly_prices = weekly_prices.rename(columns=lambda c: c.replace(" ", "_"))
    return weekly_returns, weekly_prices


def project_to_horizon(mu: pd.Series, sigma: pd.DataFrame, annualisation: int):
    mu_proj, sigma_proj = project_mean_covariance(mu, sigma, annualization_factor=annualisation)
    return log2simple(mu_proj, sigma_proj)


def build_black_litterman(mu_prior: pd.Series, sigma_prior: pd.DataFrame):
    """Encode a simple macro view: SPY outperforms TLT by 50 bps."""
    if "SPY" not in mu_prior.index or "TLT" not in mu_prior.index:
        raise ValueError("Expected ETFs 'SPY' and 'TLT' in the dataset for the macro view.")
    # Views are specified in the same units as the prior: weekly log returns.
    # We express modest tilts to avoid unrealistic annualised expectations.
    processor = BlackLittermanProcessor(
        prior_cov=sigma_prior,
        prior_mean=mu_prior,
        mean_views={
            "SPY": 0.0003,                # ~1.6%/year absolute tilt for SPY
            ("SPY", "TLT"): 0.0001,       # ~0.5%/year relative tilt vs TLT
        },
        risk_aversion=2.5,
        tau=0.05,
    )
    mu_bl, sigma_bl = processor.get_posterior()
    return (
        pd.Series(mu_bl, index=mu_prior.index),
        pd.DataFrame(sigma_bl, index=mu_prior.index, columns=mu_prior.index),
    )


def long_term_views_nov2025(asset_names: pd.Index) -> tuple[dict, dict, dict]:
    """Return long-term macro views for Nov 2025 in weekly-log units.

    Views are deliberately modest and diversified:
    - Equities expected to outperform duration over the long run.
    - A mild positive drift for gold and broad commodities as inflation hedges.
    - Moderately negative stock-bond correlation; bounded commodity and gold vols.

    Returns:
        mean_views_bl: Mapping used with BL (equality mean views).
        mean_views_ep: Mapping used with EP (equalities/inequalities supported).
        other_ep_views: Dict containing 'vol_views' and 'corr_views'.
    """
    names = set(asset_names)
    required = {"SPY", "TLT", "GLD", "DBC"}
    if not required.issubset(names):
        missing = ", ".join(sorted(required - names))
        raise ValueError(f"ETF universe must include {missing} to use long-term views.")

    # Weekly log-return tilts roughly corresponding to annualised drifts
    # of ~3.0% (equity over bonds), ~1.2% (equity over gold), and small
    # positive absolute drifts for gold/commodities.
    mean_views_bl = {
        ("SPY", "TLT"): 0.0006,  # ~3%/year
        ("SPY", "GLD"): 0.00023,  # ~1.2%/year
        "GLD": 0.00015,           # ~0.8%/year
        "DBC": 0.00010,           # ~0.5%/year
    }

    # EP accepts inequalities as well; keep the same direction but allow flexibility.
    mean_views_ep = {
        ("SPY", "TLT"): (">=", 0.0005),
        ("SPY", "GLD"): (">=", 0.0002),
        "GLD": (">=", 0.0001),
        "DBC": (">=", 0.00005),
    }

    # Volatility bounds in weekly units (~annual targets divided by sqrt(52))
    ann_to_week = lambda x: x / np.sqrt(52.0)
    vol_views = {
        "SPY": ("<=", ann_to_week(0.23)),
        "TLT": ("<=", ann_to_week(0.18)),
        "GLD": ("<=", ann_to_week(0.22)),
        "DBC": ("<=", ann_to_week(0.28)),
    }
    corr_views = {("SPY", "TLT"): ("<=", -0.10)}
    return mean_views_bl, mean_views_ep, {"vol_views": vol_views, "corr_views": corr_views}


def build_opinion_pooling(weekly_log_returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
    """Apply entropy pooling to weekly log scenarios using long-term views."""
    mean_bl, mean_ep, extra = long_term_views_nov2025(weekly_log_returns.columns)
    processor = FlexibleViewsProcessor(
        prior_returns=weekly_log_returns,
        mean_views=mean_ep,
        vol_views=extra["vol_views"],
        corr_views=extra["corr_views"],
        sequential=True,
        random_state=42,
        num_scenarios=0,  # not used when prior_returns provided
    )
    mu_post, sigma_post = processor.get_posterior()
    q = processor.get_posterior_probabilities().reshape(-1)
    return (
        pd.Series(mu_post, index=weekly_log_returns.columns),
        pd.DataFrame(sigma_post, index=weekly_log_returns.columns, columns=weekly_log_returns.columns),
        q,
    )


def build_annual_simple_scenarios(
    weekly_log_returns: pd.DataFrame,
    posterior_probs: np.ndarray,
    *,
    horizon_weeks: int = 52,
    n_simulations: int = 8000,
) -> pd.DataFrame:
    """Generate 1Y simple-return scenarios consistent with EP probabilities.

    We draw ``horizon_weeks`` weekly log-return rows with replacement according to
    the posterior probability vector, sum them (log-additive), then convert to
    simple returns via ``exp(sum) - 1``. This maintains coherence with CVaR
    modelling where risk and return are assessed on the same annual horizon.
    """
    sums = project_scenarios(
        weekly_log_returns.values,
        investment_horizon=horizon_weeks,
        p=posterior_probs,
        n_simulations=n_simulations,
    )
    annual_simple = convert_scenarios_compound_to_simple(sums)
    return pd.DataFrame(annual_simple, columns=weekly_log_returns.columns)


def plot_frontier_weights(frontiers: dict, *, outfile: Path, max_specs: int = 4) -> None:
    """Stacked area chart of weights along each frontier for up to `max_specs` specs."""
    import matplotlib.pyplot as plt

    selected = list(frontiers.items())[:max_specs]
    n = len(selected)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(9, 2.5 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (name, f) in zip(axes, selected):
        W = np.asarray(f.weights, dtype=float)
        x = np.arange(W.shape[1])
        labels = f.asset_names or [f"A{i}" for i in range(W.shape[0])]
        ax.stackplot(x, W, labels=labels)
        ax.set_title(f"Frontier Weights - {name}")
        ax.set_ylabel("Weight")
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="upper right", ncols=len(labels))
    axes[-1].set_xlabel("Frontier index (low -> high risk)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    plt.switch_backend("Agg")
    ensure_output_dir()

    start_time = time.perf_counter()

    def log(message: str) -> None:
        elapsed = time.perf_counter() - start_time
        print(f"[quickstart {elapsed:6.2f}s] {message}")

    log("Loading and preprocessing price history...")
    weekly_returns, weekly_prices = load_weekly_data()

    # Moment estimation
    log("Estimating shrinkage moments (James-Stein + OAS)...")
    mu_shrink, sigma_oas = estimate_moments(
        weekly_returns,
        mean_estimator="james_stein",
        cov_estimator="oas",
    )
    log("Estimating robust moments (Huber + Tyler)...")
    mu_huber, sigma_tyler = estimate_moments(
        weekly_returns,
        mean_estimator="huber",
        cov_estimator="tyler",
        cov_kwargs={"shrinkage": 0.1},
    )
    log("Computing Robust Bayesian (NIW) posterior...")
    n_obs = weekly_returns.shape[0]
    n_assets = weekly_returns.shape[1]
    # Use shrinkage (JS+OAS) as prior and robust (Huber+Tyler) as sample evidence
    mu_rb, sigma_rb = posterior_moments_niw(
        prior_mu=mu_shrink,
        prior_sigma=sigma_oas,
        t0=8,                 # prior pseudo-observations (~two months)
        nu0=max(n_assets + 2, 6),  # prior df, weakly-informative
        sample_mu=mu_huber,
        sample_sigma=sigma_tyler,
        n_obs=n_obs,
    )
    log("Building Black-Litterman posterior with macro view...")
    mu_bl, sigma_bl = build_black_litterman(mu_shrink, sigma_oas)
    log("Opinion pooling via entropy pooling (Flexible Views)...")
    mu_ep, sigma_ep, q_ep = build_opinion_pooling(weekly_returns)
    # Project EP posterior to 1Y and convert to simple returns for consistency
    ep_mu_1y, ep_sigma_1y = log2simple(*project_mean_covariance(mu_ep, sigma_ep, annualization_factor=52))

    # Projection
    log("Projecting moments to the 1-year horizon...")
    horizon_mu: Dict[str, pd.Series] = {}
    horizon_sigma: Dict[str, pd.DataFrame] = {}
    for label, (mu, sigma) in {
        "Shrinkage": (mu_shrink, sigma_oas),
        "Robust": (mu_huber, sigma_tyler),
        "RobustBayes": (mu_rb, sigma_rb),
        "BL": (mu_bl, sigma_bl),
    }.items():
        horizon_mu[label], horizon_sigma[label] = project_to_horizon(mu, sigma, annualisation=52)

    # Ensemble specifications
    # Long-only, fully-invested with an upper cap to avoid degeneracy
    long_only = {"long_only": True, "total_weight": 1.0, "bounds": (None, 0.6)}
    projection = {"annualization_factor": 52, "log_to_simple": True}
    log("Configuring ensemble specifications...")
    specs = [
        make_portfolio_spec(
            name="Shrinkage_MV",
            returns=weekly_returns,
            mean_estimator="james_stein",
            cov_estimator="oas",
            projection=projection,
            optimiser="mean_variance",
            optimiser_kwargs={"num_portfolios": 21, "constraints": long_only},
            selector="risk_target",
            selector_kwargs={"max_risk": 0.12},
            metadata={"model": "Shrinkage", "horizon": "1Y"},
        ),
        make_portfolio_spec(
            name="NLS_MV",
            returns=weekly_returns,
            mean_estimator="james_stein",
            cov_estimator="nls",
            projection=projection,
            optimiser="mean_variance",
            optimiser_kwargs={"num_portfolios": 21, "constraints": long_only},
            selector="risk_target",
            selector_kwargs={"max_risk": 0.12},
            metadata={"model": "NLS", "horizon": "1Y"},
        ),
        make_portfolio_spec(
            name="Robust_RRP",
            returns=weekly_returns,
            mean_estimator="huber",
            cov_estimator="tyler",
            cov_kwargs={"shrinkage": 0.1},
            projection=projection,
            optimiser="rrp",
            optimiser_kwargs={
                "num_portfolios": 9,
                "max_multiplier": 1.5,
                "lambda_reg": 0.2,
                "constraints": long_only,
            },
            selector="risk_target",
            selector_kwargs={"max_risk": 0.12},
            metadata={"model": "Robust", "horizon": "1Y"},
        ),
        make_portfolio_spec(
            name="Robust_Bayes_MV",
            distribution=AssetsDistribution(mu=horizon_mu["RobustBayes"], cov=horizon_sigma["RobustBayes"]),
            optimiser="mean_variance",
            optimiser_kwargs={"num_portfolios": 21, "constraints": long_only},
            selector="risk_target",
            selector_kwargs={"max_risk": 0.12},
            metadata={"model": "Robust-Bayes (NIW)", "horizon": "1Y"},
        ),
        make_portfolio_spec(
            name="BL_MV",
            distribution=AssetsDistribution(mu=horizon_mu["BL"], cov=horizon_sigma["BL"]),
            optimiser="mean_variance",
            optimiser_kwargs={"num_portfolios": 21, "constraints": long_only},
            selector="risk_target",
            selector_kwargs={"max_risk": 0.12},
            metadata={"model": "Black-Litterman", "horizon": "1Y"},
        ),
        make_portfolio_spec(
            name="EP_MV",
            distribution=AssetsDistribution(mu=ep_mu_1y, cov=ep_sigma_1y),
            optimiser="mean_variance",
            optimiser_kwargs={"num_portfolios": 21, "constraints": long_only},
            selector="risk_target",
            selector_kwargs={"max_risk": 0.12},
            metadata={"model": "Entropy Pooling", "horizon": "1Y"},
        ),
        # Annual-horizon CVaR using EP posterior probabilities to generate scenarios
        make_portfolio_spec(
            name="EP_CVaR",
            distribution=AssetsDistribution(
                scenarios=build_annual_simple_scenarios(weekly_returns, q_ep),
            ),
            use_scenarios=True,
            optimiser="cvar",
            optimiser_kwargs={"num_portfolios": 21, "alpha": 0.05, "constraints": long_only},
            selector="tangency",
            selector_kwargs={"risk_free_rate": 0.01},
            metadata={"model": "CVaR (EP)", "horizon": "1Y"},
        ),
    ]

    log("Solving frontiers and assembling ensemble portfolios...")
    ensemble = assemble_portfolio_ensemble(
        specs,
        ensemble=("average", "stack"),
        stack_folds=3,
    )

    # Ensure CVaR frontiers are convex (guard against numerical kinks)
    def _lower_convex_envelope(x: np.ndarray, y: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        order = np.argsort(x)
        x, y = x[order], y[order]
        stack: list[int] = []
        for i in range(len(x)):
            while len(stack) >= 2:
                i1, i2 = stack[-2], stack[-1]
                cross = (x[i2] - x[i1]) * (y[i] - y[i1]) - (y[i2] - y[i1]) * (x[i] - x[i1])
                if cross <= eps:
                    stack.pop()
                else:
                    break
            stack.append(i)
        # Map back to original indices
        return order[stack]

    fixed_frontiers: Dict[str, PortfolioFrontier] = {}
    for name, f in ensemble.frontiers.items():
        if "CVaR" not in f.risk_measure:
            fixed_frontiers[name] = f
            continue
        idx = _lower_convex_envelope(np.asarray(f.returns, float), np.asarray(f.risks, float))
        if idx.size < f.returns.size:
            fixed_frontiers[name] = PortfolioFrontier(
                weights=f.weights[:, idx],
                returns=f.returns[idx],
                risks=f.risks[idx],
                risk_measure=f.risk_measure,
                asset_names=f.asset_names,
            )
        else:
            fixed_frontiers[name] = f
    ensemble.frontiers = fixed_frontiers

    print("Selected portfolios (per model):")
    print(ensemble.selections.round(4))
    print("\nStacked allocation (top 5 holdings):")
    print(ensemble.stacked.sort_values(ascending=False).head())

    # For trailing metrics we compute on simple returns to aid interpretation.
    weekly_simple = weekly_prices.pct_change().dropna()
    stacked_weights = ensemble.stacked.reindex(weekly_simple.columns, fill_value=0.0)
    portfolio_weekly = weekly_simple.dot(stacked_weights)
    annualised_return = (1.0 + portfolio_weekly.mean()) ** 52 - 1.0
    annualised_vol = portfolio_weekly.std(ddof=0) * np.sqrt(52)
    sharpe = (annualised_return - 0.01) / annualised_vol if annualised_vol > 0 else np.nan
    print(
        f"\nStacked portfolio trailing metrics: "
        f"return={annualised_return:.2%}, vol={annualised_vol:.2%}, Sharpe~{sharpe:.2f} (rf=1%)"
    )

    # Plot the frontiers together
    log("Plotting frontier comparison...")
    fig, ax = plt.subplots(figsize=(8, 5))
    from pyvallocation.plotting import plot_frontiers

    # Plot MV/BL/Robust frontiers together (same risk metric)
    mv_names = [k for k in ensemble.frontiers.keys() if "CVaR" not in k]
    mv_frontiers = {k: ensemble.frontiers[k] for k in mv_names}
    # Merge EP_CVaR into volatility plot by recomputing risk under ep_sigma_1y
    if "EP_CVaR" in ensemble.frontiers:
        f_ep_cvar = ensemble.frontiers["EP_CVaR"]
        W = np.asarray(f_ep_cvar.weights, dtype=float)
        if W.ndim == 2 and W.shape[1] > 0:
            Sigma = np.asarray(ep_sigma_1y, float)
            risks_vol = np.sqrt(np.sum((W.T @ Sigma) * W.T, axis=1))
            mv_frontiers["EP_CVaR (vol)"] = PortfolioFrontier(
                weights=W,
                returns=f_ep_cvar.returns,
                risks=risks_vol,
                risk_measure="Volatility",
                asset_names=f_ep_cvar.asset_names,
            )
    plot_frontiers(mv_frontiers, ax=ax, highlight=())
    ax.set_title("ETF Frontier Comparison - 1Y Horizon (Volatility)")
    ax.set_xlabel("Risk")
    ax.set_ylabel("Expected Return")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "frontiers_vol.png", dpi=150)
    plt.close(fig)

    # Plot CVaR frontiers separately due to different risk units
    if any("CVaR" in k for k in ensemble.frontiers.keys()):
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        cvar_frontiers = {k: v for k, v in ensemble.frontiers.items() if "CVaR" in k}
        plot_frontiers(cvar_frontiers, ax=ax2, highlight=())
        ax2.set_title("ETF Frontier - 1Y Horizon (CVaR)")
        ax2.set_xlabel("CVaR Risk")
        ax2.set_ylabel("Expected Return")
        fig2.tight_layout()
        fig2.savefig(OUTPUT_DIR / "frontiers_cvar.png", dpi=150)
        plt.close(fig2)

        # Backward-compatible composite saved as frontiers.png
        fig3, (axA, axB) = plt.subplots(ncols=2, figsize=(14, 5))
        plot_frontiers(mv_frontiers, ax=axA, highlight=())
        axA.set_title("Volatility")
        axA.set_xlabel("Risk")
        axA.set_ylabel("Expected Return")
        plot_frontiers(cvar_frontiers, ax=axB, highlight=())
        axB.set_title("CVaR (alpha=0.05)")
        axB.set_xlabel("Risk")
        axB.set_ylabel("Expected Return")
        fig3.suptitle("ETF Frontier Comparison - 1Y Horizon")
        fig3.tight_layout()
        fig3.savefig(OUTPUT_DIR / "frontiers.png", dpi=150)
        plt.close(fig3)

    # Plot frontier weights for a subset of specs
    plot_frontier_weights(ensemble.frontiers, outfile=OUTPUT_DIR / "frontier_weights.png", max_specs=4)

    # Discretise to trades
    log("Converting stacked allocation to discrete share counts...")
    latest_prices = weekly_prices.iloc[-1]
    from pyvallocation.discrete_allocation import discretize_weights

    allocation = discretize_weights(
        weights=ensemble.stacked,
        latest_prices=latest_prices,
        total_value=10_000_000,
    )

    print("\nDiscrete allocation (share counts):")
    shares = pd.Series(allocation.shares, dtype=int).reindex(ensemble.stacked.index, fill_value=0)
    summary = pd.DataFrame(
        {
            "Target Weight": ensemble.stacked.round(4),
            "Achieved Weight": allocation.achieved_weights.reindex(ensemble.stacked.index).round(4),
            "Shares": shares.astype(int),
            "Market Value": (shares * latest_prices).round(2),
        }
    )
    print(summary[summary["Shares"] > 0].sort_values("Market Value", ascending=False))
    print(f"Residual cash: {allocation.leftover_cash:,.2f}")
    print(f"Tracking error (RMSE): {allocation.tracking_error:.6f}")

    # Persist artefacts
    log("Saving artefacts to the output directory...")
    ensemble.selections.to_csv(OUTPUT_DIR / "selected_weights.csv")
    ensemble.stacked.to_csv(OUTPUT_DIR / "stacked_weights.csv")
    if ensemble.average is not None:
        ensemble.average.to_csv(OUTPUT_DIR / "average_weights.csv")
    pd.DataFrame.from_dict(ensemble.metadata, orient="index").to_csv(
        OUTPUT_DIR / "ensemble_metadata.csv"
    )

    # Stress-test with flexible views: adverse equity drift and weaker hedging
    log("Stress-testing with adverse flexible views...")
    stress_processor = FlexibleViewsProcessor(
        prior_returns=weekly_returns,
        mean_views={"SPY": ("<=", -0.0002)},  # soft negative drift
        corr_views={("SPY", "TLT"): (">=", -0.02)},  # weaker negative correlation
        sequential=True,
        random_state=7,
    )
    mu_stress, sigma_stress = stress_processor.get_posterior()
    mu_stress_1y, sigma_stress_1y = log2simple(*project_mean_covariance(
        pd.Series(mu_stress, index=weekly_returns.columns),
        pd.DataFrame(sigma_stress, index=weekly_returns.columns, columns=weekly_returns.columns),
        annualization_factor=52,
    ))
    from pyvallocation.portfolioapi import PortfolioWrapper
    stress_frontier = PortfolioWrapper(AssetsDistribution(mu=mu_stress_1y, cov=sigma_stress_1y))
    stress_frontier.set_constraints(long_only)
    stress_mv = stress_frontier.mean_variance_frontier(num_portfolios=21)
    stress_weights, *_ = stress_mv.portfolio_at_risk_target(max_risk=0.12)
    drift = (stress_weights - ensemble.stacked).dropna()
    print("\nStress-test change vs baseline (top 5 by abs delta):")
    print(drift.reindex(drift.abs().sort_values(ascending=False).head().index))

    log("Quickstart completed successfully.")


if __name__ == "__main__":
    main()

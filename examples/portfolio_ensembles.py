"""Blend multiple allocation specifications into a single ensemble portfolio."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from pyvallocation import moments, probabilities
from pyvallocation.ensembles import average_exposures, exposure_stacking
from pyvallocation.portfolioapi import AssetsDistribution, PortfolioFrontier, PortfolioWrapper
from pyvallocation.utils.projection import (
    convert_scenarios_compound_to_simple,
    log2simple,
    project_mean_covariance,
)

DATA_PATH = Path(__file__).with_name("ETF_prices.csv")
INVESTMENT_HORIZON = 52  # weeks
MAX_ANNUALISED_VOL = 0.12


def load_weekly_returns() -> pd.DataFrame:
    prices = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True).ffill().dropna()
    weekly_prices = prices.resample("W-FRI").last().dropna(how="all")
    weekly_log = np.log(weekly_prices).diff().dropna()
    weekly_simple = convert_scenarios_compound_to_simple(weekly_log)
    return pd.DataFrame(weekly_simple, index=weekly_log.index, columns=weekly_log.columns)


def select_portfolio(frontier: PortfolioFrontier, max_vol: float) -> pd.Series:
    weights, expected_return, risk = frontier.portfolio_at_risk_target(max_vol)
    if weights.isna().any():
        weights, expected_return, risk = frontier.get_min_risk_portfolio()
        print("  ├─ Risk target infeasible; using minimum-risk portfolio instead.")
    print(f"  ├─ Selected portfolio: return={expected_return:.4%}, risk={risk:.4%}")
    return weights


def build_frontier(mu: pd.Series, cov: pd.DataFrame) -> PortfolioFrontier:
    mu_hor, cov_hor = project_mean_covariance(mu, cov, annualization_factor=INVESTMENT_HORIZON)
    mu_simple, cov_simple = log2simple(mu_hor, cov_hor)
    wrapper = PortfolioWrapper(AssetsDistribution(mu=mu_simple, cov=cov_simple))
    wrapper.set_constraints({"long_only": True, "total_weight": 1.0})
    return wrapper.mean_variance_frontier(num_portfolios=11)


def main() -> None:
    weekly_returns = load_weekly_returns()
    T = len(weekly_returns)
    print(f"Loaded {T} weekly scenarios.")

    means: Dict[str, pd.Series] = {}
    covs: Dict[str, pd.DataFrame] = {}

    p_uniform = probabilities.generate_uniform_probabilities(T)
    p_exp = probabilities.generate_exp_decay_probabilities(T, half_life=max(T // 8, 10))

    means["uniform"], covs["uniform_raw"] = moments.estimate_sample_moments(weekly_returns, p_uniform)
    means["exp"], covs["exp_raw"] = moments.estimate_sample_moments(weekly_returns, p_exp)

    means["uniform_jorion"] = moments.shrink_mean_jorion(means["uniform"], covs["uniform_raw"], T)
    means["exp_jorion"] = moments.shrink_mean_jorion(means["exp"], covs["exp_raw"], T)

    covs["uniform_lw_cc"] = moments.shrink_covariance_ledoit_wolf(weekly_returns, covs["uniform_raw"], target="constant_correlation")
    covs["uniform_lw_id"] = moments.shrink_covariance_ledoit_wolf(weekly_returns, covs["uniform_raw"], target="identity")
    covs["exp_lw_cc"] = moments.shrink_covariance_ledoit_wolf(weekly_returns, covs["exp_raw"], target="constant_correlation")
    covs["exp_lw_id"] = moments.shrink_covariance_ledoit_wolf(weekly_returns, covs["exp_raw"], target="identity")

    valid_pairs: Iterable[Tuple[str, str]] = [
        ("uniform_jorion", "uniform_lw_cc"),
        ("uniform_jorion", "uniform_lw_id"),
        ("exp", "exp_raw"),
        ("exp_jorion", "exp_lw_cc"),
        ("exp_jorion", "exp_lw_id"),
    ]

    chosen_weights: Dict[Tuple[str, str], pd.Series] = {}
    frontiers: Dict[Tuple[str, str], PortfolioFrontier] = {}

    for mean_key, cov_key in valid_pairs:
        print(f"Building frontier for mean='{mean_key}', covariance='{cov_key}'")
        frontier = build_frontier(means[mean_key], covs[cov_key])
        frontiers[(mean_key, cov_key)] = frontier
        chosen_weights[(mean_key, cov_key)] = select_portfolio(frontier, max_vol=MAX_ANNUALISED_VOL)

    samples_df = pd.concat(chosen_weights.values(), axis=1)
    samples_df.columns = [f"{mean}|{cov}" for mean, cov in chosen_weights.keys()]
    average_series = average_exposures(samples_df)
    stacked_series = exposure_stacking(samples_df, L=min(3, samples_df.shape[1]))

    print("\n=== Ensemble averages across specifications ===")
    print(average_series.round(4))
    print("\n=== Exposure stacking (L=min(3, n_portfolios)) ===")
    print(stacked_series.round(4))

    print("\nIndividual portfolio weights used in the ensemble:")
    for spec, weights in chosen_weights.items():
        print(f"  {spec}:")
        print(weights.round(4).to_string(index=True))


if __name__ == "__main__":
    main()

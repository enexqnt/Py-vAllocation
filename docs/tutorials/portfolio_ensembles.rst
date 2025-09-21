Portfolio Ensembling
====================

This tutorial shows how to blend portfolios produced under different statistical
specifications into a single allocation. We combine mean estimates (plain vs.
shrinkage) with several covariance shrinkers, extract one target portfolio from
each frontier, and ensemble the resulting weights.

Setup
-----

.. code-block:: python

    from pathlib import Path
    import numpy as np
    import pandas as pd

    from pyvallocation import moments, probabilities
    from pyvallocation.ensembles import average_exposures, exposure_stacking
    from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper
    from pyvallocation.utils.projection import convert_scenarios_compound_to_simple, log2simple, project_mean_covariance

    data_path = Path("examples/ETF_prices.csv")
    prices = pd.read_csv(data_path, index_col="Date", parse_dates=True).ffill().dropna()
    weekly_prices = prices.resample("W-FRI").last().dropna(how="all")
    weekly_returns = convert_scenarios_compound_to_simple(np.log(weekly_prices).diff().dropna())
    weekly_returns = pd.DataFrame(weekly_returns, index=weekly_prices.index[1:], columns=weekly_prices.columns)

    T = len(weekly_returns)
    p_uniform = probabilities.generate_uniform_probabilities(T)
    p_exp = probabilities.generate_exp_decay_probabilities(T, half_life=max(T // 8, 10))

    means = {}
    covs = {}
    means["uniform"], covs["uniform_raw"] = moments.estimate_sample_moments(weekly_returns, p_uniform)
    means["exp"], covs["exp_raw"] = moments.estimate_sample_moments(weekly_returns, p_exp)

    means["uniform_jorion"] = moments.shrink_mean_jorion(means["uniform"], covs["uniform_raw"], T)
    means["exp_jorion"] = moments.shrink_mean_jorion(means["exp"], covs["exp_raw"], T)

    covs["uniform_lw_cc"] = moments.shrink_covariance_ledoit_wolf(weekly_returns, covs["uniform_raw"], target="constant_correlation")
    covs["uniform_lw_id"] = moments.shrink_covariance_ledoit_wolf(weekly_returns, covs["uniform_raw"], target="identity")
    covs["exp_lw_cc"] = moments.shrink_covariance_ledoit_wolf(weekly_returns, covs["exp_raw"], target="constant_correlation")
    covs["exp_lw_id"] = moments.shrink_covariance_ledoit_wolf(weekly_returns, covs["exp_raw"], target="identity")

Extract a target portfolio from each frontier
--------------------------------------------

.. code-block:: python

    INVESTMENT_HORIZON = 52
    MAX_ANNUALISED_VOL = 0.12

    def build_frontier(mu, cov):
        mu_hor, cov_hor = project_mean_covariance(mu, cov, annualization_factor=INVESTMENT_HORIZON)
        mu_simple, cov_simple = log2simple(mu_hor, cov_hor)
        wrapper = PortfolioWrapper(AssetsDistribution(mu=mu_simple, cov=cov_simple))
        wrapper.set_constraints({"long_only": True, "total_weight": 1.0})
        return wrapper.mean_variance_frontier(num_portfolios=11)

    def pick_portfolio(frontier):
        weights, ret, risk = frontier.portfolio_at_risk_target(MAX_ANNUALISED_VOL)
        if weights.isna().any():
            weights, ret, risk = frontier.get_min_risk_portfolio()
        return weights

    valid_pairs = [
        ("uniform_jorion", "uniform_lw_cc"),
        ("uniform_jorion", "uniform_lw_id"),
        ("exp", "exp_raw"),
        ("exp_jorion", "exp_lw_cc"),
        ("exp_jorion", "exp_lw_id"),
    ]

    portfolios = []
    for mean_key, cov_key in valid_pairs:
        frontier = build_frontier(means[mean_key], covs[cov_key])
        portfolios.append(pick_portfolio(frontier))

Ensemble the exposures
----------------------

.. code-block:: python

    samples = np.column_stack([w.values for w in portfolios])
    average = average_exposures(samples)
    stacked = exposure_stacking(samples, L=min(3, samples.shape[1]))

    average_series = pd.Series(average, index=portfolios[0].index, name="Average Ensemble")
    stacked_series = pd.Series(stacked, index=portfolios[0].index, name="Exposure Stacking")

    print(average_series.round(4))
    print(stacked_series.round(4))

Interpretation
--------------

- ``average_exposures`` provides the simplest blend across specifications; it
  retains the base constraints and keeps weights roughly centred.
- ``exposure_stacking`` solves a quadratic programme that dampens extreme weights
  and often results in more concentrated allocations.
- You can also ensemble columns within a single frontier via
  :meth:`pyvallocation.portfolioapi.PortfolioFrontier.ensemble_average`.

For a runnable end-to-end script, execute ``python examples/portfolio_ensembles.py``.

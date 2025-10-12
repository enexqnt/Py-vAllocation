Portfolio Ensembling Across Models
==================================

This tutorial expands on ``portfolio_ensembles.py`` and demonstrates how the
new :func:`pyvallocation.ensembles.make_portfolio_spec` and
:func:`pyvallocation.ensembles.assemble_portfolio_ensemble` helpers produce a
stacked allocation in a handful of lines. The workflow compares two estimation
styles (shrinkage MV vs. robust RRP) using the same universe as the other
tutorials.

Setup
-----

.. code-block:: python

    from pathlib import Path
    import numpy as np
    import pandas as pd

    from pyvallocation import moments, probabilities
    from pyvallocation.ensembles import (
        average_exposures,
        exposure_stacking,
        average_frontiers,
        exposure_stack_frontiers,
    )
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
---------------------------------------------

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

Run both estimators in a single call
------------------------------------

.. code-block:: python

    from pyvallocation.ensembles import make_portfolio_spec, assemble_portfolio_ensemble

    spec_mv = make_portfolio_spec(
        name="MV",
        returns=weekly_returns,
        mean_estimator="james_stein",
        cov_estimator="oas",
        optimiser="mean_variance",
        optimiser_kwargs={
            "num_portfolios": 15,
            "constraints": {"long_only": True, "total_weight": 1.0},
        },
        selector="tangency",
        selector_kwargs={"risk_free_rate": 0.01},
    )

    spec_rrp = make_portfolio_spec(
        name="RRP",
        returns=weekly_returns,
        mean_estimator="huber",
        cov_estimator="tyler",
        cov_kwargs={"shrinkage": 0.1},
        optimiser="rrp",
        optimiser_kwargs={
            "num_portfolios": 6,
            "max_multiplier": 1.6,
            "lambda_reg": 0.3,
            "constraints": {"long_only": True, "total_weight": 1.0},
        },
        selector="max_return",
    )

    ensemble = assemble_portfolio_ensemble(
        [spec_mv, spec_rrp],
        ensemble=("average", "stack"),
        stack_folds=3,
    )

    stacked_weights = ensemble.stacked
    average_weights = ensemble.average

    print("Stacked weights (top holdings):")
    print(stacked_weights.sort_values(ascending=False).head())

    print("\nIndividual selections:")
    print(ensemble.selections.round(4))

Interpretation & reporting
--------------------------

- ``ensemble.metadata`` preserves key estimation choices per model – include it
  in investment memos.
- Plot ``ensemble.frontiers["MV"]`` and ``ensemble.frontiers["RRP"]`` using
  :func:`pyvallocation.plotting.plot_frontiers` to show the range of outcomes.
- Pass ``ensemble_weights`` when calling :func:`assemble_portfolio_ensemble` to
  reflect qualitative preferences (e.g., overweight robust models).
- The stacked weights are PSD-safe and label-preserving, so you can feed them
  directly into discrete allocation, attribution, or risk aggregation modules.

.. note::

   The exposure stacking procedure follows Vorobets
   :cite:p:`vorobets2024derivatives` and the reference implementation published
   in the GPL-3 licensed `fortitudo.tech <https://github.com/fortitudo-tech/fortitudo.tech>`_
   repository.

Interpretation
--------------

- ``average_exposures`` provides the simplest blend across specifications; it
  retains the base constraints and keeps weights roughly centred.
- ``exposure_stacking`` solves a quadratic programme that dampens extreme weights
  and often results in more concentrated allocations.
- You can also ensemble columns within a single frontier via
  :meth:`pyvallocation.portfolioapi.PortfolioFrontier.ensemble_average`.

For a runnable end-to-end script, execute ``python examples/portfolio_ensembles.py``.

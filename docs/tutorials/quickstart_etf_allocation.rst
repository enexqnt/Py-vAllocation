Comprehensive ETF Allocation Quickstart
=======================================

This tutorial mirrors :file:`examples/quickstart_etf_allocation.py`. It walks
through the same end-to-end workflow a portfolio team would execute when
preparing an allocation memo:

1. Load ETF prices and derive weekly log returns.
2. Estimate moments using shrinkage, robust, Bayesian, and view-based models.
3. Project the statistics to a one-year horizon.
4. Configure multiple optimiser specifications and assemble an ensemble.
5. Plot and sanity-check the resulting frontiers.
6. Convert the final stack into discrete trade lots and persist artefacts.
7. Stress test the allocation with adverse flexible views.

Prerequisites
-------------

.. code-block:: bash

   python -m pip install "py-vallocation[robust]"

The ``robust`` extra installs optional dependencies used in this workflow:
analytical nonlinear shrinkage, POET, and related packages. The tutorial also
uses ``matplotlib`` for plotting.

Dataset
-------

The script ships with :file:`examples/ETF_prices.csv`, a small basket of global
ETF prices. All paths below assume you run the quickstart from the repository
root:

.. code-block:: bash

   python examples/quickstart_etf_allocation.py

Step 1 - Load and preprocess prices
-----------------------------------

Weekly log returns offer a stable base for shrinkage estimators and Bayesian
updates. The helper below keeps pandas labels so downstream objects remain
label-aware:

.. code-block:: python

   from pathlib import Path
   import numpy as np
   import pandas as pd

   DATA_PATH = Path("examples/ETF_prices.csv")

   def load_weekly_data() -> tuple[pd.DataFrame, pd.DataFrame]:
       if not DATA_PATH.exists():
           raise FileNotFoundError(f"Expected data at {DATA_PATH}")
       prices = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True).ffill()
       weekly_prices = prices.resample("W-FRI").last().dropna(how="all")
       weekly_returns = np.log(weekly_prices).diff().dropna()
       weekly_returns = weekly_returns.rename(columns=lambda c: c.replace(" ", "_"))
       weekly_prices = weekly_prices.rename(columns=lambda c: c.replace(" ", "_"))
       return weekly_returns, weekly_prices

``weekly_returns`` drives all moment estimators; ``weekly_prices`` is reused for
reporting and discrete trade sizing.

Step 2 - Estimate moments under multiple models
-----------------------------------------------

The quickstart showcases four complementary approaches:

- **Shrinkage** - James-Stein mean + OAS covariance via
  :func:`pyvallocation.moments.estimate_moments`.
- **Robust** - Huber mean + Tyler covariance with shrinkage.
- **Robust Bayesian** - Normal-Inverse-Wishart posterior using shrinkage moments
  as priors and robust statistics as evidence via
  :func:`pyvallocation.moments.posterior_moments_niw`.
- **Black-Litterman** - macro view favouring SPY over TLT via
  :class:`pyvallocation.views.BlackLittermanProcessor`.
- **Entropy pooling** - opinion pooling with flexible views using
  :class:`pyvallocation.views.FlexibleViewsProcessor`.

.. code-block:: python

   from pyvallocation.moments import estimate_moments, posterior_moments_niw
   from pyvallocation.views import BlackLittermanProcessor, FlexibleViewsProcessor

   mu_shrink, sigma_oas = estimate_moments(
       weekly_returns,
       mean_estimator="james_stein",
       cov_estimator="oas",
   )
   mu_huber, sigma_tyler = estimate_moments(
       weekly_returns,
       mean_estimator="huber",
       cov_estimator="tyler",
       cov_kwargs={"shrinkage": 0.1},
   )
   mu_rb, sigma_rb = posterior_moments_niw(
       prior_mu=mu_shrink,
       prior_sigma=sigma_oas,
       t0=8,
       nu0=max(len(weekly_returns.columns) + 2, 6),
       sample_mu=mu_huber,
       sample_sigma=sigma_tyler,
       n_obs=len(weekly_returns),
   )

   bl_posterior = BlackLittermanProcessor(
       prior_cov=sigma_oas,
       prior_mean=mu_shrink,
       mean_views={"SPY": 0.0003, ("SPY", "TLT"): 0.0001},
       risk_aversion=2.5,
       tau=0.05,
   )
   mu_bl, sigma_bl = bl_posterior.get_posterior()

   flex_processor = FlexibleViewsProcessor(
       prior_returns=weekly_returns,
       mean_views={"SPY": (">=", 0.0001), ("SPY", "TLT"): (">=", 0.0)},
       corr_views={("SPY", "TLT"): ("<=", -0.05)},
       sequential=True,
       random_state=7,
   )
   mu_ep, sigma_ep, q_ep = flex_processor.get_posterior()

``mu_ep`` and ``sigma_ep`` remain in weekly log units. The quickstart later
projects them to annual simple returns for comparability.

Step 3 - Project to the investment horizon
------------------------------------------

Projection keeps annualisation and log-to-simple conversions explicit. The
helper below wraps :func:`pyvallocation.utils.projection.project_mean_covariance`
and :func:`pyvallocation.utils.projection.log2simple`:

.. code-block:: python

   from pyvallocation.utils.projection import project_mean_covariance, log2simple

   def project_to_horizon(mu: pd.Series, sigma: pd.DataFrame, annualisation: int):
       mu_proj, sigma_proj = project_mean_covariance(
           mu,
           sigma,
           annualization_factor=annualisation,
       )
       return log2simple(mu_proj, sigma_proj)

   horizon_mu = {}
   horizon_sigma = {}
   for label, (mu, sigma) in {
       "Shrinkage": (mu_shrink, sigma_oas),
       "Robust": (mu_huber, sigma_tyler),
       "RobustBayes": (mu_rb, sigma_rb),
       "BL": (mu_bl, sigma_bl),
   }.items():
       horizon_mu[label], horizon_sigma[label] = project_to_horizon(mu, sigma, annualisation=52)

Entropy pooling uses the posterior probabilities ``q_ep`` to bootstrap annual
simple-return scenarios via ``build_annual_simple_scenarios`` (see the script).

Step 4 - Configure optimiser specifications and assemble the ensemble
---------------------------------------------------------------------

Each specification pairs a distribution with an optimiser/selector combo via
:func:`pyvallocation.ensembles.make_portfolio_spec`. Long-only constraints are
encoded with :func:`pyvallocation.utils.constraints.build_G_h_A_b` under the
hood.

.. note::

   ``build_annual_simple_scenarios`` is shipped with the quickstart script. It
   resamples weekly log returns into annual simple-return scenarios using the
   entropy-pooling probabilities.

.. code-block:: python

   from examples.quickstart_etf_allocation import build_annual_simple_scenarios
   from pyvallocation.ensembles import make_portfolio_spec, assemble_portfolio_ensemble
   from pyvallocation.portfolioapi import AssetsDistribution

   long_only = {"long_only": True, "total_weight": 1.0, "bounds": (None, 0.6)}
   projection = {"annualization_factor": 52, "log_to_simple": True}

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
           name="BL_MV",
           distribution=AssetsDistribution(mu=horizon_mu["BL"], cov=horizon_sigma["BL"]),
           optimiser="mean_variance",
           optimiser_kwargs={"num_portfolios": 21, "constraints": long_only},
           selector="risk_target",
           selector_kwargs={"max_risk": 0.12},
           metadata={"model": "Black-Litterman", "horizon": "1Y"},
       ),
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
       # Additional specs omitted here match the reference script (NLS, Robust Bayes, EP MV).
   ]

   ensemble = assemble_portfolio_ensemble(
       specs,
       ensemble=("average", "stack"),
       stack_folds=3,
   )

The helper returns an :class:`pyvallocation.ensembles.EnsembleResult` containing
frontiers, per-model selections, the stacked allocation, and metadata.

Step 5 - Inspect selections and trailing metrics
------------------------------------------------

The script prints a table of selected weights per spec and highlights the top
stacked holdings. It also computes trailing Sharpe metrics using simple returns:

.. code-block:: python

   print("Selected portfolios (per model):")
   print(ensemble.selections.round(4))
   print("\nStacked allocation (top 5 holdings):")
   print(ensemble.stacked.sort_values(ascending=False).head())

   weekly_simple = weekly_prices.pct_change().dropna()
   stacked_weights = ensemble.stacked.reindex(weekly_simple.columns, fill_value=0.0)
   portfolio_weekly = weekly_simple.dot(stacked_weights)
   annualised_return = (1.0 + portfolio_weekly.mean())**52 - 1.0
   annualised_vol = portfolio_weekly.std(ddof=0) * np.sqrt(52)
   sharpe = (annualised_return - 0.01) / annualised_vol if annualised_vol > 0 else np.nan
   print(
       f"Stacked portfolio trailing metrics: "
       f"return={annualised_return:.2%}, vol={annualised_vol:.2%}, Sharpe~{sharpe:.2f} (rf=1%)"
   )

Step 6 - Plot frontiers
-----------------------

The tutorial saves three figures:

- ``frontiers_vol.png`` - volatility-based frontiers (mean-variance, BL, robust).
- ``frontiers_cvar.png`` - CVaR frontiers.
- ``frontiers.png`` - composite figure for dashboards.

All plots rely on :func:`pyvallocation.plotting.plot_frontiers`.

.. code-block:: python

   from pyvallocation.plotting import plot_frontiers

   fig, ax = plt.subplots(figsize=(8, 5))
   mv_names = [name for name in ensemble.frontiers if "CVaR" not in name]
   mv_frontiers = {name: ensemble.frontiers[name] for name in mv_names}
   plot_frontiers(mv_frontiers, ax=ax, highlight=())
   ax.set_title("ETF Frontier Comparison - 1Y Horizon (Volatility)")
   ax.set_xlabel("Risk")
   ax.set_ylabel("Expected Return")
   fig.tight_layout()
   fig.savefig(OUTPUT_DIR / "frontiers_vol.png", dpi=150)

The script repeats the call for CVaR frontiers and stores a combined figure
under ``output/frontiers.png``.

Step 7 - Discretise to trades
-----------------------------

:func:`pyvallocation.discrete_allocation.discretize_weights` converts the stacked
weights into share counts for a notional USD 10M portfolio. Residual cash and
tracking error are surfaced so you can evaluate implementation risk.

.. code-block:: python

   from pyvallocation.discrete_allocation import discretize_weights

   allocation = discretize_weights(
       weights=ensemble.stacked,
       latest_prices=weekly_prices.iloc[-1],
       total_value=10_000_000,
   )

   summary = pd.DataFrame(
       {
           "Target Weight": ensemble.stacked.round(4),
           "Achieved Weight": allocation.achieved_weights.reindex(ensemble.stacked.index).round(4),
           "Shares": pd.Series(allocation.shares, dtype=int),
           "Market Value": (pd.Series(allocation.shares) * weekly_prices.iloc[-1]).round(2),
       }
   )
   print(summary[summary["Shares"] > 0].sort_values("Market Value", ascending=False))
   print(f"Residual cash: {allocation.leftover_cash:,.2f}")
   print(f"Tracking error (RMSE): {allocation.tracking_error:.6f}")

Step 8 - Persist artefacts and stress test
------------------------------------------

CSV artefacts (selected weights, stacked weights, averages, metadata) are saved
under ``output/``. A closing stress test applies adverse flexible views and
shows the change in optimal weights relative to the baseline stack:

.. code-block:: python

   stress_processor = FlexibleViewsProcessor(
       prior_returns=weekly_returns,
       mean_views={"SPY": ("<=", -0.0002)},
       corr_views={("SPY", "TLT"): (">=", -0.02)},
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

   stress_wrapper = PortfolioWrapper(AssetsDistribution(mu=mu_stress_1y, cov=sigma_stress_1y))
   stress_wrapper.set_constraints(long_only)
   stress_frontier = stress_wrapper.mean_variance_frontier(num_portfolios=21)
   stress_weights, *_ = stress_frontier.portfolio_at_risk_target(max_risk=0.12)
   drift = (stress_weights - ensemble.stacked).dropna()
   print(drift.reindex(drift.abs().sort_values(ascending=False).head().index))

Next steps
----------

- Integrate firm-specific constraints by extending ``long_only`` and feeding the
  resulting matrices back via :func:`pyvallocation.ensembles.make_portfolio_spec`.
- Swap ``weekly_returns`` with your own dataset; the stack preserves pandas
  labels end-to-end.
- Explore additional selectors (``tangency``, ``min_risk``) available on
  :class:`pyvallocation.portfolioapi.PortfolioFrontier`.
- Use :func:`pyvallocation.ensembles.exposure_stack_frontiers` to blend more
  aggressive models or alternative taxonomies such as factor sleeves.

Running the script end-to-end takes under a minute on a laptop and produces
everything required for governance packs: tables, charts, discrete trades, and
stress diagnostics.

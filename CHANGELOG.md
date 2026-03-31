# Changelog

All notable changes to this project will be documented in this file.

## [0.6.0] - 2026-03-31
### Added
- **Meucci Prayer pipeline**: `PortfolioWrapper.from_prices()` for newbies,
  `.from_invariants()` for the full P3-P4 pipeline with any instrument mix.
- **K→N repricing**: `compose_repricers()` maps K risk drivers to N instruments.
  Multi-driver specs: `"Call": (["underlying", "vol"], greeks_fn)`.
- **Invariant-level stress**: `stress_invariants()` applies views to risk drivers,
  projects, reprices, and compares nominal vs stressed metrics.
- **View helpers**: `at_least()`, `at_most()`, `between()`, `above()`, `below()`.
- **Range views**: `vol_views={"SPY": between(0.12, 0.20)}`.
- **Rank views**: `rank_mean=["SPY", "TLT", "GLD"]` (E[SPY] >= E[TLT] >= E[GLD]).
- **Variance views**: `var_views={"SPY": at_most(0.04)}`.
- **Quantile (VaR) views**: `quantile_views={"SPY": (-0.10, at_most(0.05))}`.
- `FlexibleViewsProcessor.get_scenarios()` exposes the EP scenario matrix.
- `return_type="log"` on `from_moments()` / `from_scenarios()` auto-converts.
- `seed` parameter on `project_scenarios()`.
- `enforce_convexity` parameter on `cvar_frontier()`.
### Changed
- **`from_robust_posterior()`** now stores Sigma_ce in `dist.cov` (return covariance)
  and S_mu in `_uncertainty_cov`. Both `variance_frontier()` and
  `robust_lambda_frontier()` work from the same wrapper.
- Robust frontier auto-computes volatility overlay from Sigma_ce.
- Frontier returns/risks always reported in simple-return space via `_report_mu_cov()`.
- CVaR scenario generation upgraded from INFO to WARNING log level.
- Non-sequential EP two-pass linearization for mixed mean + higher-order views.
- `max_sharpe()` warns and skips non-homogeneous equality constraints.
### Fixed
- `from_prices()` validates for NaN, Inf, and non-positive prices early.
- View helpers validate finite targets at creation time.

## [0.5.1] - 2026-03-29
### Added
- Helper functions, additional tests, tutorial notebooks, code polish.

## [0.5.0] - 2026-03-29
### Added
- `Constraints` frozen dataclass with `group_constraints`, `from_dict()`, IDE autocomplete.
- `TransactionCosts` frozen dataclass for immutable cost specification.
- `PortfolioWrapper.from_moments()`, `.from_scenarios()`, `.from_robust_posterior()` factory classmethods.
- All frontier methods accept `constraints=` and `costs=` keyword overrides (immutable per-call).
- Budget risk parity via `risk_budgets` parameter on `RelaxedRiskParity` (Richard-Roncalli 2019).
- CVaR/VaR views in `FlexibleViewsProcessor` via recursive EP (Meucci 2011, ssrn-1542083).
- `covariance_ewma()` EWMA covariance estimator (RiskMetrics).
- `MeanVariance.max_sharpe()` direct solver (Cornuejols-Tutuncu reformulation).
- Short position support in discrete allocation (auto-delegates to MILP).
- Prayer-aligned repricing: `reprice_exp`, `reprice_taylor`, `make_repricing_fn` for stocks, bonds, and derivatives.
- `project_scenarios` accepts `reprice=` callable for P3+P4 in one step.
- `log2simple`, `simple2log`, `project_mean_covariance`, `project_scenarios` now exported from top-level.
- `PortfolioFrontier.__post_init__` validates shape consistency.
- `__repr__` on `PortfolioFrontier` and `PortfolioWrapper`.
- `THEORY.md` mapping papers to implementations with Prayer framework guide.
- 48 new tests (total: 189). New test files: `test_views.py`, `test_api_v2.py`.
- New examples: `budget_risk_parity.py`, `group_constraints.py`, `repricing_derivatives.py`.

### Changed
- **BREAKING**: Removed `set_constraints()`, `set_transaction_costs()`, `_ensure_default_constraints()`. Use factory classmethods or per-method `constraints=`/`costs=` kwargs.
- **BREAKING**: Removed 7 deprecated `PortfolioFrontier` method aliases (`get_min_risk_portfolio`, etc.). Use `min_risk()`, `max_return()`, `tangency()`, `at_risk()`, `at_return()`, `at_percentile()`, `closest_risk()`.
- **BREAKING**: Removed 5 deprecated `PortfolioWrapper` methods (`mean_variance_frontier`, etc.). Use `variance_frontier()`, `cvar_frontier()`, `robust_lambda_frontier()`, `min_variance_at_return()`, `min_cvar_at_return()`.
- **BREAKING**: Removed `prior_returns` deprecated parameter from `FlexibleViewsProcessor`. Use `prior_risk_drivers`.
- **BREAKING**: Removed `ensemble_average` / `blend_columns` (misleading per Vorobets). Use `average_frontiers` with `risk_percentile_selections` for cross-frontier ensembles.
- Cleaned `__init__.py`: removed internal symbols (`build_G_h_A_b`, `allocate_greedy`, `allocate_mip`, `average_exposures`, `exposure_stacking`). Still importable via submodule paths.
- Added `portfolio_variance`, `portfolio_volatility`, `Constraints`, `TransactionCosts` to top-level exports.

### Fixed
- Variance bias inconsistency in sequential EP (`np.cov(bias=True)` for consistency).
- BLP asset index: fixed silent wrong-asset selection for numeric string labels.
- View target validation: vol must be positive, correlation in [-1,1], skew rejects near-zero variance.
- BLP confidence key mismatch now warns when dict keys don't match view keys.
- Asymmetric matrix input now warns before silent symmetrisation in Cholesky.
- Robust optimizer: docstrings now state `dist.cov` must be S_mu (mean-uncertainty scatter), not Sigma_1.
- Documented Ledoit-Wolf `kappa/T` constant-correlation variant.
- Added `cred_radius_mu` and `cred_radius_sigma_factor` to `RobustBayesPosterior`.
- NaN clamping in `cred_radius_sigma_factor` instead of returning NaN.
- Extracted `_wrap_vector`/`_wrap_matrix` helpers in NIWPosterior; removed dead `get_posterior()`.

## [0.4.1] - 2026-03-29
### Fixed
- ENS formula: changed from Herfindahl `1/sum(p^2)` to entropy `exp(-sum(p*ln(p)))` per Meucci (2012) and Vorobets (2025, Eq. 5.1.3).
- `relative_bounds` constraint: corrected from proportional `w_i - k*w_j <= 0` to additive `w_i - w_j <= k`.
- In-place mutation of caller's scenario array when `demean=True` in `portfolio_cvar`/`portfolio_var`.
- Discrete allocation: denominator changed from invested to total portfolio value; tracking error changed from L2 (RMSE) to L1.
- OAS shrinkage: added `(1-2/p)` correction from Chen et al. (2010).
- CVaR: exact Rockafellar-Uryasev discrete formula with quantile boundary correction.
- `ddof=1` changed to `ddof=0` in `estimate_moments` for MLE consistency with `estimate_sample_moments`.
- Silent NaN returns on infeasible optimisation replaced with `InfeasibleOptimizationError`.
- `relaxed_risk_parity_portfolio` return type unified to `(Series, float, float)` matching all other portfolio methods.

### Changed
- `alpha` parameter renamed to `confidence` in `portfolio_cvar`, `portfolio_var`, `stress_test`, `performance_report`, and related helpers. `MeanCVaR` keeps `alpha` as tail probability with clarified docstring.
- Idzorek Omega documented as Walters (2014) closed-form approximation; multi-asset basket views noted as unsupported.
- `marginal_risk` docstring clarified as variance-scale gradient (not Roncalli's volatility-normalised definition).
- Default constraint injection upgraded from `DEBUG` to `WARNING` log level.
- Probability normalisation and moment estimation from scenarios now emit `WARNING` as documented.
- Transaction cost type mismatches now warn (e.g. `proportional_costs` ignored by mean-variance).
- Exposure stacking now requires `L >= 2` per cross-validation theory.
- Scale-adaptive ridge regularisation in MeanVariance QP fallback.
- Dropped Python 3.8 support (EOL Oct 2024). Minimum is now Python 3.9.

### Added
- `InfeasibleOptimizationError` exception for infeasible portfolio optimisation problems.
- `compute_effective_number_scenarios_hhi` for the Herfindahl-based alternative.
- `RobustBayesPosterior.sigma1` property exposing raw posterior scale Sigma_1 (Meucci Eq. 7.31).
- `exact=True` option on `mean_uncertainty_cov_simple` for Jensen-corrected delta-method Jacobian.
- `risk_contributions_pct` (volatility-normalised, summing to ~100%) in relaxed risk parity diagnostics.
- `relaxed_risk_parity_portfolio_with_diagnostics` for full solver diagnostics access.
- `seed` parameter on `cvar_frontier` and `min_cvar_at_return` for reproducible scenario simulation.
- Input guards: `simple2log`/`convert_scenarios_simple_to_compound` reject returns <= -1; NIW `update()` prevents double-call; `_labels` validates pandas alignment.
- Warnings for Tyler M-estimator non-convergence, POET `k >= N`, non-finite stress transforms, NaN after weight reindex.
- `estimate_moments` factory lists valid options in error messages.
- GitHub Actions CI (pytest on 3.9–3.12, ruff lint).
- ruff, mypy, pytest configuration in `pyproject.toml`.
- pytest fixtures in `conftest.py` for shared test data.
- 40 new tests covering utility modules and error paths (total: 142).

## [0.4.0] - 2026-02-01
### Added
- Robust-Bayesian uncertainty helpers via `RobustBayesPosterior`, including closed-form NIW mean-uncertainty (`S_mu`) and horizon-scaled variants.
- `posterior_moments_niw_with_uncertainty` for workflows that need posterior mean/covariance plus `S_mu` directly.
- Ensemble specification API (`EnsembleSpec`, `make_portfolio_spec`, `assemble_portfolio_ensemble`) and risk-percentile alignment utilities for consistent ensemble selection.
- New plotting utilities for diagnostics and reporting: frontier reports, robust parameter-impact plots, robust path overlays, and 3D assumption/uncertainty views.
- Performance helpers for scenario P&L and compact risk reporting (`scenario_pnl`, `performance_report`).
- Example utilities to keep tutorials concise (`examples/data_utils.py`, `examples/example_utils.py`) and refreshed notebooks.

### Changed
- Portfolio frontiers and robust optimisation paths now align on consistent risk labels/risk targets (in-sample vs out-of-sample shown explicitly).
- Robust optimisation visuals now include uncertainty overlays and richer assumptions plots; quickstart outputs updated for clearer narrative.
- Ensemble workflows now target comparable risk levels across models (instead of mixing different risk points on a single frontier).
- Documentation and examples reorganised for compactness, consistency, and a warning-free Sphinx build.

### Fixed
- Documentation rendering issues for math subscripts/citations and docstring formatting; examples now render as proper code blocks.
- API reference formatting standardised (parameter blocks and type hints moved into descriptions).

## [0.3.1] - 2025-11-06
### Changed
- Added API reference pages for discrete allocation and plotting utilities.
- Refreshed README messaging for optimisation surface and production utilities.
- Synced upstream improvements and validated on Python 3.12.

## [0.3.0] - 2025-02-25
### Changed
- Hardened discrete allocation with iteration safeguards and richer tests.
- Removed `numpy.matlib` deprecation noise in nonlinear shrinkage.
- Cleaned docs and expanded quickstart narrative.

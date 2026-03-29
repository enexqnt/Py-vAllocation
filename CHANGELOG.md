# Changelog

All notable changes to this project will be documented in this file.

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

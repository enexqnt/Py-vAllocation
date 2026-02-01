# Changelog

All notable changes to this project will be documented in this file.

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

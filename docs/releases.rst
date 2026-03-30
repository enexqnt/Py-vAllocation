Release Notes
=============

.. contents::
   :local:
   :depth: 1

Version 0.5.0 (2026-03-29)
--------------------------

**Breaking changes** -- see ``CHANGELOG.md`` for migration details.

* Replaced ``set_constraints()`` / ``set_transaction_costs()`` with typed
  ``Constraints`` / ``TransactionCosts`` dataclasses and factory classmethods
  (``from_moments``, ``from_scenarios``, ``from_robust_posterior``).
* Removed 7 deprecated ``PortfolioFrontier`` aliases and 5 deprecated
  ``PortfolioWrapper`` methods.  New canonical names: ``min_risk()``,
  ``max_return()``, ``tangency()``, ``at_risk()``, ``at_return()``, etc.

**New features**: budget risk parity, CVaR views, EWMA covariance,
max-Sharpe direct solver, Prayer repricing framework, group constraints,
short-position discrete allocation, ``PortfolioFrontier`` shape validation.

See the full ``CHANGELOG.md`` in the repository root for details.

Version 0.4.1 (2026-03-29)
--------------------------

Bug fixes, theory corrections, and hardening.  See ``CHANGELOG.md`` for details.

Version 0.4.0 (2026-02-01)
--------------------------

- Added robust-Bayesian uncertainty helpers (NIW mean-uncertainty covariance and `RobustBayesPosterior`) plus `posterior_moments_niw_with_uncertainty`.
- Introduced ensemble specification helpers and risk-percentile alignment for consistent portfolio comparisons.
- Expanded plotting utilities with robust diagnostics, report-ready frontier visuals, and 3D assumption/uncertainty views.
- Added scenario P&L and performance reporting helpers for compact risk summaries.
- Reworked examples and tutorials to be shorter, clearer, and aligned with the updated API; Sphinx build is warning-free.

Release checklist (0.4.0) — completed
-------------------------------------

- Confirm version strings (``pyproject.toml``, ``docs/conf.py``) and update ``CHANGELOG.md``.
- Run the test suite (``pytest``) and build example artefacts (``python examples/quickstart_etf_allocation.py``).
- Build documentation (``sphinx-build -b html docs docs/_build/html``) and ensure zero warnings.
- Create a git tag ``v0.4.0`` and draft the GitHub release notes.
- Build distribution artefacts (``python -m build``) and upload to PyPI.

Version 0.3.1 (2025-11-06)
--------------------------

- Added dedicated API reference pages for discrete allocation and plotting utilities, wiring them into the core module index.
- Refreshed the README messaging to highlight the consistent optimisation surface, view integration, and production plumbing helpers.
- Synced with upstream improvements and reran the full pytest suite to keep 0.3.x validated on Python 3.12.


Version 0.3.0 (2025-02-25)
--------------------------

- Hardened the discrete allocation engine with iteration safeguards, automatic
  MILP fallback, and richer unit test coverage for lot sizes and failure modes.
- Eliminated ``numpy.matlib`` deprecation noise by wrapping non-linear
  shrinkage imports in targeted warning filters.
- Ran a documentation deep clean: HTML build is warning-free and ``linkcheck``
  now passes thanks to durable DOI handling and updated references.
- Expanded the quickstart tutorial narrative so discrete trade conversion steps
  explain the new fallback behaviour.

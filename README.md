# Py-vAllocation

[![PyPI](https://img.shields.io/pypi/v/py-vallocation.svg)](https://pypi.org/project/py-vallocation/)
[![Python versions](https://img.shields.io/pypi/pyversions/py-vallocation.svg)](https://pypi.org/project/py-vallocation/)

Py-vAllocation is a research-to-production toolkit for scenario-based portfolio
optimisation following Meucci's Prayer framework (invariants, projection,
repricing, optimisation). Build mean-variance, CVaR, and relaxed risk parity
frontiers; incorporate Black-Litterman and entropy pooling views; apply
shrinkage-heavy statistics; ensemble strategies; stress-test at the
risk-driver level; and convert weights to discrete trades.

## Highlights

- **Meucci Prayer pipeline** - `from_prices()` for newbies, `from_invariants()` for the full P3-P4 pipeline with any instrument (equities, bonds, options, futures, ETFs).
- **K-to-N repricing** - `compose_repricers()` maps K risk drivers to N instruments. Multi-driver specs for options: `("underlying", "vol"): greeks_fn`.
- **Flexible views** - mean, volatility, variance, correlation, skewness, CVaR, quantile (VaR), and rank views via entropy pooling. Helpers: `at_least()`, `at_most()`, `between()`.
- **Consistent optimisation surface** - switch between mean-variance, CVaR, relaxed risk parity, and robust formulations without rewriting constraints.
- **Robust models** - Bayesian NIW posteriors, relaxed risk parity, Meucci-style probability tilts. `from_robust_posterior()` supports both MV and robust frontiers from the same wrapper.
- **Moment estimation** - Ledoit-Wolf, James-Stein, nonlinear shrinkage, Tyler, Huber, POET, graphical lasso, EWMA via `estimate_moments`.
- **Invariant-level stress testing** - `stress_invariants()` applies views to risk drivers, flows through repricing, and reports nominal vs stressed metrics.
- **Production plumbing** - ensemble builders, discrete allocation, plotting, reporting. Pandas labels preserved throughout.

## Design principles

- **No magic, no hidden assumptions** - every step of the Prayer is explicit and user-controlled.
- **Pandas-first inputs/outputs** with consistent labels.
- **Scenario-based risk** by default, with clear risk labels across frontiers.

## Installation

```bash
pip install py-vallocation
```

For nonlinear shrinkage and POET estimators:

```bash
pip install py-vallocation[robust]
```

Requires `cvxopt>=1.2.0`. If you don't have it, see the [installation guide](https://cvxopt.org/install/).

## Quickstart

Run the end-to-end ETF example (writes plots and CSVs to `output/`):

```bash
python examples/quickstart_etf_allocation.py
```

Key artefacts:

- `output/frontiers.png` - in-sample vs out-of-sample efficient frontiers with robust overlay.
- `output/robust_uncertainty.png`, `robust_param_impact.png`, `robust_assumptions_3d.png` - robust diagnostics.
- `output/stacked_weights.csv`, `selected_weights.csv`, `average_weights.csv` - ensemble summaries.
- Terminal output covering discrete trade sizing and stress results.

Or use the API directly -- five factory methods for every user level:

```python
from pyvallocation import PortfolioWrapper

# --- Newbie: from prices ---
port = PortfolioWrapper.from_prices(price_df)
frontier = port.variance_frontier()
w, ret, risk = frontier.tangency(risk_free_rate=0.04)

# --- Intermediate: log-return invariants, project 1Y, reprice to simple ---
import numpy as np
log_rets = np.log(price_df / price_df.shift(1)).dropna()
port = PortfolioWrapper.from_invariants(log_rets, horizon=52, seed=42)
frontier = port.cvar_frontier(alpha=0.05)

# --- Institutional: mixed instruments (stocks + bonds + options) ---
from pyvallocation import compose_repricers, reprice_exp, reprice_taylor
port = PortfolioWrapper.from_invariants(
    invariants_df,  # columns: equity log-return, yield change, vol change
    reprice={
        "SPY":  reprice_exp,
        "TLT":  (["yield_10y"], lambda dy: reprice_taylor(dy, delta=-17, gamma=200)),
        "Call": (["equity_lr", "iv_chg"], my_greeks_fn),
    },
    horizon=52, seed=42,
)
frontier = port.cvar_frontier(alpha=0.05)

# --- Views with helpers ---
from pyvallocation import FlexibleViewsProcessor, at_least, at_most, between
ep = FlexibleViewsProcessor(
    prior_risk_drivers=invariants_df,
    mean_views={"SPY": at_least(0.05)},
    vol_views={"TLT": between(0.08, 0.15)},
    rank_mean=["SPY", "TLT", "GLD"],  # E[SPY] >= E[TLT] >= E[GLD]
)
```

## Examples

The `examples/` directory contains runnable scripts (see `examples/README.md`):

- `quickstart_etf_allocation.py` - moments → frontiers → ensemble → trades
- `mean_variance_frontier.py`, `cvar_allocation.py`, `robust_frontier.py` (use `variance_frontier` / `cvar_frontier`)
- `relaxed_risk_parity_frontier.py`, `portfolio_ensembles.py`, `discrete_allocation.py`
- `stress_and_pnl.py` - probability tilts + linear shocks + performance reports
- `group_constraints.py` - sector/group weight constraints

Notebooks under `docs/tutorials/notebooks/` cover Bayesian views, CVaR frontiers,
derivatives repricing, stress testing, and more.

## Documentation

- Full documentation: https://py-vallocation.readthedocs.io
- Tutorials live under `docs/tutorials/` and mirror the runnable scripts.
- API reference is generated from docstrings (`docs/pyvallocation*.rst`).
- Build locally:

```bash
pip install -e .[robust]
sphinx-build -b html docs docs/_build/html
```

## Repository layout

- `pyvallocation/` - library source code.
- `examples/` - runnable workflows (ETF quickstart, CVaR frontier, ensembles, stress testing, discrete allocation).
- `docs/` - Sphinx site (tutorials, API reference, bibliography).
- `tests/` - pytest suite covering numerical routines, ensembles, plotting, and discrete allocation.
- `output/` - artefacts written by example scripts.

## Requirements

- Python 3.9+
- numpy, pandas, scipy, cvxopt

## References

- Meucci (2005) - Risk and Asset Allocation (Prayer framework)
- Meucci (2008) - Fully Flexible Views (entropy pooling)
- Vorobets (2024) - Derivatives Portfolio Optimization & Exposure Stacking
- Markowitz (1952) - Portfolio Selection
- Black & Litterman (1992) - Global Portfolio Optimization
- Rockafellar & Uryasev (2000) - CVaR optimization
- Gambeta & Kwon (2020) - Relaxed Risk Parity
- Ledoit & Wolf (2004, 2020) - Covariance shrinkage

See the [bibliography](https://py-vallocation.readthedocs.io/en/latest/bibliography.html) for the complete list.

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GPL-3.0-or-later — see [LICENSE](LICENSE) for the full text. Portions of the optimisation routines are adapted (with attribution) from [fortitudo-tech](https://github.com/fortitudo-tech/fortitudo.tech).

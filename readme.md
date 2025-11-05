# Py-vAllocation

[![PyPI](https://img.shields.io/pypi/v/py-vallocation.svg)](https://pypi.org/project/py-vallocation/)
[![Python versions](https://img.shields.io/pypi/pyversions/py-vallocation.svg)](https://pypi.org/project/py-vallocation/)

Practical portfolio allocation tools with a compact, well-tested API. Build mean-variance, CVaR, and relaxed risk parity frontiers; incorporate Black-Litterman and entropy pooling views; apply shrinkage-heavy statistics (NIW, Ledoit-Wolf, nonlinear shrinkage, Tyler, Huber, POET); ensemble strategies; and convert weights to discrete trades. Pandas labels are preserved throughout the workflow.

## Highlights

- **Consistent optimisation surface** - switch between mean-variance, CVaR, relaxed risk parity, and robust formulations without rewriting constraints.
- **View integration** - Black-Litterman mean views plus entropy pooling constraints keep discretionary macro inputs consistent with posterior moments.
- **Robust models** - relaxed risk parity, Bayesian NIW updates, and Meucci-style probability tilts.
- **Moment estimation** - Ledoit-Wolf, James-Stein, nonlinear shrinkage, Tyler, Huber, POET, graphical lasso, and more wired into `estimate_moments`.
- **Production plumbing** - ensemble builders, discrete allocation, plotting, and reporting helpers reduce friction between research and delivery.
- **Stress testing & PnL** - one-line helpers for probability tilts, linear shocks, and risk reports.
- **Optional extras** - install the `robust` extra only when heavy dependencies are needed.

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

- `output/frontiers.png`, `frontiers_vol.png`, `frontiers_cvar.png` - efficient frontiers.
- `output/stacked_weights.csv`, `selected_weights.csv`, `average_weights.csv` - ensemble summaries.
- Terminal output covering discrete trade sizing and stress results.

Or use the API directly:

```python
import pandas as pd
from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper

scenarios = pd.DataFrame({
    "Stock_A": [0.01, -0.02, 0.015],
    "Stock_B": [0.007, 0.003, 0.004]
})

port = PortfolioWrapper(AssetsDistribution(scenarios=scenarios))
port.set_constraints({"long_only": True, "total_weight": 1.0})

frontier = port.mean_variance_frontier(num_portfolios=20)
weights, ret, risk = frontier.get_tangency_portfolio(risk_free_rate=0.01)
print(weights)
```

## Examples

The `examples/` directory contains runnable scripts (see `examples/README.md`):

- `quickstart_etf_allocation.py` - moments → frontiers → ensemble → trades
- `mean_variance_frontier.py`, `cvar_allocation.py`, `robust_frontier.py`
- `relaxed_risk_parity_frontier.py`, `portfolio_ensembles.py`, `discrete_allocation.py`
- `stress_and_pnl.py` - probability tilts + linear shocks + performance reports

Notebooks (`examples/*.ipynb`) mirror the tutorials.

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

- Python 3.8+
- numpy, pandas, scipy, cvxopt

## References

- Markowitz (1952) - Portfolio Selection
- Black & Litterman (1992) - Global Portfolio Optimization
- Ledoit & Wolf (2004, 2020) - Covariance shrinkage
- Meucci (2008) - Fully Flexible Views (entropy pooling)
- Rockafellar & Uryasev (2000) - CVaR optimization

See the [bibliography](https://py-vallocation.readthedocs.io/en/latest/bibliography.html) for the complete list.

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GPL-3.0-or-later — see [LICENSE](LICENSE) for the full text. Portions of the optimisation routines are adapted (with attribution) from [fortitudo-tech](https://github.com/fortitudo-tech/fortitudo.tech).

# Py-vAllocation

[![PyPI](https://img.shields.io/pypi/v/py-vallocation.svg)](https://pypi.org/project/py-vallocation/)
[![Python versions](https://img.shields.io/pypi/pyversions/py-vallocation.svg)](https://pypi.org/project/py-vallocation/)

Practical portfolio allocation tools with a compact, well-tested API. The
library covers mean-variance, mean-CVaR, relaxed risk parity, Meucci-style
robust optimisation, Bayesian updates (Black-Litterman and NIW), entropy
pooling, portfolio ensembling, and discrete trade generation. Pandas labels are
preserved throughout every workflow.

## Highlights

- **Consistent optimisation surface** - switch between mean-variance, CVaR,
  relaxed risk parity, and robust formulations without re-writing constraints.
- **View integration out of the box** - Black-Litterman and entropy pooling
  helpers let discretionary macro views flow into posterior moments.
- **Shrinkage-heavy statistics** - Jorion, James-Stein, OAS, NLS, Tyler, Huber,
  POET, and graphical lasso estimators are wired into `estimate_moments`.
- **Production plumbing** - ensemble builders, discrete allocation, and plotting
  utilities reduce friction between research code and reporting.
- **Optional extras** - install the `robust` extra to enable POET, nonlinear
  shrinkage, and other heavy dependencies only when required.

## Installation

```bash
pip install py-vallocation

# optional extras (nonlinear shrinkage, POET, etc.)
pip install py-vallocation[robust]
```

Requires `cvxopt>=1.2.0`. If CVXOPT is new to your system, follow the
[official guide](https://cvxopt.org/install/).

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

R = pd.DataFrame({"A":[0.01,-0.02,0.015],"B":[0.007,0.003,0.004]})
port = PortfolioWrapper(AssetsDistribution(scenarios=R))
port.set_constraints({"long_only": True, "total_weight": 1.0})
front = port.mean_variance_frontier(num_portfolios=20)
w, ret, risk = front.get_tangency_portfolio(risk_free_rate=0.01)
```

## Examples & notebooks

- Scripts live in `examples/` (see `examples/README.md`). Highlights:
  - `quickstart_etf_allocation.py` - moments -> frontiers -> ensemble -> trades
  - `mean_variance_frontier.py`, `cvar_allocation.py`, `robust_frontier.py`
  - `relaxed_risk_parity_frontier.py`, `portfolio_ensembles.py`, `discrete_allocation.py`
- Notebooks (`examples/*.ipynb`) mirror the tutorials.

## Documentation & resources

- Full documentation is built with Sphinx/napoleon and is ready for Read the Docs deployment. Build locally with
  `sphinx-build -b html docs docs/_build/html`.
- Tutorials live under `docs/tutorials/` and mirror the runnable scripts.
- The API reference is generated from docstrings (`docs/pyvallocation*.rst`).

## Repository layout

- `pyvallocation/` - library source code.
- `examples/` - runnable end-to-end workflows (ETF quickstart, CVaR frontier, ensembles, discrete allocation).
- `docs/` - Sphinx site (tutorials, API reference, bibliography).
- `tests/` - pytest suite covering numerical routines, ensembles, plotting, and discrete allocation.
- `output/` - artefacts written by example scripts.

## Requirements

- Python 3.8+
- numpy, pandas, scipy, cvxopt

## References

- Meucci, A. (2008). Fully Flexible Views: Theory and Practice. https://ssrn.com/abstract=1213325
- Black, F., & Litterman, R. (1992). Global Portfolio Optimization. https://doi.org/10.2469/faj.v48.n5.28
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. https://doi.org/10.1016/S0047-259X(03)00096-4
- Ledoit, O., & Wolf, M. (2020). Analytical Nonlinear Shrinkage of Large-Dimensional Covariance Matrices. https://www.jstor.org/stable/27028732
- Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage Algorithms for MMSE Covariance Estimation. https://doi.org/10.1109/TSP.2010.2053029
- Fan, J., Liao, Y., & Mincheva, M. (2013). Large covariance estimation by thresholding principal orthogonal complements. https://doi.org/10.1093/biomet/ass070
- Tyler, D. E. (1987). A distribution-free M-estimator of multivariate scatter. https://www.jstor.org/stable/2241079
- Jorion, P. (1986). Bayes-Stein Estimation for Portfolio Analysis. https://doi.org/10.2307/2331042
- Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. https://doi.org/10.1093/biostatistics/kxm045
- Rockafellar, R. T., & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. 10.21314/JOR.2000.038
- Markowitz, H. (1952). Portfolio Selection. https://doi.org/10.2307/2975974
- Idzorek, T. (2005). A Step-by-Step Guide to the Black-Litterman Model. https://people.duke.edu/~charvey/Teaching/BA453_2006/Idzorek_onBL.pdf
- Meucci, A. (2005). Robust Bayesian Allocation, https://dx.doi.org/10.2139/ssrn.681553
- Vorobets, A. (2021). Sequential Entropy Pooling Heuristics, http://dx.doi.org/10.2139/ssrn.3936392
- Vorobets, A. (2024). Derivatives Portfolio Optimization and Parameter Uncertainty. https://ssrn.com/abstract=4825945

## Contributing

Issues and pull requests are welcome. Please see `CONTRIBUTING.md`.

## License

GPL-3.0-or-later - see [LICENSE](LICENSE) for the full text. Portions of the
optimisation routines are adapted (with attribution) from
[fortitudo-tech](https://github.com/fortitudo-tech/fortitudo.tech).

---

Copyright (c) enexqnt. GPL-3.0-or-later.

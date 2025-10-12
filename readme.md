# Py‑vAllocation

[![PyPI](https://img.shields.io/pypi/v/py-vallocation.svg)](https://pypi.org/project/py-vallocation/)
[![Python versions](https://img.shields.io/pypi/pyversions/py-vallocation.svg)](https://pypi.org/project/py-vallocation/)

Practical portfolio allocation tools with a small, consistent API. The library
covers mean–variance, mean‑CVaR, robust optimisation, Bayesian updates
(Black–Litterman and NIW), entropy pooling for flexible views, ensembling, and
discrete trade generation. Pandas labels are preserved throughout.

## Features

- Mean–variance and CVaR frontiers with simple selectors (tangency, risk target).
- Robust models: relaxed risk parity and Meucci‑style robust optimisation.
- Views: Black–Litterman (equality mean views) and entropy pooling (inequalities,
  vol/corr/skew), plus a robust‑Bayesian NIW posterior.
- Shrinkage/robust moments: James–Stein, OAS, NLS, Tyler, Huber, POET, Glasso.
- Portfolio ensembling and discrete allocation with lot sizes.

## Install

```bash
pip install py-vallocation

# optional extras (nonlinear shrinkage, POET, etc.)
pip install py-vallocation[robust]
```

Requires `cvxopt>=1.2.0`. If CVXOPT is new to your system, follow the
[official guide](https://cvxopt.org/install/).

## Quickstart

Run the end‑to‑end ETF example (writes plots/CSVs to `output/`):

```bash
python examples/quickstart_etf_allocation.py
```

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
  - `quickstart_etf_allocation.py` – moments → frontiers → ensemble → trades
  - `mean_variance_frontier.py`, `cvar_allocation.py`, `robust_frontier.py`
  - `relaxed_risk_parity_frontier.py`, `portfolio_ensembles.py`, `discrete_allocation.py`
- Notebooks (`examples/*.ipynb`) mirror the tutorials.

## Requirements

- Python 3.8+
- numpy, pandas, scipy, cvxopt

## Documentation

Build locally:

```bash
python -m pip install -e .[robust]
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` for tutorials and API reference.

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

GPL-3.0-or-later – see [LICENSE](LICENSE) for the full text. Portions of the
optimisation routines are adapted (with attribution) from
[fortitudo-tech](https://github.com/fortitudo-tech/fortitudo.tech).

---

Copyright © enexqnt. GPL‑3.0‑or‑later.

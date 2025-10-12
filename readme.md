# Py-vAllocation

[![PyPI](https://img.shields.io/pypi/v/py-vallocation.svg)](https://pypi.org/project/py-vallocation/)
[![Python versions](https://img.shields.io/pypi/pyversions/py-vallocation.svg)](https://pypi.org/project/py-vallocation/)

_Py-vAllocation_ is a batteries-included portfolio-optimisation toolkit for
Python. It provides consistent APIs for mean-variance, mean-CVaR, robust, and
Bayesian allocation workflows while keeping the modelling assumptions explicit.

## Highlights

Why practitioners and researchers use **Py-vAllocation**:

- **Single interface, multiple solvers** – switch between variance, CVaR, and
  robust objectives without refactoring your pipeline.
- **Relaxed risk parity** – reproduce Gambeta–Kwon’s return-target trade-off with
  a native solver, metadata-rich diagnostics, and plotting-ready frontiers.
- **Investor views made practical** – flexible entropy pooling and
  Black–Litterman utilities help you translate qualitative convictions into
  posterior scenarios.
- **Statistical hygiene** – shrinkage estimators and Bayesian updates are built
  in, avoiding the brittle sample-moment defaults.
- **Portfolio ensembling** – average exposures, stack frontiers, and discretise
  weights to bridge the gap between research portfolios and executable trades.
- **Notebook-friendly design** – everything works with NumPy arrays or pandas
  objects; outputs rehydrate to labelled Series/DataFrames for reporting.

## Installation

```bash
pip install py-vallocation
```

The solver stack relies on `cvxopt>=1.2.0`. If you are new to CVXOPT on macOS or
Windows, the [`cvxopt` installation guide](https://cvxopt.org/install/) walks
through the necessary system packages.

## Quick start

```python
import pandas as pd
from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper

# Toy scenario matrix (rows = scenarios, columns = assets)
scenarios = pd.DataFrame(
    {
        "Equity_US": [0.021, -0.013, 0.018, 0.007, 0.011],
        "Equity_EU": [0.017, -0.009, 0.014, 0.004, 0.006],
        "Credit_US": [0.009, 0.008, 0.007, 0.006, 0.008],
        "Govt_Bonds": [0.004, 0.003, 0.005, 0.002, 0.004],
    }
)

dist = AssetsDistribution(scenarios=scenarios)
wrapper = PortfolioWrapper(dist)
wrapper.set_constraints({"long_only": True, "total_weight": 1.0})

frontier = wrapper.mean_variance_frontier(num_portfolios=20)
weights, expected_return, risk = frontier.get_tangency_portfolio(risk_free_rate=0.001)

print(weights.round(4))
print(f"Expected return: {expected_return:.4%} | Volatility: {risk:.4%}")

# Relaxed risk parity (Gambeta & Kwon, 2020)
rrp_frontier = wrapper.relaxed_risk_parity_frontier(num_portfolios=4, max_multiplier=1.5, lambda_reg=0.25)
print(rrp_frontier.metadata)
```

## Example gallery

The `examples/` directory mirrors the workflows showcased in the documentation
and notebooks:

| Script | What it shows |
| ------ | ------------- |
| `mean_variance_frontier.py` | Build and interrogate a classical efficient frontier |
| `cvar_allocation.py` | Optimise against CVaR with scenario probabilities and inspect the tangency portfolio |
| `robust_frontier.py` | Trace Meucci’s λ-frontier to understand estimation risk budgets |
| `relaxed_risk_parity_frontier.py` | Explore relaxed risk parity targets and diagnostics |
| `discrete_allocation.py` | Turn continuous weights into tradeable share counts |
| `portfolio_ensembles.py` | Blend multiple risk models via exposure averaging and stacking |

Jupyter notebooks (`Example_01.ipynb`, `Bayesian.ipynb`, `Flexible_Views.ipynb`,
`Simple_views_on_mean.ipynb`) provide annotated walkthroughs of the same
concepts using richer datasets.

> 📘 **Documentation:** Read the full guide and API reference at
> [py-vallocation.readthedocs.io](https://py-vallocation.readthedocs.io/).

## Requirements

- Python 3.8+
- `numpy>=1.20`
- `pandas>=1.0`
- `scipy>=1.10`
- `cvxopt>=1.2`

## Development status

**Alpha release** – under active development. Expect sharp edges and potential
API changes as we finalise behaviour across the optimisation back-ends.

## Underlying literature

- Meucci, A. (2008). Fully Flexible Views: Theory and Practice. https://ssrn.com/abstract=1213325
- Black, F., & Litterman, R. (1992). Global Portfolio Optimization. https://doi.org/10.2469/faj.v48.n5.28
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. https://doi.org/10.1016/S0047-259X(03)00096-4
- Jorion, P. (1986). Bayes-Stein Estimation for Portfolio Analysis. https://doi.org/10.2307/2331042
- Rockafellar, R. T., & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. 10.21314/JOR.2000.038
- Markowitz, H. (1952). Portfolio Selection. https://doi.org/10.2307/2975974
- Idzorek, T. (2005). A Step-by-Step Guide to the Black-Litterman Model. https://people.duke.edu/~charvey/Teaching/BA453_2006/Idzorek_onBL.pdf
- Meucci, A. (2005). Robust Bayesian Allocation, https://dx.doi.org/10.2139/ssrn.681553
- Vorobets, A. (2021). Sequential Entropy Pooling Heuristics, http://dx.doi.org/10.2139/ssrn.3936392

## Contributing

Pull requests and issue reports are welcome! Start with
[CONTRIBUTING.md](CONTRIBUTING.md) and open a discussion if you would like to
collaborate on new risk models or data utilities.

## License

GPL-3.0-or-later – see [LICENSE](LICENSE) for the full text. Portions of the
optimisation routines are adapted (with attribution) from
[fortitudo-tech](https://github.com/fortitudo-tech/fortitudo.tech).

## Star history

[![Star History Chart](https://api.star-history.com/svg?repos=enexqnt/Py-vAllocation&type=Date)](https://www.star-history.com/#enexqnt/Py-vAllocation&Date)

# Py-vAllocation

Py-vAllocation is a Python package for asset allocation with a primary focus on integrating investor views.

## Features

It's yet another portfolio optimization library, but unlike many others, **Py-vAllocation** aims to:
- Be modular and beginner-friendly with a simple API, while remaining flexible and customizable for advanced users  
- Avoid hidden assumptions or black-box components, every modeling choice is explicitly stated  
- Incorporate investor views via **fully flexible probabilities** using entropy pooling and the **Black-Litterman** methodology  
- Support **shrinkage** and other **Bayesian estimation** methods  
- Support **variance-based**, **scenario-based CVaR**, and **robust optimization models**

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

## Installation

You can install Py-vAllocation from PyPI using:

```bash
pip install py-vallocation
```

## Quick Start

See [examples here](examples/)

## Requirements

- Python 3.8+
- numpy >= 1.20.0
- cvxopt >= 1.2.0
- pandas
- scipy >= 1.10.0

## Development Status

**Alpha release**: Under active development. Many features are not yet implemented or fully tested. Breaking changes may occur without notice. Use at your own risk.

## Contributing

Contributions and feedback are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the GNU General Public License v3.0 License. See [LICENSE](LICENSE) for details.

## Credits

Some code, where explicitly stated, is adapted from [fortitudo-tech](https://github.com/fortitudo-tech/fortitudo.tech)

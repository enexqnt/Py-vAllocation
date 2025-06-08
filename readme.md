# Py-vAllocation

Py-vAllocation is a Python package for asset allocation with a primary focus on integrating investor views.

## Features

It's yet another portfolio optimization library, but unlike many others, **Py-vAllocation** aims to:
- Be modular and beginner-friendly with a simple API, while remaining flexible and customizable for advanced users  
- Avoid hidden assumptions or black-box components, every modeling choice is explicitly stated  
- Incorporate investor views via **fully flexible probabilities** using entropy pooling and the **Black-Litterman** methodology  
- Support **shrinkage** and other **Bayesian estimation** methods  
- Support **variance-based**, **scenario-based CVaR**, and **robust optimization models**


## Installation

Py-vAllocation is under active development, so install from source:

```bash
git clone https://github.com/enexqnt/py-vallocation.git
cd py-vallocation
pip install .
```

## Quick Start

See [examples here](examples/)

## Requirements

- Python 3.8+
- numpy >= 1.20.0
- cvxopt >= 1.2.0
- pandas

## Development Status

**Alpha release**: Under active development. Many features are not yet implemented or fully tested. Breaking changes may occur without notice. Use at your own risk.

## Contributing

Contributions and feedback are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the GNU General Public License v3.0 License. See [LICENSE](LICENSE) for details.

## Credits

Some code, where explicitly stated, is adapted from [fortitudo-tech](https://github.com/fortitudo-tech/fortitudo.tech)

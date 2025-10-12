# Examples

This directory collects runnable scripts and Jupyter notebooks showing typical
allocation workflows.

Python scripts
--------------

- `mean_variance_frontier.py`: classical mean-variance frontier with summary statistics.
- `cvar_allocation.py`: minimum-CVaR portfolio and tangency portfolio on the CVaR frontier.
- `robust_frontier.py`: Meucci-style robust frontier parameterised by λ.
- `relaxed_risk_parity_frontier.py`: relaxed risk parity target sweep with per-point diagnostics.
- `discrete_allocation.py`: convert continuous weights into lot-sized trades.
- `portfolio_ensembles.py`: combine multiple frontiers using the ensembling helpers.

Execute a script directly after installing the project dependencies:

```bash
python examples/portfolio_ensembles.py
```

Notebooks
---------

- `Example_01.ipynb`: detailed walkthrough of the mean-variance API.
- `Bayesian.ipynb`, `Flexible_Views.ipynb`, `Simple_views_on_mean.ipynb`: view-based
  workflows that reuse `ETF_prices.csv`.

Launch `jupyter lab` or `jupyter notebook` in the project root to explore the notebooks.
Each notebook contains guidance and setup notes in the opening cells.

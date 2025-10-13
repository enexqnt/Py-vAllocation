Examples
========

Practical, runnable workflows live here. Use them as entry points and
as references for your own projects.

How to run
----------

```bash
# from the repository root
python -m pip install -e .[robust]

# run any example
python examples/quickstart_etf_allocation.py
```

Quickstarts
-----------

- `quickstart_etf_allocation.py` - end-to-end ETF allocation (moments -> frontiers -> ensemble -> discrete trades). 
  Writes results under `output/` with plots (`frontiers_vol.png`, `frontiers_cvar.png`) and CSVs.

Frontiers & risk models
-----------------------

- `mean_variance_frontier.py` - classical mean-variance frontier with tangency/min-risk picks.
- `cvar_allocation.py` - CVaR frontier (alpha=5%) and tangency portfolio.
- `robust_frontier.py` - Meucci-style robust frontier over lambda.
- `relaxed_risk_parity_frontier.py` - relaxed risk parity sweep with diagnostics.

Utilities
---------

- `discrete_allocation.py` - transform continuous weights into lot-sized share counts.
- `portfolio_ensembles.py` - stack/average multiple model selections into a single allocation.

Data
----

- `ETF_prices.csv` - small, self-contained dataset used by most examples.

Notebooks
---------

Launch Jupyter in the repo root and open any of the notebooks:

- `Example_01.ipynb` - getting started with the API.
- `Simple_views_on_mean.ipynb` - simple mean views.
- `Bayesian.ipynb` - Black-Litterman and Bayesian ideas.
- `Flexible_Views.ipynb` - entropy pooling and flexible constraints.

Tip: the `docs/` site mirrors these workflows in tutorial form if you prefer static pages.

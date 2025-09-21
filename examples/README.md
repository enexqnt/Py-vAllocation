# Examples

This directory collects a mix of scripts and notebooks that demonstrate common
allocation workflows.

- `mean_variance_frontier.py` is a lightweight script that constructs a
  classical mean-variance frontier using purely in-memory data.
- `Example_01.ipynb` provides a more detailed walkthrough of the
  mean-variance API.
- `Bayesian.ipynb`, `Flexible_Views.ipynb`, and `Simple_views_on_mean.ipynb`
  focus on incorporating views and Bayesian updates. They reuse a shared CSV
  dataset (`ETF_prices.csv`).

Run the Python script directly after installing the project dependencies:

```bash
python examples/mean_variance_frontier.py
```

For the notebooks, start `jupyter lab` or `jupyter notebook` in the project
root and open the file you are interested in. Each notebook includes inline
commentary to explain the scenario being modelled and any required setup
steps.

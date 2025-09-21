Crafting a Mean-Variance Frontier
=================================

This tutorial builds a classical mean-variance frontier and inspects a few
portfolios along the curve. It mirrors the script distributed in
``examples/mean_variance_frontier.py``.

1.  Load scenarios. Pandas objects are accepted directly and the asset names are
    inferred automatically.
2.  Wrap the data in :class:`pyvallocation.portfolioapi.AssetsDistribution`.
3.  Instantiate :class:`pyvallocation.portfolioapi.PortfolioWrapper` and choose
    your constraints.
4.  Generate the frontier and analyse the results.

.. code-block:: python

    import pandas as pd
    from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper

    scenarios = pd.read_csv("examples/ETF_prices.csv").pct_change().dropna()
    dist = AssetsDistribution(scenarios=scenarios)

    wrapper = PortfolioWrapper(dist)
    wrapper.set_constraints({"long_only": True, "total_weight": 1.0})

    frontier = wrapper.mean_variance_frontier(num_portfolios=10)
    tangency_w, tangency_r, tangency_sigma = frontier.get_tangency_portfolio(0.01)

    print(tangency_w.sort_values(ascending=False).head())
    print(f"Return: {tangency_r:.4%}  |  Risk: {tangency_sigma:.4%}")

The :class:`pyvallocation.portfolioapi.PortfolioFrontier` helper exposes several
convenience methods:

- :meth:`~pyvallocation.portfolioapi.PortfolioFrontier.get_min_risk_portfolio`
- :meth:`~pyvallocation.portfolioapi.PortfolioFrontier.get_max_return_portfolio`
- :meth:`~pyvallocation.portfolioapi.PortfolioFrontier.portfolio_at_risk_target`
- :meth:`~pyvallocation.portfolioapi.PortfolioFrontier.portfolio_at_return_target`

These methods return weight series together with the associated return and risk
metrics, ready to be visualised or exported.

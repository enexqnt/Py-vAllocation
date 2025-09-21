Targeting CVaR Risk
===================

This walkthrough shows how to compute a portfolio that minimises CVaR for a
given expected return target. The workflow reuses historical scenarios; if you
only have parametric data, the wrapper will automatically simulate scenarios
using the stored mean and covariance.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper

    returns = pd.read_csv("examples/ETF_prices.csv").pct_change().dropna()
    probabilities = np.full(len(returns), 1.0 / len(returns))

    distribution = AssetsDistribution(
        scenarios=returns,
        probabilities=probabilities.reshape(-1, 1),  # column vectors are flattened automatically
    )

    wrapper = PortfolioWrapper(distribution)
    wrapper.set_constraints({"long_only": True, "total_weight": 1.0})

    weights, expected_return, cvar = wrapper.mean_cvar_portfolio_at_return(
        return_target=0.005,
        alpha=0.05,
    )

    print(weights.round(4))
    print(f"Expected return: {expected_return:.4%} | CVaR: {cvar:.4%}")

The same :class:`~pyvallocation.portfolioapi.PortfolioWrapper` instance can be
reused to compute full CVaR frontiers via
:meth:`~pyvallocation.portfolioapi.PortfolioWrapper.mean_cvar_frontier`. For more
elaborate scenario management, look at :mod:`pyvallocation.utils.projection` for
bootstrapping and horizon projection utilities.

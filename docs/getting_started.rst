Getting Started
===============

Follow these steps to experiment with *Py-vAllocation* in a fresh Python
environment.

Installation
------------

.. code-block:: bash

    python -m pip install py-vallocation

The package requires ``numpy``, ``pandas`` and ``cvxopt``. If you are working in
an isolated environment, the command above will download all dependencies.

First Allocation
----------------

The snippet below computes a compact mean-variance frontier using the high-level
:class:`pyvallocation.portfolioapi.PortfolioWrapper` API.

.. code-block:: python

    import pandas as pd
    from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper

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

    frontier = wrapper.mean_variance_frontier(num_portfolios=5)
    weights, expected_return, risk = frontier.get_min_risk_portfolio()

    print(weights.round(4))
    print(f"Expected return: {expected_return:.4%} | Volatility: {risk:.4%}")

The examples directory contains additional notebooks and scripts that expand on
this workflow. See :doc:`tutorials/index` for guided walkthroughs.

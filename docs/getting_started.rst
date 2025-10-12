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

Optional extras for shrinkage-heavy research flows are available via

.. code-block:: bash

    python -m pip install "py-vallocation[robust]"

which installs the analytical nonlinear shrinkage backend.

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

Relaxed Risk Parity
-------------------

The relaxed risk parity solver (Gambeta & Kwon, 2020) is available through the
same wrapper. It automatically solves for the benchmark risk parity portfolio,
then sweeps relaxed targets sized by a multiplier ``m``:

.. code-block:: python

    import numpy as np
    from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper

    mu = np.array([0.08, 0.06, 0.05, 0.03])
    cov = np.array(
        [
            [0.090, 0.040, 0.025, 0.010],
            [0.040, 0.070, 0.020, 0.015],
            [0.025, 0.020, 0.060, 0.018],
            [0.010, 0.015, 0.018, 0.045],
        ]
    )

    dist = AssetsDistribution(mu=mu, cov=cov)
    wrapper = PortfolioWrapper(dist)
    wrapper.set_constraints({"long_only": True, "total_weight": 1.0})

    frontier = wrapper.relaxed_risk_parity_frontier(num_portfolios=4, max_multiplier=1.5, lambda_reg=0.25)
    weights = frontier.to_frame()
    print(weights.round(4))

Metadata attached to ``frontier.metadata`` documents the multiplier, target,
and objective value for each point – ideal inputs for dashboarding or plotting.

The :mod:`examples <pyvallocation.examples>` directory contains runnable scripts
that reproduce the main workflows:

* ``mean_variance_frontier.py`` – classical efficient frontier summary.
* ``cvar_allocation.py`` – minimum-CVaR portfolio and tangency allocation.
* ``robust_frontier.py`` – Meucci-style robust λ-frontier.
* ``relaxed_risk_parity_frontier.py`` – relaxed risk parity frontier across return targets.
* ``discrete_allocation.py`` – map continuous weights to share counts.
* ``portfolio_ensembles.py`` – blend frontiers via exposure averaging/stacking.

Run them directly from the repository root, e.g. ``python
examples/portfolio_ensembles.py``.

See :doc:`tutorials/index` for guided walkthroughs and additional context.

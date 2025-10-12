Relaxed Risk Parity Diagnostics
===============================

This walkthrough mirrors the ``relaxed_risk_parity.ipynb`` notebook. The use
case: a balanced multi-asset mandate where the client wants risk-parity as the
starting point but is willing to accept additional return targets via the
Gambeta–Kwon relaxed framework.

Setup
-----

.. code-block:: python

    import numpy as np
    import pandas as pd

    from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper

    mu = np.array([0.08, 0.06, 0.05, 0.03])
    cov = np.array([
        [0.090, 0.040, 0.025, 0.010],
        [0.040, 0.070, 0.020, 0.015],
        [0.025, 0.020, 0.060, 0.018],
        [0.010, 0.015, 0.018, 0.045],
    ])

    dist = AssetsDistribution(mu=mu, cov=cov, asset_names=["Tech", "Health", "Value", "Bonds"])
    wrapper = PortfolioWrapper(dist)
    wrapper.set_constraints({"long_only": True, "total_weight": 1.0})

Step 2 – Frontier and diagnostics
----------------------------------

.. code-block:: python

    frontier = wrapper.relaxed_risk_parity_frontier(
        num_portfolios=4,
        max_multiplier=1.5,
        lambda_reg=0.25,
    )

    diagnostics = pd.DataFrame(frontier.metadata)
    diagnostics["risk"] = frontier.risks
    diagnostics["expected_return"] = frontier.returns
    diagnostics

Step 3 – Visualise the frontier
--------------------------------

.. code-block:: python

    import matplotlib.pyplot as plt
    from pyvallocation.plotting import plot_frontiers

    ax = plot_frontiers(
        frontier,
        highlight=("min_risk", "max_return"),
        highlight_metadata_keys=("target_multiplier", "lambda_reg"),
    )
    ax.set_title("Relaxed Risk Parity Frontier")
    ax.figure

Step 4 – Interpreting the metadata
-----------------------------------

The ``frontier.metadata`` list records, for each portfolio:

* ``target_multiplier`` – the relaxed multiplier applied to the base risk-parity
  profile.
* ``lambda_reg`` – the diagonal penalty used during optimisation.
* ``target_return`` – the effective return constraint at the solution. If
  clipping occurred, this may differ from the originally requested target.

This enriched metadata is ideal for governance dashboards—combine it with the
``diagnostics`` DataFrame to monitor trade-offs between target multipliers and
achievable returns.

Step 5 – Deployment tips
-------------------------

- Replace the synthetic ``mu``/``cov`` with shrinkage estimates from
  :func:`pyvallocation.moments.estimate_moments` to incorporate estimation risk.
- Use :meth:`pyvallocation.portfolioapi.PortfolioFrontier.ensemble_average`
  to blend multiple columns (e.g., pure RP + relaxed RP) before presenting the
  recommendation.
- Store the solver output in ``frontier.metadata`` for auditability; it preserves
  all intermediate targets and penalty multipliers.

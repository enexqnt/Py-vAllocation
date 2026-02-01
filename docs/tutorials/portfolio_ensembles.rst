Portfolio Ensembling Across Models
==================================

This tutorial mirrors ``examples/portfolio_ensembles.py`` and shows how to
combine multiple model specifications into a single allocation while aligning
their risk level. The key idea is to select comparable portfolios (e.g. the
median-risk point by volatility) before averaging or exposure stacking.

Quick setup
-----------

.. code-block:: python

   from data_utils import load_returns
   from pyvallocation.ensembles import make_portfolio_spec, assemble_portfolio_ensemble

   returns = load_returns().iloc[-750:]

   # Align by volatility percentile to avoid mixing different risk levels.
   selector_kwargs = {"percentile": 0.5, "risk_label": "Volatility"}

   specs = [
       make_portfolio_spec(
           "Sample MV",
           returns=returns,
           mean_estimator="sample",
           cov_estimator="sample",
           optimiser="mean_variance",
           selector="risk_percentile",
           selector_kwargs=selector_kwargs,
       ),
       make_portfolio_spec(
           "Shrinkage MV",
           returns=returns,
           mean_estimator="jorion",
           cov_estimator="ledoit_wolf",
           optimiser="mean_variance",
           selector="risk_percentile",
           selector_kwargs=selector_kwargs,
       ),
       make_portfolio_spec(
           "CVaR",
           returns=returns,
           use_scenarios=True,
           optimiser="cvar",
           optimiser_kwargs={"alpha": 0.05},
           selector="risk_percentile",
           selector_kwargs=selector_kwargs,
       ),
   ]

   result = assemble_portfolio_ensemble(specs, ensemble=("average", "stack"), combine="selected")
   print(result.selections.round(4))
   print(result.average.round(4))
   print(result.stacked.round(4))

Interpretation & reporting
--------------------------

- Use ``risk_percentile`` or ``risk_target`` selectors to ensure *comparable* risk
  across heterogeneous frontiers (variance vs CVaR).
- ``result.selections`` is the audit trail of portfolios that were blended.
- ``result.average`` provides a simple blend; ``result.stacked`` dampens idiosyncratic
  exposures and often produces more stable weights.

.. note::

   The exposure stacking procedure follows Vorobets :cite:p:`vorobets2024derivatives`
   and the reference implementation published in the GPL-3 licensed
   `fortitudo.tech <https://github.com/fortitudo-tech/fortitudo.tech>`_ repository.

For a runnable end-to-end script, execute ``python examples/portfolio_ensembles.py``.

"""Blend multiple model specifications into an ensemble portfolio."""

from __future__ import annotations

from data_utils import load_returns
from pyvallocation.ensembles import assemble_portfolio_ensemble, make_portfolio_spec


def main() -> None:
    returns = load_returns().iloc[-750:]

    # Align portfolios by risk percentile (median risk) using volatility as the shared driver.
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

    print("=== Selected portfolios (aligned by risk percentile) ===")
    print(result.selections.round(4))

    print("\n=== Ensemble portfolios ===")
    for name, weights in result.ensembles.items():
        print(f"[{name}]")
        print(weights.round(4))


if __name__ == "__main__":
    main()

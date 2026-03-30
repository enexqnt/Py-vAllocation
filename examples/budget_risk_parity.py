"""Budget risk parity with custom risk contribution targets."""
from __future__ import annotations

import numpy as np
from pyvallocation import PortfolioWrapper


def main():
    import pandas as pd

    assets = ["Tech", "Health", "Value", "Bonds"]
    mu = pd.Series([0.08, 0.06, 0.04, 0.03], index=assets)
    cov = pd.DataFrame(
        [[0.040, 0.010, 0.005, 0.002],
         [0.010, 0.030, 0.008, 0.004],
         [0.005, 0.008, 0.025, 0.006],
         [0.002, 0.004, 0.006, 0.020]],
        index=assets, columns=assets,
    )

    wrapper = PortfolioWrapper.from_moments(mu, cov)

    # Equal risk contribution (default)
    w_erc, ret_erc, risk_erc = wrapper.relaxed_risk_parity_portfolio(
        lambda_reg=0.0, target_multiplier=None,
    )
    print("=== Equal Risk Contribution ===")
    print(w_erc.round(4))

    # Custom budgets: 50% risk from asset 0, 20% each from 1 and 2, 10% from 3
    budgets = np.array([0.50, 0.20, 0.20, 0.10])
    w_rb, ret_rb, risk_rb = wrapper.relaxed_risk_parity_portfolio(
        lambda_reg=0.0, target_multiplier=None, risk_budgets=budgets,
    )
    print("\n=== Budget Risk Parity (50/20/20/10) ===")
    print(w_rb.round(4))

    # Compare risk contributions
    _, _, _, diag_erc = wrapper.relaxed_risk_parity_portfolio_with_diagnostics(
        lambda_reg=0.0, target_multiplier=None,
    )
    _, _, _, diag_rb = wrapper.relaxed_risk_parity_portfolio_with_diagnostics(
        lambda_reg=0.0, target_multiplier=None, risk_budgets=budgets,
    )
    print("\nERC risk contributions (%):", diag_erc["risk_contributions_pct"].round(2))
    print("Budget RP risk contributions (%):", diag_rb["risk_contributions_pct"].round(2))


if __name__ == "__main__":
    main()

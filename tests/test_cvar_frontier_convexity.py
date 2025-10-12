import numpy as np
import pandas as pd

from pyvallocation.portfolioapi import AssetsDistribution, PortfolioWrapper


def is_convex_envelope(x: np.ndarray, y: np.ndarray, tol: float = 1e-8) -> bool:
    order = np.argsort(x)
    x, y = x[order], y[order]
    # Monotone in returns
    if np.any(np.diff(x) < -tol):
        return False
    # Check slopes non-decreasing (convex y(x))
    slopes = np.diff(y) / np.clip(np.diff(x), tol, None)
    return np.all(np.diff(slopes) >= -1e-7)


def test_mean_cvar_frontier_convex():
    rng = np.random.default_rng(42)
    T, N = 400, 5
    # Simulate light-tailed returns to keep solver stable/fast
    R = rng.normal(0.001, 0.02, size=(T, N))
    dist = AssetsDistribution(scenarios=pd.DataFrame(R))
    port = PortfolioWrapper(dist)
    port.set_constraints({"long_only": True, "total_weight": 1.0})

    front = port.mean_cvar_frontier(num_portfolios=13, alpha=0.05)
    x, y = np.asarray(front.returns), np.asarray(front.risks)

    assert is_convex_envelope(x, y)


"""Plotting helpers for robust optimization."""

from __future__ import annotations

import numpy as np


def plot_robust_frontier(mu_wc: np.ndarray, risk: np.ndarray, param: np.ndarray):
    """Plot robust efficient frontier with parameter annotations."""
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(risk, mu_wc, marker="o")
    for x, y, p in zip(risk, mu_wc, param):
        plt.annotate(f"{p:.2g}", (x, y), fontsize=8)
    plt.xlabel("Robust Risk")
    plt.ylabel("Worst-case Expected Return")
    plt.title("Robust-Bayesian Efficient Frontier")
    plt.grid(True)
    plt.show()

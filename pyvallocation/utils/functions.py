"""Portfolio Value-at-Risk (VaR) and Conditional VaR (CVaR) functions."""

from __future__ import annotations
import numpy as np
import pandas as pd

def portfolio_cvar(
    w: np.ndarray,
    R: np.ndarray | pd.DataFrame,
    p: np.ndarray | None = None,
    alpha: float | None = None,
    demean: bool | None = None,
) -> float | np.ndarray:
    """
    Computes portfolio Conditional Value-at-Risk (CVaR or Expected Shortfall).

    Args:
        w: Portfolio weights matrix (N, M). N=instruments, M=portfolios.
        R: Instrument P&L or returns matrix (T, N). T=scenarios.
        p: Scenario probability vector (T, 1). Defaults to uniform.
        alpha: Confidence level for CVaR. Defaults to 0.95.
        demean: If True, uses demeaned P&L. Defaults to False.

    Returns:
        The portfolio's alpha-CVaR, returned as a positive float or a 1xM array.
    """
    alpha = 0.95 if alpha is None else float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be a float in the interval (0, 1).")
    if demean is None:
        demean = False
    elif not isinstance(demean, bool):
        raise ValueError("demean must be either True or False.")

    R_arr = np.asarray(R, dtype=float)
    p = np.full((R_arr.shape[0], 1), 1.0 / R_arr.shape[0]) if p is None else np.asarray(p, dtype=float).reshape(-1, 1)

    if demean:
        R_arr -= p.T @ R_arr

    pf_pnl = R_arr @ w
    if pf_pnl.ndim == 1:
        pf_pnl = pf_pnl.reshape(-1, 1)

    order = np.argsort(pf_pnl, axis=0)
    sorted_pnl = np.take_along_axis(pf_pnl, order, axis=0)
    sorted_p = np.take_along_axis(np.broadcast_to(p, pf_pnl.shape), order, axis=0)
    var_indices = (np.cumsum(sorted_p, axis=0) >= (1.0 - alpha)).argmax(axis=0)
    var = np.take_along_axis(sorted_pnl, var_indices[np.newaxis, :], axis=0)

    tail_mask = pf_pnl <= var
    denominator = np.sum(p * tail_mask, axis=0)
    numerator = np.sum(p * pf_pnl * tail_mask, axis=0)
    cvar = np.full_like(denominator, np.nan)
    np.divide(numerator, denominator, out=cvar, where=denominator != 0)

    risk = -cvar.reshape(1, -1)
    return risk.item() if risk.size == 1 else risk


def portfolio_var(
    w: np.ndarray,
    R: np.ndarray | pd.DataFrame,
    p: np.ndarray | None = None,
    alpha: float | None = None,
    demean: bool | None = None,
) -> float | np.ndarray:
    """
    Computes portfolio Value-at-Risk (VaR).

    Args:
        w: Portfolio weights matrix (N, M). N=instruments, M=portfolios.
        R: Instrument P&L or returns matrix (T, N). T=scenarios.
        p: Scenario probability vector (T, 1). Defaults to uniform.
        alpha: Confidence level for VaR. Defaults to 0.95.
        demean: If True, uses demeaned P&L. Defaults to False.

    Returns:
        The portfolio's alpha-VaR, returned as a positive float or a 1xM array.
    """
    alpha = 0.95 if alpha is None else float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be a float in the interval (0, 1).")
    if demean is None:
        demean = False
    elif not isinstance(demean, bool):
        raise ValueError("demean must be either True or False.")

    R_arr = np.asarray(R, dtype=float)
    p = np.full((R_arr.shape[0], 1), 1.0 / R_arr.shape[0]) if p is None else np.asarray(p, dtype=float).reshape(-1, 1)

    if demean:
        R_arr -= p.T @ R_arr

    pf_pnl = R_arr @ w
    if pf_pnl.ndim == 1:
        pf_pnl = pf_pnl.reshape(-1, 1)

    order = np.argsort(pf_pnl, axis=0)
    sorted_pnl = np.take_along_axis(pf_pnl, order, axis=0)
    sorted_p = np.take_along_axis(np.broadcast_to(p, pf_pnl.shape), order, axis=0)
    var_indices = (np.cumsum(sorted_p, axis=0) >= (1.0 - alpha)).argmax(axis=0)
    var = np.take_along_axis(sorted_pnl, var_indices[np.newaxis, :], axis=0)

    risk = -var
    return risk.item() if risk.size == 1 else risk
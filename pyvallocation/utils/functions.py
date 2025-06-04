from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

def _return_portfolio_risk(risk: np.ndarray) -> float | np.ndarray:
    """Return scalar when matrix contains a single element, otherwise 1×N array."""
    return risk.item() if risk.size == 1 else risk

def _var_cvar_preprocess(
    e: np.ndarray,
    R: np.ndarray | pd.DataFrame,
    p: np.ndarray | None,
    alpha: float | None,
    demean: bool | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    alpha = 0.95 if alpha is None else float(alpha)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be a float in the interval (0, 1).")

    if demean is None:
        demean = True
    elif not isinstance(demean, bool):
        raise ValueError("demean must be either True or False.")

    if p is None:
        p = np.full((R.shape[0], 1), 1.0 / R.shape[0])
    else:
        p = np.asarray(p, float).reshape(-1, 1)

    R_arr = np.asarray(R, float)
    if demean:
        R_arr = R_arr - p.T @ R_arr

    pf_pnl = R_arr @ e

    return pf_pnl, p, alpha

def portfolio_cvar(
    e: np.ndarray,
    R: np.ndarray | pd.DataFrame,
    p: np.ndarray | None = None,
    alpha: float | None = None,
    demean: bool | None = None,
) -> float | np.ndarray:
    """Compute portfolio Conditional Value-at-Risk (CVaR).

    Args:
        e: Vector or matrix of portfolio exposures with shape (I, num_portfolios).
        R: P&L or risk factor simulation with shape (S, I).
        p: Probability vector with shape (S, 1). Defaults to uniform probabilities.
        alpha: Alpha level for alpha-CVaR. Defaults to 0.95.
        demean: Whether to use demeaned P&L. Defaults to True.

    Returns:
        Portfolio alpha-CVaR as a float or array.
    """
    pf_pnl, p, alpha = _var_cvar_preprocess(e, R, p, alpha, demean)
    var = _var_calc(pf_pnl, p, alpha)
    mask = pf_pnl <= var
    weighted = (p * pf_pnl) * mask
    denom = (p * mask).sum(axis=0)
    cvar = weighted.sum(axis=0) / denom
    return _return_portfolio_risk(-cvar.reshape(1, -1))

def _var_calc(pf_pnl: np.ndarray, p: np.ndarray, alpha: float) -> np.ndarray:
    num_portfolios = pf_pnl.shape[1]
    var = np.full((1, num_portfolios), np.nan)
    for i, pnl in enumerate(pf_pnl.T):
        order = np.argsort(pnl)
        probs = p[order, 0]
        idx = np.searchsorted(np.cumsum(probs) - probs / 2, 1 - alpha)
        lo = max(idx - 1, 0)
        var[0, i] = pnl[order[lo:idx + 1]].mean()
    return var

def portfolio_var(
    e: np.ndarray,
    R: np.ndarray | pd.DataFrame,
    p: np.ndarray | None = None,
    alpha: float | None = None,
    demean: bool | None = None,
) -> float | np.ndarray:
    """Compute portfolio Value-at-Risk (VaR).

    Args:
        e: Vector or matrix of portfolio exposures with shape (I, num_portfolios).
        R: P&L or risk factor simulation with shape (S, I).
        p: Probability vector with shape (S, 1). Defaults to uniform probabilities.
        alpha: Alpha level for alpha-VaR. Defaults to 0.95.
        demean: Whether to use demeaned P&L. Defaults to True.

    Returns:
        Portfolio alpha-VaR as a float or array.
    """
    pf_pnl, p, alpha = _var_cvar_preprocess(e, R, p, alpha, demean)
    var = _var_calc(pf_pnl, p, alpha)
    return _return_portfolio_risk(-var)

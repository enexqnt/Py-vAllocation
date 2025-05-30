from typing import Tuple, Union
import numpy as np
import pandas as pd

def _return_portfolio_risk(risk: np.ndarray) -> float | np.ndarray:
    if risk.shape[1] == 1:
        return risk[0, 0]
    return risk

def _var_cvar_preprocess(
    e: np.ndarray,
    R: np.ndarray | pd.DataFrame,
    p: np.ndarray | None,
    alpha: float | None,
    demean: bool | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if alpha is None:
        alpha = 0.95
    elif not isinstance(alpha, float) or not 0 < alpha < 1:
        raise ValueError("alpha must be a float in the interval (0, 1).")

    if demean is None:
        demean = True
    elif not isinstance(demean, bool):
        raise ValueError("demean must be either True or False.")

    if p is None:
        p = np.ones((R.shape[0], 1)) / R.shape[0]

    if demean:
        R = R - p.T @ R
    pf_pnl = R @ e

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
    num_portfolios = e.shape[1]
    cvar = np.full((1, num_portfolios), np.nan)
    for port in range(num_portfolios):
        cvar_idx = pf_pnl[:, port] <= var[0, port]
        cvar[0, port] = p[cvar_idx, 0].T @ pf_pnl[cvar_idx, port] / np.sum(p[cvar_idx, 0])
    return _return_portfolio_risk(-cvar)

def _var_calc(pf_pnl: np.ndarray, p: np.ndarray, alpha: float) -> np.ndarray:
    num_portfolios = pf_pnl.shape[1]
    var = np.full((1, num_portfolios), np.nan)
    for port in range(num_portfolios):
        idx_sorted = np.argsort(pf_pnl[:, port], axis=0)
        p_sorted = p[idx_sorted, 0]
        var_index = np.searchsorted(np.cumsum(p_sorted) - p_sorted / 2, 1 - alpha)
        var[0, port] = np.mean(pf_pnl[idx_sorted[var_index - 1 : var_index + 1], port])
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

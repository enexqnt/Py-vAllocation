# functions.py
"""Portfolio risk helper functions."""

from __future__ import annotations


import numpy as np
import pandas as pd


def _return_portfolio_risk(risk: np.ndarray) -> float | np.ndarray:
    """Return scalar when matrix contains a single element, otherwise 1xN array."""
    return risk.item() if risk.size == 1 else risk


def _var_cvar_preprocess(
    e: np.ndarray,
    R: np.ndarray | pd.DataFrame,
    p: np.ndarray | None,
    alpha: float | None,
    demean: bool | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Common preprocessing steps for VaR and CVaR."""
    alpha = 0.95 if alpha is None else float(alpha)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be a float in the interval (0, 1).")

    # --- FIX ---
    # The default for `demean` is now False to align with the optimizer's
    # definition of risk, which uses raw (non-demeaned) returns.
    if demean is None:
        demean = False
    elif not isinstance(demean, bool):
        raise ValueError("demean must be either True or False.")

    if p is None:
        # Defaults to uniform probabilities for scenarios
        p = np.full((R.shape[0], 1), 1.0 / R.shape[0])
    else:
        p = np.asarray(p, float).reshape(-1, 1)

    R_arr = np.asarray(R, float)
    if demean:
        # Demean returns only if explicitly requested
        R_arr = R_arr - p.T @ R_arr

    pf_pnl = R_arr @ e

    # Ensure pf_pnl is a 2D array to handle the single portfolio case correctly
    if pf_pnl.ndim == 1:
        pf_pnl = pf_pnl.reshape(-1, 1)

    return pf_pnl, p, alpha

def portfolio_cvar(
    e: np.ndarray,
    R: np.ndarray | pd.DataFrame,
    p: np.ndarray | None = None,
    alpha: float | None = None,
    demean: bool | None = None,
) -> float | np.ndarray:
    """
    Compute portfolio Conditional Value-at-Risk (CVaR).

    CVaR, or Expected Shortfall, is the expected loss given that the loss is
    greater than or equal to the Value-at-Risk (VaR). This implementation
    follows the definition of Rockafellar and Uryasev[cite: 6].

    Args:
        e: Vector or matrix of portfolio exposures with shape (I, num_portfolios).
        R: P&L or risk factor simulation with shape (S, I).
        p: Probability vector with shape (S, 1). Defaults to uniform probabilities.
        alpha: Confidence level for CVaR (e.g., 0.95 for 95% CVaR). Defaults to 0.95.
        demean: Whether to use demeaned P&L. Defaults to False to align with
                standard optimization formulations.

    Returns:
        Portfolio alpha-CVaR as a positive float or array.
    """
    # Preprocess inputs. Note: `demean` defaults to False inside this function.
    pf_pnl, p, alpha = _var_cvar_preprocess(e, R, p, alpha, demean)
    
    # 1. Calculate VaR (the loss threshold)
    var = _var_calc(pf_pnl, p, alpha)
    
    # 2. Identify all P&L scenarios that are less than or equal to the VaR
    mask = pf_pnl <= var
    
    # 3. Calculate the weighted average of these tail losses
    weighted_losses = (p * pf_pnl) * mask
    sum_of_tail_probs = (p * mask).sum(axis=0)
    
    # Avoid division by zero if no scenarios fall in the tail
    cvar = np.full(sum_of_tail_probs.shape, np.nan)
    np.divide(weighted_losses.sum(axis=0), sum_of_tail_probs, out=cvar, where=sum_of_tail_probs!=0)

    # Return CVaR as a positive number representing risk
    return _return_portfolio_risk(-cvar.reshape(1, -1))


def _var_calc(pf_pnl: np.ndarray, p: np.ndarray, alpha: float) -> np.ndarray:
    """
    --- REVISED ---
    Compute the historical VaR for each portfolio at a given alpha level.

    VaR is defined as the (1-alpha) quantile of the P&L distribution. This
    function correctly calculates the quantile for weighted scenarios.
    
    Args:
        pf_pnl: Portfolio P&L matrix with shape (S, num_portfolios).
        p: Probability vector with shape (S, 1).
        alpha: The confidence level (e.g., 0.95 for 95% VaR).

    Returns:
        The VaR for each portfolio as a 1xN array.
    """
    num_portfolios = pf_pnl.shape[1]
    var = np.full((1, num_portfolios), np.nan)
    quantile = 1 - alpha  # The tail probability cutoff (e.g., 0.05)

    for i, pnl_col in enumerate(pf_pnl.T):
        # Sort the P&L values and their corresponding probabilities
        order = np.argsort(pnl_col)
        sorted_pnl = pnl_col[order]
        sorted_p = p[order, 0]

        # Calculate the cumulative probability distribution
        cum_p = np.cumsum(sorted_p)

        # Find the first index where the cumulative probability is >= the target quantile.
        # This identifies the P&L value at the threshold of the tail.
        var_idx = np.searchsorted(cum_p, quantile, side='left')
        
        # Ensure index does not go out of bounds
        var_idx = min(var_idx, len(sorted_pnl) - 1)
        
        var[0, i] = sorted_pnl[var_idx]
        
    return var


def portfolio_var(
    e: np.ndarray,
    R: np.ndarray | pd.DataFrame,
    p: np.ndarray | None = None,
    alpha: float | None = None,
    demean: bool | None = None,
) -> float | np.ndarray:
    """
    Compute portfolio Value-at-Risk (VaR).

    Args:
        e: Vector or matrix of portfolio exposures with shape (I, num_portfolios).
        R: P&L or risk factor simulation with shape (S, I).
        p: Probability vector with shape (S, 1). Defaults to uniform probabilities.
        alpha: Confidence level for VaR (e.g., 0.95 for 95% VaR). Defaults to 0.95.
        demean: Whether to use demeaned P&L. Defaults to False.

    Returns:
        Portfolio alpha-VaR as a positive float or array.
    """
    # Preprocess inputs. Note: `demean` defaults to False inside this function.
    pf_pnl, p, alpha = _var_cvar_preprocess(e, R, p, alpha, demean)
    var = _var_calc(pf_pnl, p, alpha)
    
    # Return VaR as a positive number representing risk
    return _return_portfolio_risk(-var)
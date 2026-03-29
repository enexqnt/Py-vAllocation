"""Portfolio risk utilities (VaR, CVaR, variance, volatility)."""

from __future__ import annotations
import numpy as np
import pandas as pd

def _ensure_weights_matrix(w: np.ndarray | pd.Series | pd.DataFrame) -> np.ndarray:
    """Accept vector or matrix weights and return an ``(N, M)`` ndarray.

    Args:
        w: Weights vector or matrix.

    Returns:
        np.ndarray: Weights as a 2D array with shape ``(N, M)``.
    """
    arr = np.asarray(w, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr

def portfolio_variance(
    w: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
) -> float | np.ndarray:
    """
    Compute portfolio variance for one or many portfolios.

    Args:
        w: Weights vector ``(N,)`` or matrix ``(N, M)``.
        cov: Covariance matrix ``(N, N)``.

    Returns:
        Scalar variance (single portfolio) or ``(M,)`` array for multiple portfolios.
    """
    W = _ensure_weights_matrix(w)
    cov_np = np.asarray(cov, dtype=float)
    quad = np.sum((W.T @ cov_np) * W.T, axis=1)
    return quad.item() if quad.size == 1 else quad

def portfolio_volatility(
    w: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
) -> float | np.ndarray:
    """
    Compute portfolio volatility (standard deviation).

    Args:
        w: Weights vector ``(N,)`` or matrix ``(N, M)``.
        cov: Covariance matrix ``(N, N)``.
    """
    var = portfolio_variance(w, cov)
    return float(np.sqrt(var)) if np.isscalar(var) else np.sqrt(var)

def portfolio_cvar(
    w: np.ndarray,
    R: np.ndarray | pd.DataFrame,
    p: np.ndarray | None = None,
    confidence: float | None = None,
    demean: bool | None = None,
) -> float | np.ndarray:
    """
    Computes portfolio Conditional Value-at-Risk (CVaR or Expected Shortfall).

    Args:
        w: Portfolio weights matrix (N, M). N=instruments, M=portfolios.
        R: Instrument P&L or returns matrix (T, N). T=scenarios.
        p: Scenario probability vector (T, 1). Defaults to uniform.
        confidence: Confidence level for CVaR (e.g. 0.95 means 5% tail).
            Defaults to 0.95.
        demean: If True, uses demeaned P&L. Defaults to False.

    Returns:
        The portfolio's CVaR, returned as a positive float or a 1xM array.
    """
    conf = 0.95 if confidence is None else float(confidence)
    if not 0.0 < conf < 1.0:
        raise ValueError("confidence must be a float in the interval (0, 1).")
    if demean is None:
        demean = False
    elif not isinstance(demean, bool):
        raise ValueError("demean must be either True or False.")

    R_arr = np.asarray(R, dtype=float)
    p = np.full((R_arr.shape[0], 1), 1.0 / R_arr.shape[0]) if p is None else np.asarray(p, dtype=float).reshape(-1, 1)

    if demean:
        R_arr = R_arr - p.T @ R_arr

    W = _ensure_weights_matrix(w)
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        pf_pnl = R_arr @ W
    if pf_pnl.ndim == 1:
        pf_pnl = pf_pnl.reshape(-1, 1)

    order = np.argsort(pf_pnl, axis=0)
    sorted_pnl = np.take_along_axis(pf_pnl, order, axis=0)
    sorted_p = np.take_along_axis(np.broadcast_to(p, pf_pnl.shape), order, axis=0)
    var_indices = (np.cumsum(sorted_p, axis=0) >= (1.0 - conf)).argmax(axis=0)
    var = np.take_along_axis(sorted_pnl, var_indices[np.newaxis, :], axis=0)

    # Exact Rockafellar-Uryasev discrete CVaR with quantile boundary correction.
    # Handles probability mass concentrated at VaR by splitting strict-tail
    # from the at-boundary contribution.
    tail_alpha = 1.0 - conf
    strict_mask = pf_pnl < var            # strictly below VaR
    strict_prob = np.sum(p * strict_mask, axis=0)
    strict_sum = np.sum(p * pf_pnl * strict_mask, axis=0)
    # Exact: CVaR = -(1/alpha) * [E[PnL * 1{PnL<VaR}] + VaR * (alpha - P(PnL<VaR))]
    cvar = (strict_sum + var.reshape(-1) * (tail_alpha - strict_prob)) / tail_alpha

    risk = -cvar.reshape(-1)
    return risk.item() if risk.size == 1 else risk


def portfolio_var(
    w: np.ndarray,
    R: np.ndarray | pd.DataFrame,
    p: np.ndarray | None = None,
    confidence: float | None = None,
    demean: bool | None = None,
) -> float | np.ndarray:
    """
    Computes portfolio Value-at-Risk (VaR).

    Args:
        w: Portfolio weights matrix (N, M). N=instruments, M=portfolios.
        R: Instrument P&L or returns matrix (T, N). T=scenarios.
        p: Scenario probability vector (T, 1). Defaults to uniform.
        confidence: Confidence level for VaR (e.g. 0.95 means 5% tail).
            Defaults to 0.95.
        demean: If True, uses demeaned P&L. Defaults to False.

    Returns:
        The portfolio's VaR, returned as a positive float or a 1xM array.
    """
    conf = 0.95 if confidence is None else float(confidence)
    if not 0.0 < conf < 1.0:
        raise ValueError("confidence must be a float in the interval (0, 1).")
    if demean is None:
        demean = False
    elif not isinstance(demean, bool):
        raise ValueError("demean must be either True or False.")

    R_arr = np.asarray(R, dtype=float)
    p = np.full((R_arr.shape[0], 1), 1.0 / R_arr.shape[0]) if p is None else np.asarray(p, dtype=float).reshape(-1, 1)

    if demean:
        R_arr = R_arr - p.T @ R_arr

    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        pf_pnl = R_arr @ w
    if pf_pnl.ndim == 1:
        pf_pnl = pf_pnl.reshape(-1, 1)

    order = np.argsort(pf_pnl, axis=0)
    sorted_pnl = np.take_along_axis(pf_pnl, order, axis=0)
    sorted_p = np.take_along_axis(np.broadcast_to(p, pf_pnl.shape), order, axis=0)
    var_indices = (np.cumsum(sorted_p, axis=0) >= (1.0 - conf)).argmax(axis=0)
    var = np.take_along_axis(sorted_pnl, var_indices[np.newaxis, :], axis=0)

    risk = -var
    return risk.item() if risk.size == 1 else risk

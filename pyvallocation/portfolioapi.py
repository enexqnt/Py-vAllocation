from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

# --- Assumed to be available from other modules ---
from .optimization import MeanCVaR, MeanVariance, RobustOptimizer
from .probabilities import generate_uniform_probabilities
from .utils.constraints import build_G_h_A_b
from .utils.functions import portfolio_cvar

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='[%(name)s - %(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class AssetsDistribution:
    """
    An immutable container for asset return distributions.

    This class holds either the parametric representation (mean and covariance)
    or the scenario-based representation (scenarios and probabilities) of
    asset returns. It performs extensive validation to ensure data consistency.

    Attributes:
        mu (Optional[np.ndarray]): 1D array of expected returns (N,).
        cov (Optional[np.ndarray]): 2D array of covariance matrix (N, N).
        scenarios (Optional[np.ndarray]): 2D array of scenarios (T, N).
        probabilities (Optional[np.ndarray]): 1D array of probabilities (T,).
        N (int): Number of assets (calculated).
        T (Optional[int]): Number of scenarios (calculated).
        asset_names (List[str]): Names of the assets, inferred from pandas inputs or provided.
    """
    mu: Optional[Union[npt.NDArray[np.floating], pd.Series]] = None
    cov: Optional[Union[npt.NDArray[np.floating], pd.DataFrame]] = None
    scenarios: Optional[Union[npt.NDArray[np.floating], pd.DataFrame]] = None
    probabilities: Optional[Union[npt.NDArray[np.floating], pd.Series]] = None
    asset_names: Optional[List[str]] = None
    
    # FIX: Declare N and T as non-initialized fields to work with slots=True
    N: int = field(init=False, repr=False)
    T: Optional[int] = field(init=False, repr=False)


    def __post_init__(self):
        # Use object.__setattr__ as the dataclass is frozen
        mu, cov, scenarios, probs = self.mu, self.cov, self.scenarios, self.probabilities
        asset_names = self.asset_names
        
        # Infer asset names and convert pandas objects to numpy arrays
        if isinstance(mu, pd.Series):
            asset_names = mu.index.tolist()
            mu = mu.values
        if isinstance(cov, pd.DataFrame):
            if asset_names is None:
                asset_names = cov.index.tolist()
            elif asset_names != cov.index.tolist():
                raise ValueError("Inconsistent asset names between mu and cov.")
            cov = cov.values
        if isinstance(scenarios, pd.DataFrame):
            if asset_names is None:
                asset_names = scenarios.columns.tolist()
            elif asset_names != scenarios.columns.tolist():
                 raise ValueError("Inconsistent asset names in inputs.")
            scenarios = scenarios.values
        if isinstance(probs, pd.Series):
            probs = probs.values

        # --- Validation and final attribute setting ---
        if mu is not None and cov is not None:
            mu, cov = np.asarray(mu, dtype=float), np.asarray(cov, dtype=float)
            if mu.ndim != 1: raise ValueError("`mu` must be a 1D array.")
            if cov.ndim != 2: raise ValueError("`cov` must be a 2D array.")
            if mu.shape[0] != cov.shape[0] or cov.shape[0] != cov.shape[1]:
                raise ValueError("Inconsistent shapes for mu and cov.")
            object.__setattr__(self, 'N', mu.shape[0])
            object.__setattr__(self, 'T', None)
        elif scenarios is not None:
            scenarios = np.asarray(scenarios, dtype=float)
            if scenarios.ndim != 2: raise ValueError("`scenarios` must be a 2D array (T, N).")
            T_val, N_val = scenarios.shape
            if probs is None:
                probs = generate_uniform_probabilities(T_val)
            else:
                probs = np.asarray(probs, dtype=float)
                if probs.shape != (T_val,): raise ValueError("Probabilities must match the number of scenarios.")
                if not np.isclose(np.sum(probs), 1.0):
                    logger.warning("Probabilities do not sum to 1. Normalizing.")
                    probs /= np.sum(probs)
            object.__setattr__(self, 'N', N_val)
            object.__setattr__(self, 'T', T_val)
        else:
            raise ValueError("Provide either (mu, cov) or (scenarios).")

        if self.N == 0:
            raise ValueError("Number of assets (N) cannot be zero.")
            
        if asset_names is not None and len(asset_names) != self.N:
            raise ValueError("`asset_names` must have the same length as the number of assets (N).")
        
        object.__setattr__(self, 'mu', mu)
        object.__setattr__(self, 'cov', cov)
        object.__setattr__(self, 'scenarios', scenarios)
        object.__setattr__(self, 'probabilities', probs)
        object.__setattr__(self, 'asset_names', asset_names)


@dataclass(frozen=True)
class PortfolioFrontier:
    """
    Represents an efficient frontier of portfolios.

    This is an immutable container for the results of an optimization that
    produces a set of efficient portfolios. It provides convenient methods
    for analyzing the frontier.
    """
    weights: npt.NDArray[np.floating]
    returns: npt.NDArray[np.floating]
    risks: npt.NDArray[np.floating]
    risk_measure: str
    asset_names: Optional[List[str]] = None

    def _to_pandas(self, w: np.ndarray, name: str) -> pd.Series:
        """Converts a numpy array to a pandas Series, using asset_names as the index."""
        return pd.Series(w, index=self.asset_names, name=name)

    def get_min_risk_portfolio(self) -> Tuple[pd.Series, float, float]:
        """Finds the portfolio with the minimum risk on this frontier."""
        min_risk_idx = np.argmin(self.risks)
        w = self.weights[:, min_risk_idx]
        ret, risk = self.returns[min_risk_idx], self.risks[min_risk_idx]
        return self._to_pandas(w, "Min Risk Portfolio"), ret, risk

    def get_max_return_portfolio(self) -> Tuple[pd.Series, float, float]:
        """Finds the portfolio with the maximum return on this frontier."""
        max_ret_idx = np.argmax(self.returns)
        w = self.weights[:, max_ret_idx]
        ret, risk = self.returns[max_ret_idx], self.risks[max_ret_idx]
        return self._to_pandas(w, "Max Return Portfolio"), ret, risk

    def get_tangency_portfolio(self, risk_free_rate: float) -> Tuple[pd.Series, float, float]:
        """Calculates the tangency portfolio (maximum Sharpe ratio)."""
        if np.all(np.isclose(self.risks, 0)):
             logger.warning("All portfolios on the frontier have zero risk. Sharpe ratio is undefined.")
             nan_weights = np.full(self.weights.shape[0], np.nan)
             return self._to_pandas(nan_weights, "Undefined"), np.nan, np.nan
        
        with np.errstate(divide='ignore', invalid='ignore'):
            sharpe_ratios = (self.returns - risk_free_rate) / self.risks
        sharpe_ratios[~np.isfinite(sharpe_ratios)] = -np.inf

        tangency_idx = np.argmax(sharpe_ratios)
        w, ret, risk = self.weights[:, tangency_idx], self.returns[tangency_idx], self.risks[tangency_idx]
        return self._to_pandas(w, f"Tangency Portfolio (rf={risk_free_rate:.2%})"), ret, risk

    def portfolio_at_risk_target(self, max_risk: float) -> Tuple[pd.Series, float, float]:
        """Finds the portfolio that maximizes return for a risk level at or below `max_risk`."""
        feasible_indices = np.where(self.risks <= max_risk)[0]
        if feasible_indices.size == 0:
            nan_weights = np.full(self.weights.shape[0], np.nan)
            return self._to_pandas(nan_weights, "Infeasible"), np.nan, np.nan
        
        optimal_idx = feasible_indices[np.argmax(self.returns[feasible_indices])]
        w, ret, risk = self.weights[:, optimal_idx], self.returns[optimal_idx], self.risks[optimal_idx]
        return self._to_pandas(w, f"Portfolio (Risk <= {max_risk:.4f})"), ret, risk

    def portfolio_at_return_target(self, min_return: float) -> Tuple[pd.Series, float, float]:
        """Finds the portfolio that minimizes risk for a return level at or above `min_return`."""
        feasible_indices = np.where(self.returns >= min_return)[0]
        if feasible_indices.size == 0:
            nan_weights = np.full(self.weights.shape[0], np.nan)
            return self._to_pandas(nan_weights, "Infeasible"), np.nan, np.nan

        optimal_idx = feasible_indices[np.argmin(self.risks[feasible_indices])]
        w, ret, risk = self.weights[:, optimal_idx], self.returns[optimal_idx], self.risks[optimal_idx]
        return self._to_pandas(w, f"Portfolio (Return >= {min_return:.4f})"), ret, risk


class PortfolioWrapper:
    """
    A high-level interface for guided portfolio construction and optimization.

    Workflow:
    1.  **Initialize**: `PortfolioWrapper(AssetsDistribution(...))`
    2.  **Set Constraints**: `port.set_constraints(...)`
    3.  **Compute**: Call a method like `mean_variance_frontier()` or `solve_robust_gamma_portfolio()`.
    4.  **Analyze**: Use the returned `PortfolioFrontier` object or result tuple.
    """
    def __init__(self, distribution: AssetsDistribution):
        self.dist = distribution
        self.G: Optional[np.ndarray] = None
        self.h: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        logger.info(f"PortfolioWrapper initialized for {self.dist.N} assets.")

    def set_constraints(self, params: Dict[str, Any]):
        """Builds and sets linear constraints for the portfolio."""
        logger.info(f"Setting constraints with parameters: {params}")
        try:
            G, h, A, b = build_G_h_A_b(self.dist.N, **params)
            self.G, self.h = np.atleast_2d(G), np.atleast_1d(h)
            self.A, self.b = np.atleast_2d(A), np.atleast_1d(b)
        except Exception as e:
            logger.error(f"Failed to build constraints: {e}", exc_info=True)
            raise RuntimeError(f"Constraint building failed: {e}") from e

    def _ensure_default_constraints(self):
        """Sets default long-only, fully invested constraints if none are set."""
        if self.G is None and self.A is None:
            logger.warning("No constraints set. Applying default: long-only and sum-to-one.")
            self.set_constraints({'long_only': True, 'total_weight': 1.0})

    def mean_variance_frontier(self, num_portfolios: int = 20) -> PortfolioFrontier:
        """Computes the classical Mean-Variance efficient frontier[cite: 7]."""
        if self.dist.mu is None or self.dist.cov is None:
            raise ValueError("Mean-Variance optimization requires `mu` and `cov`.")
        self._ensure_default_constraints()
        
        optimizer = MeanVariance(self.dist.mu, self.dist.cov, self.G, self.h, self.A, self.b)
        weights = optimizer.efficient_frontier(num_portfolios)
        returns = self.dist.mu @ weights
        risks = np.sqrt(np.sum((weights.T @ self.dist.cov) * weights.T, axis=1))
        
        logger.info(f"Successfully computed Mean-Variance frontier with {weights.shape[1]} portfolios.")
        return PortfolioFrontier(
            weights=weights, returns=returns, risks=risks,
            risk_measure='Volatility', asset_names=self.dist.asset_names
        )
        
    def mean_cvar_frontier(self, num_portfolios: int = 20, alpha: float = 0.05) -> PortfolioFrontier:
        """Computes the Mean-CVaR efficient frontier following Rockafellar and Uryasev[cite: 6]."""
        scenarios, probs = self.dist.scenarios, self.dist.probabilities
        if scenarios is None:
            if self.dist.mu is None or self.dist.cov is None:
                raise ValueError("Cannot simulate scenarios for CVaR without `mu` and `cov`.")
            logger.info("Simulating 5000 scenarios for CVaR assuming a Normal distribution.")
            scenarios = np.random.multivariate_normal(self.dist.mu, self.dist.cov, 5000)
            probs = generate_uniform_probabilities(5000)

        mu_for_frontier = self.dist.mu if self.dist.mu is not None else np.mean(scenarios, axis=0)

        self._ensure_default_constraints()
        optimizer = MeanCVaR(R=scenarios, p=probs, alpha=alpha, G=self.G, h=self.h, A=self.A, b=self.b)
        weights = optimizer.efficient_frontier(num_portfolios)
        returns = mu_for_frontier @ weights
        risks = abs(np.array([portfolio_cvar(w, scenarios, probs, alpha) for w in weights.T]))

        logger.info(f"Successfully computed Mean-CVaR frontier with {weights.shape[1]} portfolios.")
        return PortfolioFrontier(
            weights=weights, returns=returns, risks=risks,
            risk_measure=f'CVaR (alpha={alpha:.2f})', asset_names=self.dist.asset_names
        )

    def robust_lambda_frontier(self, num_portfolios: int = 20, max_lambda: float = 2.0) -> PortfolioFrontier:
        """Computes the robust frontier using the λ-variant from Meucci's framework[cite: 1]."""
        if self.dist.mu is None or self.dist.cov is None:
            raise ValueError("Robust optimization requires `mu` (μ₁) and `cov` (Σ₁).")
        logger.info(
            "Computing robust λ-frontier. Assuming `dist.mu` is posterior mean (μ₁) "
            "and `dist.cov` is posterior scale matrix (Σ₁), used as uncertainty covariance (Σ')."
        )
        self._ensure_default_constraints()
        
        optimizer = RobustOptimizer(
            expected_return=self.dist.mu,
            uncertainty_covariance=self.dist.cov,
            G=self.G, h=self.h, A=self.A, b=self.b
        )
        
        lambdas = np.linspace(0, max_lambda, num_portfolios)
        returns, risks, weights = optimizer.efficient_frontier(lambdas)

        logger.info(f"Successfully computed Robust λ-frontier with {weights.shape[1]} portfolios.")
        return PortfolioFrontier(
            weights=np.array(weights), returns=np.array(returns), risks=np.array(risks),
            risk_measure="Estimation Risk (‖Σ'¹/²w‖₂)", asset_names=self.dist.asset_names
        )

    def solve_robust_gamma_portfolio(self, gamma_mu: float, gamma_sigma_sq: float) -> Tuple[pd.Series, float, float]:
        """Solves the γ-variant robust problem from Meucci's framework[cite: 1]."""
        if self.dist.mu is None or self.dist.cov is None:
            raise ValueError("Robust optimization requires `mu` (μ₁) and `cov` (Σ₁).")
        logger.info(
            "Solving robust γ-portfolio. Assuming `dist.mu` is posterior mean (μ₁) "
            "and `dist.cov` is posterior scale matrix (Σ₁), used as uncertainty covariance (Σ')."
        )
        self._ensure_default_constraints()

        optimizer = RobustOptimizer(
            expected_return=self.dist.mu,
            uncertainty_covariance=self.dist.cov,
            G=self.G, h=self.h, A=self.A, b=self.b
        )

        result = optimizer.solve_gamma_variant(gamma_mu, gamma_sigma_sq)
        
        w_series = pd.Series(result.weights, index=self.dist.asset_names, name="Robust Gamma Portfolio")
            
        logger.info(
            f"Successfully solved robust γ-portfolio. "
            f"Nominal Return: {result.nominal_return:.4f}, Estimation Risk: {result.risk:.4f}"
        )
        return w_series, result.nominal_return, result.risk
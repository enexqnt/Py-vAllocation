from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

# --- Assumed to be available from other modules ---
from .moments import estimate_sample_moments
from .optimization import MeanCVaR, MeanVariance, RobustOptimizer
from .probabilities import generate_uniform_probabilities
from .utils.constraints import build_G_h_A_b
from .utils.functions import portfolio_cvar

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='[%(name)s - %(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssetsDistribution:
    """
    An immutable container for asset return distributions.

    This class validates and stores the statistical properties of assets, which can
    be represented either parametrically (mean and covariance) or non-parametrically
    (scenarios and their probabilities). It automatically handles both NumPy arrays
    and pandas Series/DataFrames, ensuring data consistency.

    Attributes:
        mu (Optional[Union[npt.NDArray[np.floating], pd.Series]]): A 1D array or pandas.Series of expected returns for each asset (N,).
        cov (Optional[Union[npt.NDArray[np.floating], pd.DataFrame]]): A 2D covariance matrix of asset returns (N, N).
        scenarios (Optional[Union[npt.NDArray[np.floating], pd.DataFrame]]): A 2D array or pandas.DataFrame of shape (T, N), where each row is a market scenario.
        probabilities (Optional[Union[npt.NDArray[np.floating], pd.Series]]): A 1D array or pandas.Series of probabilities corresponding to each scenario (T,).
        asset_names (Optional[List[str]]): A list of names for the assets. If not provided, inferred from pandas inputs.
        N (int): The number of assets, inferred from the input data.
        T (Optional[int]): The number of scenarios, inferred from the input data. None if parametric distribution is used.

    Assumptions & Design Choices:
        - If "scenarios" are provided without "probabilities", probabilities are
          assumed to be uniform across all scenarios.
        - If `scenarios` are provided but `mu` and `cov` are not, the mean and covariance
          will be estimated from the scenarios, accompanied by a warning.
        - If provided "probabilities" do not sum to 1.0, they are automatically
          normalized with a warning. This choice ensures downstream solvers
          receive valid probability distributions.
        - If pandas objects are used for inputs, asset names are inferred from
          their indices or columns. It is assumed that the order and names are
          consistent across all provided pandas objects.
    """
    mu: Optional[Union[npt.NDArray[np.floating], pd.Series]] = None
    cov: Optional[Union[npt.NDArray[np.floating], pd.DataFrame]] = None
    scenarios: Optional[Union[npt.NDArray[np.floating], pd.DataFrame]] = None
    probabilities: Optional[Union[npt.NDArray[np.floating], pd.Series]] = None
    asset_names: Optional[List[str]] = None
    N: int = field(init=False, repr=False)
    T: Optional[int] = field(init=False, repr=False)

    def __post_init__(self):
        """
        Validates inputs and initializes calculated fields after dataclass initialization.

        This method performs checks on the consistency of provided `mu`, `cov`,
        `scenarios`, and `probabilities`. It infers the number of assets (N)
        and scenarios (T), and handles the conversion of pandas inputs to
        NumPy arrays internally while preserving asset names. Probabilities
        are normalized if they do not sum to one.

        Raises:
            ValueError: If input parameters have inconsistent shapes or if insufficient
                        data is provided (i.e., neither (mu, cov) nor scenarios).
        """
        # Use object.__setattr__ as the dataclass is frozen
        mu, cov, scenarios, probs = self.mu, self.cov, self.scenarios, self.probabilities
        asset_names = self.asset_names
        
        # Infer asset names from pandas objects if not explicitly provided
        if isinstance(mu, pd.Series):
            if asset_names is None:
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

        # Initialize N and T
        N, T = None, None

        # Process scenarios and estimate moments if needed
        if scenarios is not None:
            scenarios = np.asarray(scenarios, dtype=float)
            if scenarios.ndim != 2: raise ValueError("`scenarios` must be a 2D array (T, N).")
            T, N = scenarios.shape
            
            if probs is None:
                logger.info("No probabilities provided for scenarios. Assuming a uniform distribution.")
                probs = generate_uniform_probabilities(T)
            else:
                probs = np.asarray(probs, dtype=float)
                if probs.shape != (T,): raise ValueError(f"Probabilities shape mismatch: expected ({T},), got {probs.shape}.")
                prob_sum = np.sum(probs)
                if not np.isclose(prob_sum, 1.0):
                    logger.warning(f"Probabilities sum to {prob_sum:.4f}, not 1.0. Normalizing to enforce valid distribution.")
                    probs /= prob_sum
            
            if mu is None or cov is None:
                logger.warning(
                    "`mu` and/or `cov` were not provided. Estimating from scenarios. "
                )
                estimated_mu, estimated_cov = estimate_sample_moments(scenarios, probs)
                if mu is None:
                    mu = estimated_mu
                if cov is None:
                    cov = estimated_cov
        
        # Process and validate parametric inputs (mu, cov)
        if mu is not None and cov is not None:
            mu, cov = np.asarray(mu, dtype=float), np.asarray(cov, dtype=float)
            if mu.ndim != 1: raise ValueError("`mu` must be a 1D array.")
            if cov.ndim != 2: raise ValueError("`cov` must be a 2D array.")
            if mu.shape[0] != cov.shape[0] or cov.shape[0] != cov.shape[1]:
                raise ValueError(f"Inconsistent shapes for mu {mu.shape} and cov {cov.shape}.")
            
            current_N = mu.shape[0]
            if N is not None and N != current_N:
                raise ValueError(f"Shape mismatch: N from scenarios is {N}, but from mu/cov is {current_N}.")
            N = current_N
        
        # Final validation
        if N is None:
            raise ValueError("Insufficient data. Provide either (mu, cov) or (scenarios).")

        if N == 0:
            raise ValueError("Number of assets (N) cannot be zero.")
            
        if asset_names is not None and len(asset_names) != N:
            raise ValueError(f"`asset_names` must have length {N}, but has length {len(asset_names)}.")
        
        # Set final attributes
        object.__setattr__(self, 'N', N)
        object.__setattr__(self, 'T', T)
        object.__setattr__(self, 'mu', mu)
        object.__setattr__(self, 'cov', cov)
        object.__setattr__(self, 'scenarios', scenarios)
        object.__setattr__(self, 'probabilities', probs)
        object.__setattr__(self, 'asset_names', asset_names)

@dataclass(frozen=True)
class PortfolioFrontier:
    """
    Represents an efficient frontier of optimal portfolios.

    This immutable container holds the results of an optimization run that
    generates a series of efficient portfolios. It provides methods to easily
    query and analyze specific portfolios on the frontier.

    Attributes:
        weights (npt.NDArray[np.floating]): A 2D NumPy array of shape (N, M), where N is the
            number of assets and M is the number of portfolios on the frontier. Each column represents the weights of an optimal portfolio.
        returns (npt.NDArray[np.floating]): A 1D NumPy array of shape (M,) containing the expected returns for each portfolio on the frontier.
        risks (npt.NDArray[np.floating]): A 1D NumPy array of shape (M,) containing the risk values for each portfolio on the frontier. The specific risk measure (e.g., volatility, CVaR, uncertainty budget) is indicated by `risk_measure`.
        risk_measure (str): A string describing the risk measure used to construct this efficient frontier (e.g., 'Volatility', 'CVaR (alpha=0.05)', 'Estimation Risk (‖Σ'¹/²w‖₂)').
        asset_names (Optional[List[str]]): An optional list of names for the assets. If provided, enables pandas Series/DataFrame output for portfolio weights.
    """
    weights: npt.NDArray[np.floating]
    returns: npt.NDArray[np.floating]
    risks: npt.NDArray[np.floating]
    risk_measure: str
    asset_names: Optional[List[str]] = None

    def _to_pandas(self, w: np.ndarray, name: str) -> pd.Series:
        return pd.Series(w, index=self.asset_names, name=name)

    def get_min_risk_portfolio(self) -> Tuple[pd.Series, float, float]:
        """
        Finds the portfolio with the minimum risk on the efficient frontier.

        Returns:
            Tuple[pd.Series, float, float]: A tuple containing:
                -   **weights** (:class:`pandas.Series`): The weights of the minimum risk portfolio.
                -   **returns** (float): The expected return of the minimum risk portfolio.
                -   **risk** (float): The risk of the minimum risk portfolio.
        """
        min_risk_idx = np.argmin(self.risks)
        w = self.weights[:, min_risk_idx]
        ret, risk = self.returns[min_risk_idx], self.risks[min_risk_idx]
        return self._to_pandas(w, "Min Risk Portfolio"), ret, risk

    def get_max_return_portfolio(self) -> Tuple[pd.Series, float, float]:
        """
        Finds the portfolio with the maximum expected return on the efficient frontier.

        Returns:
            Tuple[pd.Series, float, float]: A tuple containing:
                -   **weights** (:class:`pandas.Series`): The weights of the maximum return portfolio.
                -   **returns** (float): The expected return of the maximum return portfolio.
                -   **risk** (float): The risk of the maximum return portfolio.
        """
        max_ret_idx = np.argmax(self.returns)
        w = self.weights[:, max_ret_idx]
        ret, risk = self.returns[max_ret_idx], self.risks[max_ret_idx]
        return self._to_pandas(w, "Max Return Portfolio"), ret, risk

    def get_tangency_portfolio(self, risk_free_rate: float) -> Tuple[pd.Series, float, float]:
        """
        Calculates the tangency portfolio, which represents the portfolio with the maximum Sharpe ratio.

        The Sharpe ratio is defined as (portfolio_return - risk_free_rate) / portfolio_risk.

        Args:
            risk_free_rate (float): The risk-free rate of return.

        Returns:
            Tuple[pd.Series, float, float]: A tuple containing:
                -   **weights** (:class:`pandas.Series`): The weights of the tangency portfolio.
                -   **returns** (float): The expected return of the tangency portfolio.
                -   **risk** (float): The risk of the tangency portfolio.
        """
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
        """
        Finds the portfolio that maximizes return for a given risk tolerance.

        This method identifies the portfolio on the frontier that has the highest
        return, subject to its risk being less than or equal to `max_risk`.

        Args:
            max_risk (float): The maximum allowable risk.

        Returns:
            Tuple[pd.Series, float, float]: A tuple containing:
                -   **weights** (:class:`pandas.Series`): The weights of the portfolio.
                -   **returns** (float): The expected return of the portfolio.
                -   **risk** (float): The risk of the portfolio.
        """
        feasible_indices = np.where(self.risks <= max_risk)[0]
        if feasible_indices.size == 0:
            nan_weights = np.full(self.weights.shape[0], np.nan)
            return self._to_pandas(nan_weights, "Infeasible"), np.nan, np.nan
        
        optimal_idx = feasible_indices[np.argmax(self.returns[feasible_indices])]
        w, ret, risk = self.weights[:, optimal_idx], self.returns[optimal_idx], self.risks[optimal_idx]
        return self._to_pandas(w, f"Portfolio (Risk <= {max_risk:.4f})"), ret, risk

    def portfolio_at_return_target(self, min_return: float) -> Tuple[pd.Series, float, float]:
        """
        Finds the portfolio that minimizes risk for a given expected return target.

        This method identifies the portfolio on the frontier that has the lowest
        risk, subject to its return being greater than or equal to `min_return`.

        Args:
            min_return (float): The minimum required expected return.

        Returns:
            Tuple[pd.Series, float, float]: A tuple containing:
                -   **weights** (:class:`pandas.Series`): The weights of the portfolio.
                -   **returns** (float): The expected return of the portfolio.
                -   **risk** (float): The risk of the portfolio.
        """
        feasible_indices = np.where(self.returns >= min_return)[0]
        if feasible_indices.size == 0:
            nan_weights = np.full(self.weights.shape[0], np.nan)
            return self._to_pandas(nan_weights, "Infeasible"), np.nan, np.nan

        optimal_idx = feasible_indices[np.argmin(self.risks[feasible_indices])]
        w, ret, risk = self.weights[:, optimal_idx], self.returns[optimal_idx], self.risks[optimal_idx]
        return self._to_pandas(w, f"Portfolio (Return >= {min_return:.4f})"), ret, risk


class PortfolioWrapper:
    """
    A high-level interface for portfolio construction and optimization.

    This class serves as the main entry point for performing portfolio
    optimization. It simplifies the process by managing asset data, constraints,
    transaction costs, and the underlying optimization models.

    Typical Workflow:

    1.  Initialize: ``port = PortfolioWrapper(AssetsDistribution(...))``
    2.  Set Constraints: ``port.set_constraints(...)``
    3.  (Optional) Set Costs: ``port.set_transaction_costs(...)``
    4.  Compute: ``frontier = port.mean_variance_frontier()`` or ``portfolio = port.mean_variance_portfolio_at_return(0.10)``
    5.  Analyze: Use the returned :class:`PortfolioFrontier` or portfolio objects.
    """
    def __init__(self, distribution: AssetsDistribution):
        """
        Initializes the PortfolioWrapper with asset distribution data.

        Args:
            distribution (AssetsDistribution): An :class:`AssetsDistribution` object
                containing the statistical properties of the assets.

        Attributes:
            dist (AssetsDistribution): The stored asset distribution.
            G (Optional[np.ndarray]): Matrix for linear inequality constraints (G * w <= h).
            h (Optional[np.ndarray]): Vector for linear inequality constraints (G * w <= h).
            A (Optional[np.ndarray]): Matrix for linear equality constraints (A * w = b).
            b (Optional[np.ndarray]): Vector for linear equality constraints (A * w = b).
            initial_weights (Optional[np.ndarray]): Current portfolio weights, used for
                transaction cost calculations.
            market_impact_costs (Optional[np.ndarray]): Quadratic market impact cost coefficients.
            proportional_costs (Optional[np.ndarray]): Linear proportional transaction cost coefficients.
        """
        self.dist = distribution
        self.G: Optional[np.ndarray] = None
        self.h: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.initial_weights: Optional[np.ndarray] = None
        self.market_impact_costs: Optional[np.ndarray] = None
        self.proportional_costs: Optional[np.ndarray] = None
        logger.info(f"PortfolioWrapper initialized for {self.dist.N} assets.")

    def set_constraints(self, params: Dict[str, Any]):
        """
        Builds and sets linear constraints for the portfolio.

        This method uses the `build_G_h_A_b` utility to construct the constraint
        matrices and vectors based on a dictionary of parameters. These constraints
        are then stored internally and applied during optimization.

        Args:
            params (Dict[str, Any]): A dictionary of constraint parameters.
                Expected keys and their types/meanings include:

                * ``"long_only"`` (bool): If True, enforces non-negative weights (w >= 0).
                * ``"total_weight"`` (float): Sets the sum of weights (sum(w) = value).
                * ``"box_constraints"`` (Tuple[np.ndarray, np.ndarray]): A tuple (lower_bounds, upper_bounds)
                    for individual asset weights.
                * ``"group_constraints"`` (List[Dict[str, Any]]): A list of dictionaries,
                    each defining a group constraint (e.g., min/max weight for a subset of assets).
                * Any other parameters supported by `pyvallocation.utils.constraints.build_G_h_A_b`.

        Raises:
            RuntimeError: If constraint building fails due to invalid parameters or other issues.
        """
        logger.info(f"Setting constraints with parameters: {params}")
        try:
            G, h, A, b = build_G_h_A_b(self.dist.N, **params)
            self.G, self.h = np.atleast_2d(G), np.atleast_1d(h)
            self.A, self.b = np.atleast_2d(A), np.atleast_1d(b)
        except Exception as e:
            logger.error(f"Failed to build constraints: {e}", exc_info=True)
            raise RuntimeError(f"Constraint building failed: {e}") from e

    def set_transaction_costs(
        self,
        initial_weights: Union["pd.Series", npt.NDArray[np.floating]],
        market_impact_costs: Optional[Union["pd.Series", npt.NDArray[np.floating]]] = None,
        proportional_costs: Optional[Union["pd.Series", npt.NDArray[np.floating]]] = None,
    ):
        """
        Sets transaction cost parameters for rebalancing optimizations.

        This method allows specifying initial portfolio weights and associated
        transaction costs (either quadratic market impact or linear proportional costs).
        These costs are incorporated into the optimization problem when applicable.

        Assumptions & Design Choices:
            - If :class:`pandas.Series` are provided for cost parameters, they are
              aligned to the official asset list of the portfolio (`self.dist.asset_names`).
              Assets present in the portfolio but missing from the input Series are
              assumed to have a cost of zero.
            - ``initial_weights`` that do not sum to 1.0 imply a starting position
              that includes cash (if sum < 1) or leverage (if sum > 1).

        Args:
            initial_weights (Union[pd.Series, npt.NDArray[np.floating]]): A 1D array or
                :class:`pandas.Series` of current portfolio weights. This is required
                if any transaction costs are to be applied.
            market_impact_costs (Optional[Union[pd.Series, npt.NDArray[np.floating]]]):
                For Mean-Variance optimization, a 1D array or :class:`pandas.Series` of
                quadratic market impact cost coefficients. Defaults to None.
            proportional_costs (Optional[Union[pd.Series, npt.NDArray[np.floating]]]):
                For Mean-CVaR and Robust optimization, a 1D array or :class:`pandas.Series` of
                linear proportional cost coefficients. Defaults to None.

        Raises:
            ValueError: If the shape of any provided cost parameter array does not match
                        the number of assets (N).
        """
        logger.info("Setting transaction cost parameters.")
        
        def _process_input(data, name):
            """Helper to convert pandas Series to aligned numpy array."""
            if isinstance(data, pd.Series):
                if self.dist.asset_names:
                    original_assets = set(data.index)
                    portfolio_assets = set(self.dist.asset_names)
                    missing_in_input = portfolio_assets - original_assets
                    if missing_in_input:
                        logger.info(f"Input for '{name}' was missing {len(missing_in_input)} asset(s). Assuming their cost/weight is 0.")
                    data = data.reindex(self.dist.asset_names).fillna(0)
                data = data.values
            arr = np.asarray(data, dtype=float)
            if arr.shape != (self.dist.N,):
                raise ValueError(f"`{name}` must have shape ({self.dist.N},), but got {arr.shape}")
            return arr

        self.initial_weights = _process_input(initial_weights, 'initial_weights')
        weight_sum = np.sum(self.initial_weights)
        if not np.isclose(weight_sum, 1.0):
            logger.warning(f"Initial weights sum to {weight_sum:.4f}, not 1.0. This implies a starting cash or leverage position.")
            
        if market_impact_costs is not None:
            self.market_impact_costs = _process_input(market_impact_costs, 'market_impact_costs')
            
        if proportional_costs is not None:
            self.proportional_costs = _process_input(proportional_costs, 'proportional_costs')

    def _ensure_default_constraints(self):
        """Applies default constraints if none were explicitly set."""
        if self.G is None and self.A is None:
            logger.warning("No constraints were set. To ensure a solvable problem, applying default constraints: long-only (weights >= 0) and fully-invested (weights sum to 1.0).")
            self.set_constraints({'long_only': True, 'total_weight': 1.0})

    def mean_variance_frontier(self, num_portfolios: int = 20) -> PortfolioFrontier:
        """Computes the classical Mean-Variance efficient frontier.

        Args:
            num_portfolios: The number of portfolios to compute. Defaults to 20.

        Returns:
            A `PortfolioFrontier` object.
        """
        if self.dist.mu is None or self.dist.cov is None:
            raise ValueError("Mean-Variance optimization requires `mu` and `cov`.")
        self._ensure_default_constraints()
        
        if self.initial_weights is not None and self.market_impact_costs is not None:
            logger.info("Computing Mean-Variance frontier with quadratic transaction costs.")
        
        optimizer = MeanVariance(
            self.dist.mu, self.dist.cov, self.G, self.h, self.A, self.b,
            initial_weights=self.initial_weights,
            market_impact_costs=self.market_impact_costs
        )
        weights = optimizer.efficient_frontier(num_portfolios)
        returns = self.dist.mu @ weights
        risks = np.sqrt(np.sum((weights.T @ self.dist.cov) * weights.T, axis=1))
        
        logger.info(f"Successfully computed Mean-Variance frontier with {weights.shape[1]} portfolios.")
        return PortfolioFrontier(
            weights=weights, returns=returns, risks=risks,
            risk_measure='Volatility', asset_names=self.dist.asset_names
        )
        
    def mean_cvar_frontier(self, num_portfolios: int = 20, alpha: float = 0.05) -> PortfolioFrontier:
        r"""Computes the Mean-CVaR efficient frontier.

        Implementation Notes:
            - This method requires scenarios. If only ``mu`` and ``cov`` are provided,
              it makes a strong modeling assumption to simulate scenarios from a
              multivariate normal distribution.

        Args:
            num_portfolios: The number of portfolios to compute. Defaults to 20.
            alpha: The tail probability for CVaR. Defaults to 0.05.

        Returns:
            A :class:`PortfolioFrontier` object.
        """
        scenarios, probs = self.dist.scenarios, self.dist.probabilities
        if scenarios is None:
            if self.dist.mu is None or self.dist.cov is None:
                raise ValueError("Cannot simulate scenarios for CVaR without `mu` and `cov`.")
            n_sim=5000
            logger.info(f"No scenarios provided. Making a modeling choice to simulate {n_sim} scenarios from a Multivariate Normal distribution using the provided `mu` and `cov`.")
            scenarios = np.random.multivariate_normal(self.dist.mu, self.dist.cov, n_sim)
            probs = generate_uniform_probabilities(n_sim)

        mu_for_frontier = self.dist.mu if self.dist.mu is not None else np.mean(scenarios, axis=0)

        self._ensure_default_constraints()
        if self.initial_weights is not None and self.proportional_costs is not None:
            logger.info("Computing Mean-CVaR frontier with proportional transaction costs.")
            
        optimizer = MeanCVaR(
            R=scenarios, p=probs, alpha=alpha, G=self.G, h=self.h, A=self.A, b=self.b,
            initial_weights=self.initial_weights,
            proportional_costs=self.proportional_costs
        )
        weights = optimizer.efficient_frontier(num_portfolios)
        returns = mu_for_frontier @ weights
        risks = abs(np.array([portfolio_cvar(w, scenarios, probs, alpha) for w in weights.T]))

        logger.info(f"Successfully computed Mean-CVaR frontier with {weights.shape[1]} portfolios.")
        return PortfolioFrontier(
            weights=weights, returns=returns, risks=risks,
            risk_measure=f'CVaR (alpha={alpha:.2f})', asset_names=self.dist.asset_names
        )

    def robust_lambda_frontier(self, num_portfolios: int = 20, max_lambda: float = 2.0) -> PortfolioFrontier:
        r"""Computes a robust frontier based on uncertainty in expected returns.

        Assumptions & Design Choices:
            - This method follows Meucci's robust framework. It assumes that the ``mu``
              and ``cov`` from :class:`AssetsDistribution` represent the posterior mean
              and the posterior scale matrix (for uncertainty), respectively.

        Args:
            num_portfolios: The number of portfolios to compute. Defaults to 20.
            max_lambda: The maximum value for the risk aversion parameter lambda,
              which controls the trade-off between nominal return and robustness.

        Returns:
            A :class:`PortfolioFrontier` object.
        """
        if self.dist.mu is None or self.dist.cov is None:
            raise ValueError("Robust optimization requires `mu` (μ₁) and `cov` (Σ₁).")
        logger.info(
            "Computing robust λ-frontier. Critical Assumption: `dist.mu` is interpreted as the posterior mean and `dist.cov` as the uncertainty covariance matrix."
        )
        self._ensure_default_constraints()
        if self.initial_weights is not None and self.proportional_costs is not None:
            logger.info("Including proportional transaction costs in robust optimization.")
        
        optimizer = RobustOptimizer(
            expected_return=self.dist.mu,
            uncertainty_cov=self.dist.cov,
            G=self.G, h=self.h, A=self.A, b=self.b,
            initial_weights=self.initial_weights,
            proportional_costs=self.proportional_costs
        )
        
        lambdas = np.linspace(0, max_lambda, num_portfolios)
        returns, risks, weights = optimizer.efficient_frontier(lambdas)

        logger.info(f"Successfully computed Robust λ-frontier with {weights.shape[1]} portfolios.")
        return PortfolioFrontier(
            weights=np.array(weights), returns=np.array(returns), risks=np.array(risks),
            risk_measure="Estimation Risk (‖Σ'¹/²w‖₂)", asset_names=self.dist.asset_names
        )

    def mean_variance_portfolio_at_return(self, return_target: float) -> Tuple[pd.Series, float, float]:
        """
        Solves for the minimum variance portfolio that achieves a given expected return.

        This method directly solves the optimization problem for a specific target
        return, rather than interpolating from a pre-computed frontier. This is more
        accurate and efficient if only a single portfolio is of interest.

        Args:
            return_target (float): The desired minimum expected return.

        Returns:
            Tuple[pd.Series, float, float]: A tuple containing:
                - **weights** (:class:`pandas.Series`): The weights of the optimal portfolio.
                - **return** (float): The expected return of the portfolio.
                - **risk** (float): The volatility (standard deviation) of the portfolio.
        
        Raises:
            ValueError: If `mu` and `cov` are not available in the distribution.
        """
        if self.dist.mu is None or self.dist.cov is None:
            raise ValueError("Mean-Variance optimization requires `mu` and `cov`.")
        self._ensure_default_constraints()

        logger.info(f"Solving for minimum variance portfolio with target return >= {return_target:.4f}")
        
        optimizer = MeanVariance(
            self.dist.mu, self.dist.cov, self.G, self.h, self.A, self.b,
            initial_weights=self.initial_weights,
            market_impact_costs=self.market_impact_costs
        )
        
        try:
            # The efficient_portfolio method in the optimizer is an alias for _solve_target
            weights = optimizer.efficient_portfolio(return_target)
        except RuntimeError as e:
            logger.error(f"Optimization failed for target return {return_target}. This may be because the target is infeasible (e.g., too high). Details: {e}", exc_info=True)
            nan_weights = np.full(self.dist.N, np.nan)
            return pd.Series(nan_weights, index=self.dist.asset_names, name="Infeasible"), np.nan, np.nan

        actual_return = self.dist.mu @ weights
        risk = np.sqrt(weights.T @ self.dist.cov @ weights)

        w_series = pd.Series(weights, index=self.dist.asset_names, name=f"MV Portfolio (Return >= {return_target:.4f})")

        logger.info(
            f"Successfully solved for MV portfolio. "
            f"Target Return: {return_target:.4f}, Actual Return: {actual_return:.4f}, Risk: {risk:.4f}"
        )
        return w_series, actual_return, risk

    def mean_cvar_portfolio_at_return(self, return_target: float, alpha: float = 0.05) -> Tuple[pd.Series, float, float]:
        """
        Solves for the minimum CVaR portfolio that achieves a given expected return.

        This method directly solves the optimization problem for a specific target
        return, rather than interpolating from a pre-computed frontier.

        Args:
            return_target (float): The desired minimum expected return.
            alpha (float): The tail probability for CVaR. Defaults to 0.05.

        Returns:
            Tuple[pd.Series, float, float]: A tuple containing:
                - **weights** (:class:`pandas.Series`): The weights of the optimal portfolio.
                - **return** (float): The expected return of the portfolio.
                - **risk** (float): The CVaR of the portfolio.
                
        Raises:
            ValueError: If scenarios cannot be used or generated.
        """
        scenarios, probs = self.dist.scenarios, self.dist.probabilities
        if scenarios is None:
            if self.dist.mu is None or self.dist.cov is None:
                raise ValueError("Cannot simulate scenarios for CVaR without `mu` and `cov`.")
            n_sim = 5000
            logger.info(f"No scenarios provided. Simulating {n_sim} scenarios from a Multivariate Normal distribution.")
            scenarios = np.random.multivariate_normal(self.dist.mu, self.dist.cov, n_sim)
            probs = generate_uniform_probabilities(n_sim)
        
        mu_for_cvar = self.dist.mu if self.dist.mu is not None else np.mean(scenarios, axis=0)
        
        self._ensure_default_constraints()
        
        logger.info(f"Solving for minimum CVaR portfolio with target return >= {return_target:.4f} and alpha = {alpha:.2f}")
        
        optimizer = MeanCVaR(
            R=scenarios, p=probs, alpha=alpha, G=self.G, h=self.h, A=self.A, b=self.b,
            initial_weights=self.initial_weights,
            proportional_costs=self.proportional_costs
        )
        
        try:
            # The efficient_portfolio method in the optimizer is an alias for _solve_target
            weights = optimizer.efficient_portfolio(return_target)
        except RuntimeError as e:
            logger.error(f"Optimization failed for target return {return_target}. This may be because the target is infeasible. Details: {e}", exc_info=True)
            nan_weights = np.full(self.dist.N, np.nan)
            return pd.Series(nan_weights, index=self.dist.asset_names, name="Infeasible"), np.nan, np.nan

        actual_return = mu_for_cvar @ weights
        risk = abs(portfolio_cvar(weights, scenarios, probs, alpha))
        
        w_series = pd.Series(weights, index=self.dist.asset_names, name=f"CVaR Portfolio (Return >= {return_target:.4f})")

        logger.info(
            f"Successfully solved for CVaR portfolio. "
            f"Target Return: {return_target:.4f}, Actual Return: {actual_return:.4f}, Risk (CVaR): {risk:.4f}"
        )
        return w_series, actual_return, risk

    def solve_robust_gamma_portfolio(self, gamma_mu: float, gamma_sigma_sq: float) -> Tuple[pd.Series, float, float]:
        """Solves for a single robust portfolio with explicit uncertainty constraints.

        Args:
            gamma_mu: The penalty for estimation error in the mean.
            gamma_sigma_sq: The squared upper bound for the total portfolio risk.

        Returns:
            A tuple containing the portfolio weights, return, and risk.
        """
        if self.dist.mu is None or self.dist.cov is None:
            raise ValueError("Robust optimization requires `mu` (μ₁) and `cov` (Σ₁).")
        logger.info(
            "Solving robust γ-portfolio. Critical Assumption: `dist.mu` is interpreted as the posterior mean and `dist.cov` as the uncertainty covariance matrix."
        )
        self._ensure_default_constraints()
        if self.initial_weights is not None and self.proportional_costs is not None:
            logger.info("Including proportional transaction costs in robust γ-portfolio optimization.")

        optimizer = RobustOptimizer(
            expected_return=self.dist.mu,
            uncertainty_cov=self.dist.cov,
            G=self.G, h=self.h, A=self.A, b=self.b,
            initial_weights=self.initial_weights,
            proportional_costs=self.proportional_costs
        )

        result = optimizer.solve_gamma_variant(gamma_mu, gamma_sigma_sq)
        
        w_series = pd.Series(result.weights, index=self.dist.asset_names, name="Robust Gamma Portfolio")
            
        logger.info(
            f"Successfully solved for robust γ-portfolio. "
            f"Nominal Return: {result.nominal_return:.4f}, Estimation Risk: {result.risk:.4f}"
        )
        return w_series, result.nominal_return, result.risk
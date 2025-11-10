from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
import copy
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from .discrete_allocation import DiscreteAllocationResult, discretize_weights
from .ensembles import average_exposures
from .moments import estimate_sample_moments
from .optimization import (
    MeanCVaR,
    MeanVariance,
    RelaxedRiskParity,
    RelaxedRiskParityResult,
    RobustOptimizer,
)
from .probabilities import generate_uniform_probabilities
from .utils.constraints import build_G_h_A_b
from .utils.functions import portfolio_cvar
from .utils.weights import wrap_exposure_vector

if TYPE_CHECKING:
    from .ensembles import EnsembleResult, EnsembleSpec

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
        mu, cov = self.mu, self.cov
        scenarios, probs = self.scenarios, self.probabilities
        asset_names = list(self.asset_names) if self.asset_names is not None else None

        def _merge_names(existing: Optional[List[str]], candidate: Sequence[str]) -> Optional[List[str]]:
            candidate_list = list(candidate)
            if not candidate_list:
                return existing
            if existing is None:
                return candidate_list
            if candidate_list != existing:
                raise ValueError("Inconsistent asset names across inputs.")
            return existing

        if isinstance(mu, pd.Series):
            asset_names = _merge_names(asset_names, mu.index)
            mu = mu.to_numpy(dtype=float)
        elif mu is not None:
            mu = np.asarray(mu, dtype=float)

        if isinstance(cov, pd.DataFrame):
            asset_names = _merge_names(asset_names, cov.index)
            if asset_names is not None and list(cov.columns) != asset_names:
                raise ValueError("Covariance matrix columns must match asset names.")
            cov = cov.to_numpy(dtype=float)
        elif cov is not None:
            cov = np.asarray(cov, dtype=float)

        if isinstance(scenarios, pd.DataFrame):
            asset_names = _merge_names(asset_names, scenarios.columns)
            scenarios = scenarios.to_numpy(dtype=float)
        elif scenarios is not None:
            scenarios = np.asarray(scenarios, dtype=float)

        if isinstance(probs, pd.Series):
            probs = probs.to_numpy(dtype=float)
        elif probs is not None:
            probs = np.asarray(probs, dtype=float)

        N: Optional[int] = None
        T: Optional[int] = None

        if scenarios is not None:
            if scenarios.ndim != 2:
                raise ValueError("`scenarios` must be a 2D array with shape (T, N).")
            T, N = scenarios.shape
            if probs is None:
                logger.debug("No probabilities passed with scenarios; assuming uniform weights.")
                probs = generate_uniform_probabilities(T)
            else:
                probs = probs.reshape(-1)
                if probs.shape[0] != T:
                    raise ValueError(
                        f"Probabilities shape mismatch: expected ({T},), got {probs.shape}."
                    )
            if np.any(probs < 0):
                raise ValueError("Scenario probabilities must be non-negative.")
            prob_sum = probs.sum()
            if not np.isfinite(prob_sum) or prob_sum <= 0:
                raise ValueError("Scenario probabilities must sum to a positive finite value.")
            if not np.isclose(prob_sum, 1.0):
                logger.debug("Normalising scenario probabilities (sum=%s).", prob_sum)
            probs = probs / prob_sum

            if mu is None or cov is None:
                estimated_mu, estimated_cov = estimate_sample_moments(scenarios, probs)
                if mu is None:
                    mu = np.asarray(estimated_mu, dtype=float)
                if cov is None:
                    cov = np.asarray(estimated_cov, dtype=float)

        if mu is not None:
            mu = np.asarray(mu, dtype=float).reshape(-1)
            N = mu.size if N is None else N
            if mu.size != N:
                raise ValueError(
                    f"Expected {N} entries in `mu`, received {mu.size}."
                )

        if cov is not None:
            cov = np.asarray(cov, dtype=float)
            if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
                raise ValueError("`cov` must be a square 2D array.")
            N = cov.shape[0] if N is None else N
            if cov.shape != (N, N):
                raise ValueError("`cov` shape must match the number of assets inferred from other inputs.")

        if N is None or N == 0:
            raise ValueError("Insufficient data. Provide either (`mu`, `cov`) or `scenarios`.")

        if asset_names is not None and len(asset_names) != N:
            raise ValueError(
                f"`asset_names` must have length {N}, received {len(asset_names)}."
            )

        object.__setattr__(self, "N", N)
        object.__setattr__(self, "T", T)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "cov", cov)
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "probabilities", None if scenarios is None else probs)
        object.__setattr__(self, "asset_names", asset_names)

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
        risk_measure (str): A string describing the risk measure used to construct this efficient frontier (e.g., 'Volatility', 'CVaR (alpha=0.05)', 'Estimation Risk (||\\Sigma'^1/^2w||_2)').
        asset_names (Optional[List[str]]): An optional list of names for the assets. If provided, enables pandas Series/DataFrame output for portfolio weights.
        metadata (Optional[List[Dict[str, Any]]]): Optional per-portfolio diagnostics. Each entry
            maps diagnostic field names (e.g., ``target_multiplier``) to their values.
    """
    weights: npt.NDArray[np.floating]
    returns: npt.NDArray[np.floating]
    risks: npt.NDArray[np.floating]
    risk_measure: str
    asset_names: Optional[List[str]] = None
    metadata: Optional[List[Dict[str, Any]]] = None

    def _to_pandas(self, w: np.ndarray, name: str) -> pd.Series:
        wrapped = wrap_exposure_vector(w, self.asset_names, label=name)
        if isinstance(wrapped, np.ndarray):
            return pd.Series(wrapped, name=name)
        return wrapped

    def _select_weights(self, columns: Optional[Iterable[int]]) -> np.ndarray:
        if columns is None:
            return self.weights.copy()
        indices = np.array(list(columns), dtype=int)
        return self.weights[:, indices]

    def to_frame(
        self,
        columns: Optional[Iterable[int]] = None,
        *,
        column_labels: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Return the frontier weights as a pandas DataFrame.

        Args:
            columns: Optional iterable of column indices selecting specific portfolios.
            column_labels: Optional labels for the resulting DataFrame columns.

        Returns:
            A DataFrame whose rows correspond to assets and whose columns correspond
            to efficient portfolios.

        Raises:
            ValueError: If ``column_labels`` is supplied with a length that does not
                match the number of selected portfolios.
        """

        selection = list(columns) if columns is not None else None
        matrix = self._select_weights(selection)

        if column_labels is not None:
            labels = list(column_labels)
            if len(labels) != matrix.shape[1]:
                raise ValueError(
                    "`column_labels` length must match the number of selected portfolios."
                )
        elif selection is not None:
            labels = selection
        else:
            labels = list(range(matrix.shape[1]))

        index = self.asset_names if self.asset_names is not None else None
        return pd.DataFrame(matrix, index=index, columns=labels)

    def to_samples(
        self,
        columns: Optional[Iterable[int]] = None,
        *,
        as_frame: bool = True,
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, Optional[List[str]]]]:
        """
        Return the frontier weights as either a DataFrame or raw NumPy samples.

        Parameters
        ----------
        columns
            Optional iterable selecting specific portfolio indices.
        as_frame
            When ``True`` (default) return a pandas DataFrame. When ``False``
            return ``(matrix, asset_names)`` suitable for downstream NumPy
            consumption.
        """
        if as_frame:
            return self.to_frame(columns=columns)
        matrix = self._select_weights(columns)
        names = list(self.asset_names) if self.asset_names is not None else None
        return matrix.copy(), names

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


    def as_discrete_allocation(
        self,
        column: int,
        latest_prices: Union[pd.Series, Mapping[str, float]],
        total_value: float,
        *,
        method: str = "greedy",
        lot_sizes: Optional[Union[pd.Series, Mapping[str, int]]] = None,
        **kwargs,
    ) -> DiscreteAllocationResult:
        """Converts a selected frontier portfolio into a discrete allocation."""

        if column < 0 or column >= self.weights.shape[1]:
            raise IndexError(
                f"Column index {column} is out of bounds for {self.weights.shape[1]} portfolios."
            )

        weights = self.weights[:, column]
        if self.asset_names is not None:
            weight_series = pd.Series(weights, index=self.asset_names)
        else:
            asset_labels = [f"Asset_{i}" for i in range(weights.shape[0])]
            weight_series = pd.Series(weights, index=asset_labels)

        return discretize_weights(
            weights=weight_series,
            latest_prices=latest_prices,
            total_value=total_value,
            method=method,
            lot_sizes=lot_sizes,
            **kwargs,
        )

    def ensemble_average(
        self,
        columns: Optional[Iterable[int]] = None,
        *,
        ensemble_weights: Optional[Sequence[float]] = None,
    ) -> pd.Series:
        matrix = self._select_weights(columns)
        combined = average_exposures(matrix, weights=ensemble_weights)
        return self._to_pandas(combined, "Average Ensemble")


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
            def _matrix_or_none(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
                if value is None:
                    return None
                arr = np.asarray(value, dtype=float)
                if arr.size == 0:
                    return None
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                return arr

            def _vector_or_none(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
                if value is None:
                    return None
                arr = np.asarray(value, dtype=float).reshape(-1)
                return None if arr.size == 0 else arr

            self.G, self.h = _matrix_or_none(G), _vector_or_none(h)
            self.A, self.b = _matrix_or_none(A), _vector_or_none(b)
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
            logger.debug("Injecting default long-only, fully-invested constraints.")
            self.set_constraints({"long_only": True, "total_weight": 1.0})

    def _scenario_inputs(
        self,
        *,
        n_simulations: int = 5000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return scenarios, probabilities, and an expected-return vector."""

        scenarios = self.dist.scenarios
        probs = self.dist.probabilities

        if scenarios is None:
            if self.dist.mu is None or self.dist.cov is None:
                raise ValueError("Cannot simulate scenarios without both `mu` and `cov`.")
            if n_simulations <= 0:
                raise ValueError("`n_simulations` must be a positive integer.")
            logger.info(
                "No scenarios supplied. Simulating %d multivariate normal scenarios for CVaR calculations.",
                n_simulations,
            )
            rng = np.random.default_rng()
            scenarios = rng.multivariate_normal(self.dist.mu, self.dist.cov, n_simulations)
            probs = generate_uniform_probabilities(n_simulations)
        else:
            scenarios = np.asarray(scenarios, dtype=float)
            if probs is None:
                logger.debug("Distribution supplied scenarios without probabilities; defaulting to uniform weights.")
                probs = generate_uniform_probabilities(scenarios.shape[0])
            else:
                probs = np.asarray(probs, dtype=float).reshape(-1)

        prob_sum = probs.sum()
        if np.any(probs < 0) or not np.isfinite(prob_sum) or prob_sum <= 0:
            raise ValueError("Scenario probabilities must be non-negative and sum to a positive finite value.")
        if not np.isclose(prob_sum, 1.0):
            probs = probs / prob_sum

        expected_returns = (
            np.asarray(self.dist.mu, dtype=float)
            if self.dist.mu is not None
            else scenarios.T @ probs
        )

        return scenarios, probs, expected_returns

    def _solve_relaxed_rp(
        self,
        optimizer: RelaxedRiskParity,
        lambda_reg: float,
        requested_target: Optional[float],
        lower_bound: float,
    ) -> Tuple[RelaxedRiskParityResult, Optional[str]]:
        r"""
        Solve the relaxed risk parity problem with defensive target fallback.

        Parameters
        ----------
        optimizer :
            Instance of :class:`pyvallocation.optimization.RelaxedRiskParity` configured
            with the current distribution and constraints.
        lambda_reg :
            Regulator coefficient :math:`\\lambda` applied to the diagonal penalty term.
        requested_target :
            Desired return level :math:`R`. ``None`` signals an unconstrained solve.
        lower_bound :
            Non-negative baseline used for target shrinkage (typically the pure risk
            parity return).

        Returns
        -------
        RelaxedRiskParityResult
            Optimal solution (possibly obtained after clipping the requested target).
        Optional[str]
            Warning message describing why a fallback was required; ``None`` if the
            requested target proved feasible.
        """
        if requested_target is None:
            result = optimizer.solve(lambda_reg=lambda_reg, return_target=None)
            return result, None

        target_candidate = float(requested_target)
        last_error: Optional[Exception] = None
        for _ in range(8):
            try:
                result = optimizer.solve(
                    lambda_reg=lambda_reg,
                    return_target=target_candidate,
                    min_target=lower_bound,
                )
                return result, None
            except RuntimeError as exc:
                last_error = exc
                target_candidate = 0.5 * (target_candidate + lower_bound)
                if target_candidate <= lower_bound + 1e-6:
                    target_candidate = lower_bound

        warning = str(last_error) if last_error is not None else None
        result = optimizer.solve(lambda_reg=lambda_reg, return_target=None)
        return result, warning

    def mean_variance_frontier(self, num_portfolios: int = 10) -> PortfolioFrontier:
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
        
    def mean_cvar_frontier(self, num_portfolios: int = 10, alpha: float = 0.05) -> PortfolioFrontier:
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
        scenarios, probs, mu_for_frontier = self._scenario_inputs()
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
        risks = np.abs(np.asarray(portfolio_cvar(weights, scenarios, probs, alpha))).reshape(-1)

        # Numerical guards: enforce non-decreasing, convex CVaR frontier
        order = np.argsort(returns)
        returns, risks, weights = returns[order], risks[order], weights[:, order]
        # 1) monotone non-decreasing risk w.r.t. return target (feasible set shrinks)
        risks = np.maximum.accumulate(risks)

        # 2) convexify via lower convex envelope in (return, risk)
        def _lower_convex_envelope(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
            idx_stack: list[int] = []
            for i in range(len(x)):
                while len(idx_stack) >= 2:
                    i1, i2 = idx_stack[-2], idx_stack[-1]
                    cross = (x[i2] - x[i1]) * (y[i] - y[i1]) - (y[i2] - y[i1]) * (x[i] - x[i1])
                    if cross <= eps:
                        idx_stack.pop()
                    else:
                        break
                idx_stack.append(i)
            return np.array(idx_stack, dtype=int)

        keep = _lower_convex_envelope(returns, risks)
        if keep.size >= 2 and keep.size < returns.size:
            x_env, y_env = returns[keep], risks[keep]
            # Interpolate envelope risk onto the original grid (keeps matrix shape)
            risks = np.interp(returns, x_env, y_env)

        logger.info(f"Successfully computed Mean-CVaR frontier with {weights.shape[1]} portfolios.")
        return PortfolioFrontier(
            weights=weights, returns=returns, risks=risks,
            risk_measure=f'CVaR (alpha={alpha:.2f})', asset_names=self.dist.asset_names
        )

    def robust_lambda_frontier(
        self,
        num_portfolios: int = 10,
        max_lambda: float = 2.0,
        *,
        lambdas: Optional[Sequence[float]] = None,
    ) -> PortfolioFrontier:
        r"""Computes a robust frontier based on uncertainty in expected returns.

        Assumptions & Design Choices:
            - This method follows Meucci's robust framework. It assumes that the ``mu``
              and ``cov`` from :class:`AssetsDistribution` represent the posterior mean
              and the posterior scale matrix (for uncertainty), respectively.

        Args:
            num_portfolios: The number of portfolios to compute. Defaults to 20.
            max_lambda: The maximum value for the risk aversion parameter lambda,
              which controls the trade-off between nominal return and robustness.
            lambdas: Optional explicit sequence of \lambda values. When provided, it takes
              precedence over ``num_portfolios``/``max_lambda``.

        Returns:
            A :class:`PortfolioFrontier` object.
        """
        if self.dist.mu is None or self.dist.cov is None:
            raise ValueError(r"Robust optimization requires `mu` (\mu_1) and `cov` (\Sigma_1).")
        logger.info(
            r"Computing robust \lambda-frontier. Critical Assumption: `dist.mu` is interpreted as the posterior mean and `dist.cov` as the uncertainty covariance matrix."
        )
        self._ensure_default_constraints()
        if self.initial_weights is not None and self.proportional_costs is not None:
            logger.info("Including proportional transaction costs in robust optimization.")

        if lambdas is not None:
            lambda_grid = np.asarray(list(lambdas), dtype=float).reshape(-1)
            if lambda_grid.size == 0:
                raise ValueError("`lambdas` must contain at least one value.")
            if np.any(lambda_grid < 0):
                raise ValueError("`lambdas` must be non-negative.")
        else:
            if num_portfolios <= 0:
                raise ValueError("`num_portfolios` must be positive.")
            if max_lambda < 0:
                raise ValueError("`max_lambda` must be non-negative.")
            if num_portfolios == 1:
                lambda_grid = np.array([float(max_lambda)])
            else:
                if max_lambda == 0:
                    lambda_grid = np.zeros(1)
                else:
                    lambda_grid = np.linspace(0, max_lambda, num_portfolios)

        optimizer = RobustOptimizer(
            expected_return=self.dist.mu,
            uncertainty_cov=self.dist.cov,
            G=self.G, h=self.h, A=self.A, b=self.b,
            initial_weights=self.initial_weights,
            proportional_costs=self.proportional_costs
        )
        lambda_list = lambda_grid.tolist()
        returns, risks, weights = optimizer.efficient_frontier(lambda_list)

        logger.info(f"Successfully computed Robust \\lambda-frontier with {weights.shape[1]} portfolios.")
        return PortfolioFrontier(
            weights=np.asarray(weights, dtype=float),
            returns=np.asarray(returns, dtype=float),
            risks=np.asarray(risks, dtype=float),
            risk_measure="Estimation Risk (||\\Sigma'^1/^2w||_2)", asset_names=self.dist.asset_names
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
        scenarios, probs, mu_for_cvar = self._scenario_inputs()
        
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
        risk = float(np.abs(portfolio_cvar(weights, scenarios, probs, alpha)))
        
        w_series = pd.Series(weights, index=self.dist.asset_names, name=f"CVaR Portfolio (Return >= {return_target:.4f})")

        logger.info(
            f"Successfully solved for CVaR portfolio. "
            f"Target Return: {return_target:.4f}, Actual Return: {actual_return:.4f}, Risk (CVaR): {risk:.4f}"
        )
        return w_series, actual_return, risk

    def relaxed_risk_parity_portfolio(
        self,
        *,
        lambda_reg: float = 0.2,
        target_multiplier: Optional[float] = 1.2,
        return_target: Optional[float] = None,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        r"""
        Compute a single relaxed risk parity allocation and expose solver diagnostics.

        The routine first solves the baseline risk parity programme (\lambda = 0) to obtain
        the benchmark return :math:`r_{RP} = \mu^{\top}x^{RP}`. Unless an explicit
        ``return_target`` is supplied, the relaxed model is then solved with the adaptive
        target :math:`R = m \cdot \max(r_{RP}, 0)` where ``m`` is ``target_multiplier``.
        Infeasible targets are clipped via a backtracking line-search toward the RP
        return before falling back to the unconstrained problem if necessary.

        Parameters
        ----------
        lambda_reg :
            Non-negative regulator coefficient :math:`\lambda`. Setting ``0`` recovers
            the pure risk parity allocation.
        target_multiplier :
            Optional multiplier :math:`m` governing the adaptive target-return rule.
            Ignored whenever ``return_target`` is provided. Must be ``None`` when
            pairing with ``lambda_reg == 0`` to avoid redundant relaxation.
        return_target :
            Explicit target return :math:`R`. When supplied, overrides the adaptive
            rule. The method clips :math:`R` down to the feasible region if necessary.

        Returns
        -------
        pandas.Series
            Optimal portfolio weights indexed by asset names (when available).
        Dict[str, Any]
            Rich diagnostics including achieved return, variance, marginal risks,
            risk contributions, target information, and any solver warning emitted
            during target clipping.
        """
        if self.dist.mu is None or self.dist.cov is None:
            raise ValueError("Relaxed risk parity requires `mu` and `cov`.")

        self._ensure_default_constraints()
        logger.info(
            "Solving relaxed risk parity portfolio (lambda=%s, return_target=%s, target_multiplier=%s).",
            lambda_reg,
            return_target,
            target_multiplier,
        )

        optimizer = RelaxedRiskParity(
            mean=self.dist.mu,
            covariance=self.dist.cov,
            G=self.G,
            h=self.h,
            A=self.A,
            b=self.b,
        )

        rp_solution = optimizer.solve(lambda_reg=0.0, return_target=None)
        rp_weights = np.asarray(rp_solution.weights, dtype=float)
        rp_return = float(self.dist.mu @ rp_weights)
        requested_target: Optional[float] = None
        solver_warning: Optional[str] = None
        lower_bound = max(rp_return, 0.0)

        if lambda_reg == 0.0 and return_target is None and target_multiplier is None:
            solution = rp_solution
        else:
            if return_target is not None:
                requested_target = float(return_target)
            else:
                if target_multiplier is None:
                    raise ValueError("Provide `target_multiplier` or `return_target` when lambda_reg > 0.")
                if target_multiplier < 0:
                    raise ValueError("`target_multiplier` must be non-negative.")
                requested_target = float(target_multiplier * lower_bound)

            solution, solver_warning = self._solve_relaxed_rp(
                optimizer, lambda_reg, requested_target, lower_bound
            )
            if solver_warning is not None:
                logger.warning(
                    "Relaxed risk parity target %s infeasible; reverting to unconstrained solve. Details: %s",
                    requested_target,
                    solver_warning,
                )

        weights = np.asarray(solution.weights, dtype=float)
        asset_names = self.dist.asset_names
        name = "Relaxed Risk Parity" if solution.target_return is None else f"Relaxed Risk Parity (\\lambda={lambda_reg:.3f})"
        w_series = pd.Series(weights, index=asset_names, name=name)

        achieved_return = float(self.dist.mu @ weights)
        portfolio_variance = float(weights @ (self.dist.cov @ weights))
        risk_contributions = weights * np.asarray(solution.marginal_risk, dtype=float)

        target_clipped = (
            solution.target_return is not None
            and requested_target is not None
            and not np.isclose(solution.target_return, requested_target, rtol=1e-8, atol=1e-10)
        )

        diagnostics: Dict[str, Any] = {
            "lambda_reg": float(lambda_reg),
            "requested_target": requested_target,
            "target_return": solution.target_return,
            "target_clipped": target_clipped,
            "max_feasible_return": solution.max_return,
            "achieved_return": achieved_return,
            "risk_parity_return": rp_return,
            "portfolio_variance": portfolio_variance,
            "psi": solution.psi,
            "gamma": solution.gamma,
            "rho": solution.rho,
            "objective": solution.objective,
            "risk_contributions": risk_contributions,
            "marginal_risk": np.asarray(solution.marginal_risk, dtype=float),
            "risk_parity_weights": rp_weights,
            "solver_warning": solver_warning,
        }

        logger.info(
            "Relaxed risk parity solved. TargetUsed=%s, Achieved=%s, Variance=%s.",
            solution.target_return,
            achieved_return,
            portfolio_variance,
        )

        return w_series, diagnostics

    def relaxed_risk_parity_frontier(
        self,
        num_portfolios: int = 10,
        max_multiplier: float = 1.6,
        *,
        lambda_reg: float = 0.2,
        target_multipliers: Optional[Sequence[float]] = None,
        include_risk_parity: bool = True,
    ) -> PortfolioFrontier:
        r"""
        Build a relaxed risk parity frontier by sweeping target-return multipliers.

        Each frontier column corresponds to a distinct multiplier :math:`m` applied to
        the benchmark risk parity return :math:`r_{RP}` to generate the target
        :math:`R = m \cdot \max(r_{RP}, 0)`. The method solves the regulated RP
        programme for each multiplier using the shared :math:`\lambda` value and stores
        per-point diagnostics (effective target after clipping, objective value, cone
        slack variables, solver warnings) in ``PortfolioFrontier.metadata``.

        Parameters
        ----------
        num_portfolios :
            Number of grid points when ``target_multipliers`` is omitted. Must be
            positive; includes the upper endpoint ``max_multiplier``.
        max_multiplier :
            Upper bound for the automatically generated multiplier grid. Ignored if
            ``target_multipliers`` is supplied.
        lambda_reg :
            Regulator coefficient :math:`\lambda`. Applies to every relaxed point; the
            optional RP anchor always uses :math:`\lambda = 0`.
        target_multipliers :
            Explicit iterable of multipliers. When provided the method skips automatic
            grid generation and uses the supplied values verbatim.
        include_risk_parity :
            If ``True`` (default) the frontier prepends the pure risk parity solution so
            downstream plots can intercept the anchor directly.

        Returns
        -------
        PortfolioFrontier
            Object containing weights ``(n, k)``, realised returns, volatility proxy
            (standard deviation), and diagnostic metadata for each node on the sweep.
        """
        if self.dist.mu is None or self.dist.cov is None:
            raise ValueError("Relaxed risk parity frontier requires `mu` and `cov`.")
        if lambda_reg < 0:
            raise ValueError("`lambda_reg` must be non-negative.")
        self._ensure_default_constraints()

        optimizer = RelaxedRiskParity(
            mean=self.dist.mu,
            covariance=self.dist.cov,
            G=self.G,
            h=self.h,
            A=self.A,
            b=self.b,
        )

        rp_solution = optimizer.solve(lambda_reg=0.0, return_target=None)
        rp_weights = np.asarray(rp_solution.weights, dtype=float)
        rp_return = float(self.dist.mu @ rp_weights)
        rp_variance = float(rp_weights @ (self.dist.cov @ rp_weights))
        lower_bound = max(rp_return, 0.0)

        if target_multipliers is not None:
            multipliers = np.asarray(list(target_multipliers), dtype=float).reshape(-1)
            if multipliers.size == 0:
                raise ValueError("`target_multipliers` must contain at least one entry.")
            if np.any(multipliers < 0):
                raise ValueError("`target_multipliers` must be non-negative.")
        else:
            if num_portfolios <= 0:
                raise ValueError("`num_portfolios` must be positive.")
            if max_multiplier < 1.0:
                raise ValueError("`max_multiplier` must be at least 1.0.")
            if num_portfolios == 1:
                multipliers = np.array([max(1.0, max_multiplier)], dtype=float)
            else:
                multipliers = np.linspace(1.0, max_multiplier, num_portfolios)

        weights_list: list[np.ndarray] = []
        returns_list: list[float] = []
        risks_list: list[float] = []
        metadata: list[Dict[str, Any]] = []

        if include_risk_parity:
            weights_list.append(rp_weights)
            returns_list.append(rp_return)
            risks_list.append(np.sqrt(max(rp_variance, 0.0)))
            metadata.append(
                {
                    "lambda_reg": 0.0,
                    "target_multiplier": None,
                    "requested_target": None,
                    "effective_target": rp_solution.target_return,
                    "objective": rp_solution.objective,
                    "psi": rp_solution.psi,
                    "gamma": rp_solution.gamma,
                    "rho": rp_solution.rho,
                    "solver_warning": None,
                }
            )

        for multiplier in multipliers:
            requested_target = float(multiplier * lower_bound)
            solution, warning = self._solve_relaxed_rp(
                optimizer, lambda_reg, requested_target, lower_bound
            )
            weights = np.asarray(solution.weights, dtype=float)
            returns = float(self.dist.mu @ weights)
            variance = float(weights @ (self.dist.cov @ weights))

            weights_list.append(weights)
            returns_list.append(returns)
            risks_list.append(np.sqrt(max(variance, 0.0)))
            metadata.append(
                {
                    "lambda_reg": float(lambda_reg),
                    "target_multiplier": float(multiplier),
                    "requested_target": requested_target,
                    "effective_target": solution.target_return,
                    "objective": solution.objective,
                    "psi": solution.psi,
                    "gamma": solution.gamma,
                    "rho": solution.rho,
                    "solver_warning": warning,
                }
            )

        weight_matrix = np.column_stack(weights_list)
        returns_array = np.array(returns_list, dtype=float)
        risks_array = np.array(risks_list, dtype=float)

        logger.info(
            r"Computed relaxed risk parity frontier with %d portfolios (\lambda=%s).",
            weight_matrix.shape[1],
            lambda_reg,
        )
        return PortfolioFrontier(
            weights=weight_matrix,
            returns=returns_array,
            risks=risks_array,
            risk_measure="Volatility (Relaxed RP)",
            asset_names=self.dist.asset_names,
            metadata=metadata,
        )

    def solve_robust_gamma_portfolio(self, gamma_mu: float, gamma_sigma_sq: float) -> Tuple[pd.Series, float, float]:
        """Solves for a single robust portfolio with explicit uncertainty constraints.

        Args:
            gamma_mu: The penalty for estimation error in the mean.
            gamma_sigma_sq: The squared upper bound for the total portfolio risk.

        Returns:
            A tuple containing the portfolio weights, return, and risk.
        """
        if self.dist.mu is None or self.dist.cov is None:
            raise ValueError(r"Robust optimization requires `mu` (\mu_1) and `cov` (\Sigma_1).")
        logger.info(
            r"Solving robust \gamma-portfolio. Critical Assumption: `dist.mu` is interpreted as the posterior mean and `dist.cov` as the uncertainty covariance matrix."
        )
        self._ensure_default_constraints()
        if self.initial_weights is not None and self.proportional_costs is not None:
            logger.info(r"Including proportional transaction costs in robust \gamma-portfolio optimization.")

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
            f"Successfully solved for robust \\gamma-portfolio. "
            f"Nominal Return: {result.nominal_return:.4f}, Estimation Risk: {result.risk:.4f}"
        )
        return w_series, result.nominal_return, result.risk

    def make_ensemble_spec(
        self,
        name: str,
        *,
        optimiser: Union[
            str,
            Callable[["PortfolioWrapper"], "PortfolioFrontier"],
            Callable[..., "PortfolioFrontier"],
        ] = "mean_variance",
        optimiser_kwargs: Optional[Dict[str, Any]] = None,
        selector: Union[
            str,
            Callable[["PortfolioFrontier"], Union[pd.Series, Tuple[Any, ...], np.ndarray]],
        ] = "tangency",
        selector_kwargs: Optional[Dict[str, Any]] = None,
        frontier_selection: Optional[Sequence[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "EnsembleSpec":
        """
        Convenience wrapper around :func:`pyvallocation.ensembles.make_portfolio_spec`
        that reuses the wrapper's distribution.
        """
        from .ensembles import make_portfolio_spec

        use_scenarios = self.dist.scenarios is not None

        def _distribution_factory() -> AssetsDistribution:
            return copy.deepcopy(self.dist)

        return make_portfolio_spec(
            name=name,
            distribution_factory=_distribution_factory,
            use_scenarios=use_scenarios,
            optimiser=optimiser,
            optimiser_kwargs=optimiser_kwargs,
            selector=selector,
            selector_kwargs=selector_kwargs,
            frontier_selection=frontier_selection,
            metadata=metadata,
        )

    def assemble_ensembles(
        self,
        specs: Sequence["EnsembleSpec"],
        **kwargs: Any,
    ) -> "EnsembleResult":
        """
        Proxy to :func:`pyvallocation.ensembles.assemble_portfolio_ensemble`.
        """
        from .ensembles import assemble_portfolio_ensemble

        return assemble_portfolio_ensemble(specs, **kwargs)

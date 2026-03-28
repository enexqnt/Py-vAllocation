from __future__ import annotations

import copy
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from .discrete_allocation import DiscreteAllocationResult, discretize_weights
from .ensembles import average_exposures
from .moments import estimate_sample_moments
from .optimization import (
    InfeasibleOptimizationError,
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
            """Merge asset names ensuring consistent ordering.

            Args:
                existing: Existing asset name list (or ``None``).
                candidate: Candidate asset names.

            Returns:
                Optional[List[str]]: Consolidated asset names.
            """
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
                logger.warning("Normalising scenario probabilities (sum=%s).", prob_sum)
            probs = probs / prob_sum

            if mu is None or cov is None:
                logger.warning(
                    "Estimating %s from scenarios (no explicit values provided).",
                    "mu and cov" if mu is None and cov is None
                    else ("mu" if mu is None else "cov"),
                )
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
        risks (npt.NDArray[np.floating]): Primary risk vector of length ``M`` (e.g., volatility, CVaR, estimation risk).
        risk_measure (str): Label describing the primary risk measure.
        asset_names (Optional[List[str]]): An optional list of names for the assets. If provided, enables pandas Series/DataFrame output for portfolio weights.
        metadata (Optional[List[Dict[str, Any]]]): Optional per-portfolio diagnostics. Each entry
            maps diagnostic field names (e.g., ``target_multiplier``) to their values.
        alternate_risks (Dict[str, np.ndarray]): Optional auxiliary risk grids keyed by label
            (e.g., ``"Volatility"`` when the frontier is built on CVaR). Shapes must
            match ``(M,)``. Access via ``risk_label=...`` parameters.
    """
    weights: npt.NDArray[np.floating]
    returns: npt.NDArray[np.floating]
    risks: npt.NDArray[np.floating]
    risk_measure: str
    asset_names: Optional[List[str]] = None
    metadata: Optional[List[Dict[str, Any]]] = None
    alternate_risks: Dict[str, np.ndarray] = field(default_factory=dict)

    def available_risk_measures(self) -> List[str]:
        """Return list of risk measure labels carried by this frontier.

        Returns:
            list[str]: Primary and alternate risk labels.
        """
        return [self.risk_measure] + list(self.alternate_risks.keys())

    def _risk_vector(self, risk_label: Optional[str]) -> np.ndarray:
        """Select the risk vector corresponding to the requested risk label.

        Args:
            risk_label: Risk label to select. ``None`` selects the primary risk.

        Returns:
            np.ndarray: Risk vector aligned to the frontier columns.
        """
        if risk_label is None or risk_label == self.risk_measure:
            return self.risks
        try:
            return self.alternate_risks[risk_label]
        except KeyError as exc:  # pragma: no cover - defensive guard
            available = ", ".join(self.available_risk_measures())
            raise KeyError(
                f"Risk measure '{risk_label}' not found. Available: {available}"
            ) from exc

    def _to_pandas(self, w: np.ndarray, name: str) -> pd.Series:
        """Wrap raw weights into a pandas Series when asset labels are available.

        Args:
            w: Weight vector.
            name: Series name.

        Returns:
            pd.Series: Weight vector wrapped as a Series.
        """
        wrapped = wrap_exposure_vector(w, self.asset_names, label=name)
        if isinstance(wrapped, np.ndarray):
            return pd.Series(wrapped, name=name)
        return wrapped

    def _select_weights(self, columns: Optional[Iterable[int]]) -> np.ndarray:
        """Return weight columns as a dense NumPy matrix.

        Args:
            columns: Optional column indices to select.

        Returns:
            np.ndarray: Weight matrix.
        """
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

        Args:
            columns: Optional iterable selecting specific portfolio indices.
            as_frame: When ``True`` (default) return a pandas DataFrame. When ``False``
                return ``(matrix, asset_names)`` suitable for downstream NumPy consumption.
        """
        if as_frame:
            return self.to_frame(columns=columns)
        matrix = self._select_weights(columns)
        names = list(self.asset_names) if self.asset_names is not None else None
        return matrix.copy(), names

    def get_min_risk_portfolio(
        self,
        *,
        risk_label: Optional[str] = None,
    ) -> Tuple[pd.Series, float, float]:
        """
        Finds the portfolio with the minimum risk on the efficient frontier.

        Args:
            risk_label: Optional alternate risk measure name (see
                :meth:`available_risk_measures`). Defaults to the primary risk.

        Returns:
            Tuple[pd.Series, float, float]: A tuple containing:
                -   **weights** (:class:`pandas.Series`): The weights of the minimum risk portfolio.
                -   **returns** (float): The expected return of the minimum risk portfolio.
                -   **risk** (float): The risk of the minimum risk portfolio.
        """
        risks = self._risk_vector(risk_label)
        min_risk_idx = np.argmin(risks)
        w = self.weights[:, min_risk_idx]
        ret, risk = self.returns[min_risk_idx], risks[min_risk_idx]
        label = "Min Risk Portfolio" if risk_label is None else f"Min {risk_label}"
        return self._to_pandas(w, label), ret, risk

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
            raise InfeasibleOptimizationError(
                "All portfolios on the frontier have zero risk. Sharpe ratio is undefined."
            )

        with np.errstate(divide='ignore', invalid='ignore'):
            sharpe_ratios = (self.returns - risk_free_rate) / self.risks
        sharpe_ratios[~np.isfinite(sharpe_ratios)] = -np.inf

        tangency_idx = np.argmax(sharpe_ratios)
        w, ret, risk = self.weights[:, tangency_idx], self.returns[tangency_idx], self.risks[tangency_idx]
        return self._to_pandas(w, f"Tangency Portfolio (rf={risk_free_rate:.2%})"), ret, risk

    def portfolio_at_risk_target(
        self,
        max_risk: float,
        *,
        risk_label: Optional[str] = None,
    ) -> Tuple[pd.Series, float, float]:
        """
        Finds the portfolio that maximizes return for a given risk tolerance.

        This method identifies the portfolio on the frontier that has the highest
        return, subject to its risk being less than or equal to `max_risk`.

        Args:
            max_risk (float): The maximum allowable risk.
            risk_label: Optional alternate risk measure name (see
                :meth:`available_risk_measures`). Defaults to the primary risk.

        Returns:
            Tuple[pd.Series, float, float]: A tuple containing:
                -   **weights** (:class:`pandas.Series`): The weights of the portfolio.
                -   **returns** (float): The expected return of the portfolio.
                -   **risk** (float): The risk of the portfolio.
        """
        risks = self._risk_vector(risk_label)
        feasible_indices = np.where(risks <= max_risk)[0]
        if feasible_indices.size == 0:
            raise InfeasibleOptimizationError(
                "No valid portfolio found matching the criteria."
            )
        
        optimal_idx = feasible_indices[np.argmax(self.returns[feasible_indices])]
        w, ret, risk_value = (
            self.weights[:, optimal_idx],
            self.returns[optimal_idx],
            risks[optimal_idx],
        )
        label = f"Portfolio ({risk_label or self.risk_measure} <= {max_risk:.4f})"
        return self._to_pandas(w, label), ret, risk_value

    def max_return_at_risk(
        self,
        max_risk: float,
        *,
        risk_label: Optional[str] = None,
    ) -> Tuple[pd.Series, float, float]:
        """Alias for :meth:`portfolio_at_risk_target` for clearer naming.

        Args:
            max_risk: Risk upper bound.
            risk_label: Optional risk label override.

        Returns:
            Tuple[pd.Series, float, float]: Weights, return, and risk value.
        """
        return self.portfolio_at_risk_target(max_risk, risk_label=risk_label)

    def portfolio_at_return_target(
        self,
        min_return: float,
        *,
        risk_label: Optional[str] = None,
    ) -> Tuple[pd.Series, float, float]:
        """
        Finds the portfolio that minimizes risk for a given expected return target.

        This method identifies the portfolio on the frontier that has the lowest
        risk, subject to its return being greater than or equal to `min_return`.

        Args:
            min_return (float): The minimum required expected return.
            risk_label: Optional alternate risk measure used for the minimisation.

        Returns:
            Tuple[pd.Series, float, float]: A tuple containing:
                -   **weights** (:class:`pandas.Series`): The weights of the portfolio.
                -   **returns** (float): The expected return of the portfolio.
                -   **risk** (float): The risk of the portfolio.
        """
        feasible_indices = np.where(self.returns >= min_return)[0]
        if feasible_indices.size == 0:
            raise InfeasibleOptimizationError(
                "No valid portfolio found matching the criteria."
            )

        risks = self._risk_vector(risk_label)
        optimal_idx = feasible_indices[np.argmin(risks[feasible_indices])]
        w, ret, risk_value = (
            self.weights[:, optimal_idx],
            self.returns[optimal_idx],
            risks[optimal_idx],
        )
        label = f"Portfolio (Return >= {min_return:.4f})"
        if risk_label is not None and risk_label != self.risk_measure:
            label += f" | Risk: {risk_label}"
        return self._to_pandas(w, label), ret, risk_value

    def min_risk_at_return(
        self,
        min_return: float,
        *,
        risk_label: Optional[str] = None,
    ) -> Tuple[pd.Series, float, float]:
        """Alias for :meth:`portfolio_at_return_target`.

        Args:
            min_return: Minimum expected return.
            risk_label: Optional risk label override.

        Returns:
            Tuple[pd.Series, float, float]: Weights, return, and risk value.
        """
        return self.portfolio_at_return_target(min_return, risk_label=risk_label)

    def portfolio_closest_risk(
        self,
        target_risk: float,
        *,
        risk_label: Optional[str] = None,
    ) -> Tuple[pd.Series, float, float]:
        """
        Select the portfolio whose risk is nearest ``target_risk`` (L1 distance).

        Useful for aligning risk levels across frontiers built with different
        models.
        """
        risks = self._risk_vector(risk_label)
        idx = int(np.argmin(np.abs(risks - target_risk)))
        w = self.weights[:, idx]
        ret, risk_value = self.returns[idx], risks[idx]
        label = f"Closest to Risk={target_risk:.4f}" if risk_label is None else f"{risk_label}≈{target_risk:.4f}"
        return self._to_pandas(w, label), ret, risk_value

    def risk_percentiles(self, risk_label: Optional[str] = None) -> np.ndarray:
        """Return per-portfolio risk percentiles on ``[0, 1]``.

        Args:
            risk_label: Risk label to use for ranking. Defaults to primary risk.

        Returns:
            np.ndarray: Percentile ranks in ``[0, 1]``.
        """
        risks = np.asarray(self._risk_vector(risk_label), dtype=float)
        if risks.size == 1:
            return np.array([0.0])
        order = np.argsort(risks)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(risks.size, dtype=float)
        return ranks / (risks.size - 1)

    @staticmethod
    def _normalize_percentile(percentile: float) -> float:
        """Normalize percentile input to the [0, 1] interval.

        Args:
            percentile: Percentile in ``[0, 1]`` or ``[0, 100]``.

        Returns:
            float: Normalized percentile in ``[0, 1]``.
        """
        pct = float(percentile)
        if pct > 1.0:
            if pct > 100.0:
                raise ValueError("`percentile` must be within [0, 1] or [0, 100].")
            pct /= 100.0
        if pct < 0.0 or pct > 1.0:
            raise ValueError("`percentile` must be within [0, 1] or [0, 100].")
        return pct

    def index_at_risk_percentile(
        self,
        percentile: float,
        *,
        risk_label: Optional[str] = None,
    ) -> int:
        """Return the column index closest to a risk percentile.

        Args:
            percentile: Percentile in ``[0, 1]`` or ``[0, 100]``.
            risk_label: Risk label to use for ranking.

        Returns:
            int: Column index on the frontier.
        """
        pct = self._normalize_percentile(percentile)
        percentiles = self.risk_percentiles(risk_label=risk_label)
        return int(np.argmin(np.abs(percentiles - pct)))

    def portfolio_at_risk_percentile(
        self,
        percentile: float,
        *,
        risk_label: Optional[str] = None,
    ) -> Tuple[pd.Series, float, float]:
        """
        Select the portfolio closest to a risk percentile (0-1 or 0-100).

        Args:
            percentile: Percentile in ``[0, 1]`` or ``[0, 100]``.
            risk_label: Risk label to use for ranking.

        Returns:
            Tuple[pd.Series, float, float]: Weights, return, and risk value.
        """
        pct = self._normalize_percentile(percentile)
        idx = self.index_at_risk_percentile(pct, risk_label=risk_label)
        w = self.weights[:, idx]
        ret, risk_value = self.returns[idx], self._risk_vector(risk_label)[idx]
        label = f"Risk Percentile {pct:.0%}"
        return self._to_pandas(w, label), ret, risk_value


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
        """Convert a selected frontier portfolio into a discrete allocation.

        Args:
            column: Frontier column index to discretize.
            latest_prices: Series or mapping of asset prices.
            total_value: Total portfolio value available.
            method: Discrete allocation method (default ``"greedy"``).
            lot_sizes: Optional lot sizes per asset.
            **kwargs: Extra arguments forwarded to the allocator.

        Returns:
            DiscreteAllocationResult: Allocation result with share counts.
        """

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
        """Return the average ensemble across selected frontier columns.

        Args:
            columns: Optional column indices to include.
            ensemble_weights: Optional weights for averaging.

        Returns:
            pd.Series: Averaged portfolio weights.
        """
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
    4.  Compute: ``frontier = port.variance_frontier()`` or ``portfolio = port.min_variance_at_return(0.10)``
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
                * ``"bounds"`` (Tuple or Sequence of Tuples): Per-asset ``(lower, upper)``
                    weight bounds.
                * ``"relative_bounds"`` (Sequence of Tuples): Triples ``(i, j, bound)``
                    implementing ``w_i - w_j <= bound``.
                * ``"additional_G_h"`` (Sequence of Tuples): Extra inequality rows ``(row, rhs)``.
                * ``"additional_A_b"`` (Sequence of Tuples): Extra equality rows ``(row, rhs)``.
                * See :func:`pyvallocation.utils.constraints.build_G_h_A_b` for full details.

        Raises:
            RuntimeError: If constraint building fails due to invalid parameters or other issues.
        """
        logger.info(f"Setting constraints with parameters: {params}")
        try:
            G, h, A, b = build_G_h_A_b(self.dist.N, **params)
            def _matrix_or_none(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
                """Normalize matrix inputs, returning ``None`` for empties.

                Args:
                    value: Input array or ``None``.

                Returns:
                    Optional[np.ndarray]: Normalized 2D array or ``None``.
                """
                if value is None:
                    return None
                arr = np.asarray(value, dtype=float)
                if arr.size == 0:
                    return None
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                return arr

            def _vector_or_none(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
                """Normalize vector inputs, returning ``None`` for empties.

                Args:
                    value: Input array or ``None``.

                Returns:
                    Optional[np.ndarray]: Normalized 1D array or ``None``.
                """
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
            """Helper to convert pandas Series to aligned numpy array.

            Args:
                data: Series or array-like input.
                name: Parameter name for error messages.

            Returns:
                np.ndarray: Aligned vector of length ``N``.
            """
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
        """Applies default constraints if none were explicitly set.

        Defaults to long-only, fully-invested constraints.
        """
        if self.G is None and self.A is None:
            logger.warning("Injecting default long-only, fully-invested constraints.")
            self.set_constraints({"long_only": True, "total_weight": 1.0})

    def _scenario_inputs(
        self,
        *,
        n_simulations: int = 5000,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return scenarios, probabilities, and an expected-return vector.

        Args:
            n_simulations: Number of simulated scenarios when sampling is required.
            seed: Random seed for reproducibility. If ``None``, a warning is
                logged because the simulation will not be reproducible.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Scenarios, probabilities, and mean vector.
        """

        scenarios = self.dist.scenarios
        probs = self.dist.probabilities

        if scenarios is None:
            if self.dist.mu is None or self.dist.cov is None:
                raise ValueError("Cannot simulate scenarios without both `mu` and `cov`.")
            if n_simulations <= 0:
                raise ValueError("`n_simulations` must be a positive integer.")
            if seed is None:
                logger.warning("Scenario simulation is non-reproducible (no seed). Pass seed= for reproducibility.")
            logger.info(
                "No scenarios supplied. Simulating %d multivariate normal scenarios for CVaR calculations.",
                n_simulations,
            )
            rng = np.random.default_rng(seed)
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

        Args:
            optimizer: Instance of :class:`pyvallocation.optimization.RelaxedRiskParity`
                configured with the current distribution and constraints.
            lambda_reg: Regulator coefficient :math:`\\lambda` applied to the diagonal
                penalty term.
            requested_target: Desired return level :math:`R`. ``None`` signals an
                unconstrained solve.
            lower_bound: Non-negative baseline used for target shrinkage (typically
                the pure risk parity return).

        Returns:
            Tuple[RelaxedRiskParityResult, Optional[str]]: Optimal solution (possibly
            after clipping the requested target) and an optional warning message
            describing why a fallback was required.
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
        """Deprecated. Use :meth:`variance_frontier`.

        Args:
            num_portfolios: Number of portfolios to compute.

        Returns:
            PortfolioFrontier: Mean-variance frontier.
        """
        warnings.warn(
            "`mean_variance_frontier` is deprecated; use `variance_frontier`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.variance_frontier(num_portfolios=num_portfolios)

    def variance_frontier(self, num_portfolios: int = 10) -> PortfolioFrontier:
        """Compute the classical mean-variance efficient frontier.

        Args:
            num_portfolios: The number of portfolios to compute. Defaults to 10.

        Returns:
            A `PortfolioFrontier` object. When scenarios are available on the
            distribution, a CVaR overlay is added to ``alternate_risks`` to
            allow CVaR-based selection on the same weights.
        """
        if self.dist.mu is None or self.dist.cov is None:
            raise ValueError("Mean-Variance optimization requires `mu` and `cov`.")
        if self.proportional_costs is not None:
            logger.warning("proportional_costs are ignored by mean-variance; use market_impact_costs instead.")
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

        alternate_risks: Dict[str, np.ndarray] = {}
        if self.dist.scenarios is not None:
            try:
                scen = np.asarray(self.dist.scenarios, dtype=float)
                probs = (
                    np.asarray(self.dist.probabilities, dtype=float).reshape(-1)
                    if self.dist.probabilities is not None
                    else generate_uniform_probabilities(scen.shape[0])
                )
                alt_cvar = np.abs(np.asarray(portfolio_cvar(weights, scen, probs, confidence=0.95))).reshape(-1)
                alternate_risks["CVaR (95%)"] = alt_cvar
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Skipping CVaR overlay for MV frontier: %s", exc)
        
        logger.info(f"Successfully computed Mean-Variance frontier with {weights.shape[1]} portfolios.")
        return PortfolioFrontier(
            weights=weights, returns=returns, risks=risks,
            risk_measure='Volatility',
            asset_names=self.dist.asset_names,
            alternate_risks=alternate_risks,
        )
        
    def mean_cvar_frontier(self, num_portfolios: int = 10, alpha: float = 0.05) -> PortfolioFrontier:
        """Deprecated. Use :meth:`cvar_frontier`.

        Args:
            num_portfolios: Number of portfolios to compute.
            alpha: CVaR tail probability.

        Returns:
            PortfolioFrontier: Mean-CVaR frontier.
        """
        warnings.warn(
            "`mean_cvar_frontier` is deprecated; use `cvar_frontier`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.cvar_frontier(num_portfolios=num_portfolios, alpha=alpha)

    def cvar_frontier(self, num_portfolios: int = 10, alpha: float = 0.05, seed: Optional[int] = None) -> PortfolioFrontier:
        r"""Compute the Mean-CVaR efficient frontier.

        Implementation Notes:
            - This method requires scenarios. If only ``mu`` and ``cov`` are provided,
              it makes a strong modeling assumption to simulate scenarios from a
              multivariate normal distribution.

        Args:
            num_portfolios: The number of portfolios to compute. Defaults to 10.
            alpha: The tail probability for CVaR. Defaults to 0.05.
            seed: Random seed for scenario simulation reproducibility.

        Returns:
            A :class:`PortfolioFrontier` object whose columns are sorted by
            non-decreasing estimation risk so downstream tooling can rely on
            the usual "left = conservative, right = aggressive" ordering. An
            auxiliary ``alternate_risks['Volatility']`` is attached for
            variance-based selection.
        """
        scenarios, probs, mu_for_frontier = self._scenario_inputs(seed=seed)
        if self.market_impact_costs is not None:
            logger.warning("market_impact_costs are ignored by mean-CVaR; use proportional_costs instead.")
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
        risks = np.abs(np.asarray(portfolio_cvar(weights, scenarios, probs, confidence=1.0 - alpha))).reshape(-1)
        raw_risks = risks.copy()
        alternate_risks: Dict[str, np.ndarray] = {}
        try:
            cov = np.cov(scenarios.T, aweights=probs, bias=True)
            vols = np.sqrt(np.sum((weights.T @ cov) * weights.T, axis=1))
            alternate_risks["Volatility"] = vols
        except Exception as exc:  # pragma: no cover - defensive convenience
            logger.debug("Unable to compute volatility proxy for CVaR frontier: %s", exc)

        # Numerical guards: enforce non-decreasing, convex CVaR frontier
        order = np.argsort(returns)
        returns, risks, weights = returns[order], risks[order], weights[:, order]
        raw_risks = raw_risks[order]
        for key, arr in list(alternate_risks.items()):
            alternate_risks[key] = np.asarray(arr, dtype=float)[order]
        alternate_risks[f"CVaR (raw, alpha={alpha:.2f})"] = raw_risks
        # 1) monotone non-decreasing risk w.r.t. return target (feasible set shrinks)
        risks = np.maximum.accumulate(risks)

        # 2) convexify via lower convex envelope in (return, risk)
        def _lower_convex_envelope(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
            """Return indices of the lower convex envelope of (x, y) points.

            Args:
                x: X-coordinates (e.g., returns).
                y: Y-coordinates (e.g., risks).
                eps: Numerical tolerance for convexity checks.

            Returns:
                np.ndarray: Indices of points on the lower envelope.
            """
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
            risk_measure=f'CVaR (alpha={alpha:.2f})',
            asset_names=self.dist.asset_names,
            alternate_risks=alternate_risks,
        )

    def robust_lambda_frontier(
        self,
        num_portfolios: int = 10,
        max_lambda: float = 2.0,
        *,
        lambdas: Optional[Sequence[float]] = None,
        return_cov: Optional[Union[npt.NDArray[np.floating], pd.DataFrame]] = None,
    ) -> PortfolioFrontier:
        r"""Computes a robust frontier based on uncertainty in expected returns.

        Assumptions & Design Choices:
            - This method follows Meucci's robust framework. It assumes that the ``mu``
              and ``cov`` from :class:`AssetsDistribution` represent the posterior mean
              and the posterior scale matrix (for uncertainty), respectively.

        Args:
            num_portfolios: The number of portfolios to compute. Defaults to 10.
            max_lambda: The maximum value for the risk aversion parameter lambda,
              which controls the trade-off between nominal return and robustness.
            lambdas: Optional explicit sequence of \lambda values. When provided, it takes
              precedence over ``num_portfolios``/``max_lambda``.
            return_cov: Optional covariance matrix of returns used to compute a
              volatility overlay for risk targeting/plotting. This is distinct from
              ``dist.cov``, which is treated as the uncertainty covariance.

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
        returns_arr = np.asarray(returns, dtype=float)
        risks_arr = np.asarray(risks, dtype=float)
        weights_arr = np.asarray(weights, dtype=float)
        metadata = [{"lambda": float(lam)} for lam in lambda_list]

        alternate_risks: Dict[str, np.ndarray] = {}
        if return_cov is not None:
            try:
                cov_ref = return_cov
                if isinstance(cov_ref, pd.DataFrame):
                    if self.dist.asset_names is not None:
                        cov_ref = cov_ref.loc[self.dist.asset_names, self.dist.asset_names]
                    cov_ref = cov_ref.to_numpy(dtype=float)
                else:
                    cov_ref = np.asarray(cov_ref, dtype=float)
                if cov_ref.shape != (weights_arr.shape[0], weights_arr.shape[0]):
                    raise ValueError("`return_cov` must be a square matrix matching asset count.")
                vols = np.sqrt(np.sum((weights_arr.T @ cov_ref) * weights_arr.T, axis=1))
                alternate_risks["Volatility"] = vols
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Unable to compute volatility overlay for robust frontier: %s", exc)

        if self.dist.scenarios is not None:
            try:
                scen = np.asarray(self.dist.scenarios, dtype=float)
                probs = (
                    np.asarray(self.dist.probabilities, dtype=float).reshape(-1)
                    if self.dist.probabilities is not None
                    else generate_uniform_probabilities(scen.shape[0])
                )
                if "Volatility" not in alternate_risks:
                    cov = np.cov(scen.T, aweights=probs, bias=True)
                    vols = np.sqrt(np.sum((weights_arr.T @ cov) * weights_arr.T, axis=1))
                    alternate_risks["Volatility"] = vols
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Unable to compute volatility overlay for robust frontier: %s", exc)
            try:
                alt_cvar = np.abs(np.asarray(portfolio_cvar(weights_arr, scen, probs, confidence=0.95))).reshape(-1)
                alternate_risks["CVaR (95%)"] = alt_cvar
            except Exception as exc:  # pragma: no cover
                logger.debug("Unable to compute CVaR overlay for robust frontier: %s", exc)

        if risks_arr.size > 1:
            order = np.argsort(risks_arr, kind="mergesort")
            risks_arr = risks_arr[order]
            returns_arr = returns_arr[order]
            weights_arr = weights_arr[:, order]
            for k, arr in list(alternate_risks.items()):
                alternate_risks[k] = np.asarray(arr, dtype=float)[order]
            metadata = [metadata[i] for i in order]

        logger.info(f"Successfully computed Robust \\lambda-frontier with {weights_arr.shape[1]} portfolios.")
        return PortfolioFrontier(
            weights=weights_arr,
            returns=returns_arr,
            risks=risks_arr,
            risk_measure="Estimation Risk (||\\Sigma'^1/^2w||_2)",
            asset_names=self.dist.asset_names,
            alternate_risks=alternate_risks,
            metadata=metadata,
        )

    def robust_frontier(
        self,
        num_portfolios: int = 10,
        max_lambda: float = 2.0,
        *,
        lambdas: Optional[Sequence[float]] = None,
    ) -> PortfolioFrontier:
        """
        Backwards-compatible alias for :meth:`robust_lambda_frontier`.

        Many ensemble utilities expect a ``robust_frontier`` attribute; this
        thin wrapper keeps that contract intact while reusing the main
        implementation.
        """
        warnings.warn(
            "`robust_frontier` is deprecated; use `robust_lambda_frontier`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.robust_lambda_frontier(
            num_portfolios=num_portfolios,
            max_lambda=max_lambda,
            lambdas=lambdas,
        )

    def mean_variance_portfolio_at_return(self, return_target: float) -> Tuple[pd.Series, float, float]:
        """Deprecated. Use :meth:`min_variance_at_return`.

        Args:
            return_target: Minimum expected return.

        Returns:
            Tuple[pd.Series, float, float]: Weights, return, and risk value.
        """
        warnings.warn(
            "`mean_variance_portfolio_at_return` is deprecated; use `min_variance_at_return`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.min_variance_at_return(return_target)

    def min_variance_at_return(self, return_target: float) -> Tuple[pd.Series, float, float]:
        """
        Solve for the minimum-variance portfolio achieving the target return.

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
            raise InfeasibleOptimizationError(
                f"Optimization infeasible for target return {return_target}: {e}"
            ) from e

        actual_return = self.dist.mu @ weights
        risk = np.sqrt(weights.T @ self.dist.cov @ weights)

        w_series = pd.Series(weights, index=self.dist.asset_names, name=f"MV Portfolio (Return >= {return_target:.4f})")

        logger.info(
            f"Successfully solved for MV portfolio. "
            f"Target Return: {return_target:.4f}, Actual Return: {actual_return:.4f}, Risk: {risk:.4f}"
        )
        return w_series, actual_return, risk

    def mean_cvar_portfolio_at_return(self, return_target: float, alpha: float = 0.05) -> Tuple[pd.Series, float, float]:
        """Deprecated. Use :meth:`min_cvar_at_return`.

        Args:
            return_target: Minimum expected return.
            alpha: CVaR tail probability.

        Returns:
            Tuple[pd.Series, float, float]: Weights, return, and risk value.
        """
        warnings.warn(
            "`mean_cvar_portfolio_at_return` is deprecated; use `min_cvar_at_return`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.min_cvar_at_return(return_target, alpha=alpha)

    def min_cvar_at_return(self, return_target: float, alpha: float = 0.05, seed: Optional[int] = None) -> Tuple[pd.Series, float, float]:
        """
        Solve for the minimum CVaR portfolio that achieves a given expected return.

        This method directly solves the optimization problem for a specific target
        return, rather than interpolating from a pre-computed frontier.

        Args:
            return_target (float): The desired minimum expected return.
            alpha (float): The tail probability for CVaR. Defaults to 0.05.
            seed: Random seed for scenario simulation reproducibility.

        Returns:
            Tuple[pd.Series, float, float]: A tuple containing:
                - **weights** (:class:`pandas.Series`): The weights of the optimal portfolio.
                - **return** (float): The expected return of the portfolio.
                - **risk** (float): The CVaR of the portfolio.

        Raises:
            ValueError: If scenarios cannot be used or generated.
        """
        scenarios, probs, mu_for_cvar = self._scenario_inputs(seed=seed)
        
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
            raise InfeasibleOptimizationError(
                f"Optimization infeasible for target return {return_target}: {e}"
            ) from e

        actual_return = mu_for_cvar @ weights
        risk = float(np.abs(portfolio_cvar(weights, scenarios, probs, confidence=1.0 - alpha)))
        
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
    ) -> Tuple[pd.Series, float, float]:
        """Compute a relaxed risk parity allocation.

        Returns the same triple as other portfolio methods: (weights, return, risk).
        For full diagnostics, use :meth:`relaxed_risk_parity_portfolio_with_diagnostics`.

        Args/Returns: See :meth:`relaxed_risk_parity_portfolio_with_diagnostics`.
        """
        w, ret, risk, _ = self.relaxed_risk_parity_portfolio_with_diagnostics(
            lambda_reg=lambda_reg,
            target_multiplier=target_multiplier,
            return_target=return_target,
        )
        return w, ret, risk

    def relaxed_risk_parity_portfolio_with_diagnostics(
        self,
        *,
        lambda_reg: float = 0.2,
        target_multiplier: Optional[float] = 1.2,
        return_target: Optional[float] = None,
    ) -> Tuple[pd.Series, float, float, Dict[str, Any]]:
        r"""
        Compute a single relaxed risk parity allocation and expose solver diagnostics.

        The routine first solves the baseline risk parity programme (\lambda = 0) to obtain
        the benchmark return :math:`r_{RP} = \mu^{\top}x^{RP}`. Unless an explicit
        ``return_target`` is supplied, the relaxed model is then solved with the adaptive
        target :math:`R = m \cdot \max(r_{RP}, 0)` where ``m`` is ``target_multiplier``.
        Infeasible targets are clipped via a backtracking line-search toward the RP
        return before falling back to the unconstrained problem if necessary.

        Args:
            lambda_reg: Non-negative regulator coefficient :math:`\\lambda`. Setting
                ``0`` recovers the pure risk parity allocation. Defaults to ``0.2``.
            target_multiplier: Optional multiplier :math:`m` governing the adaptive
                target-return rule. Ignored when ``return_target`` is provided. Must
                be ``None`` when pairing with ``lambda_reg == 0``. Defaults to ``1.2``.
            return_target: Explicit target return :math:`R`. When supplied, overrides
                the adaptive rule. The method clips :math:`R` down to the feasible
                region if necessary. Defaults to ``None``.

        Returns:
            Tuple[pd.Series, float, float, Dict[str, Any]]: Optimal portfolio weights
            (Series), achieved return, portfolio volatility, and rich diagnostics
            including variance, marginal risks, risk contributions, target information,
            and any solver warning emitted during target clipping.
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
        sigma_w = np.sqrt(portfolio_variance)
        risk_contributions_pct = (risk_contributions / sigma_w) / sigma_w * 100 if sigma_w > 0 else risk_contributions * 0

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
            "risk_contributions_pct": risk_contributions_pct,
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

        return w_series, achieved_return, np.sqrt(portfolio_variance), diagnostics

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

        Args:
            num_portfolios: Number of grid points when ``target_multipliers`` is omitted.
                Must be positive; includes the upper endpoint ``max_multiplier``.
            max_multiplier: Upper bound for the auto-generated multiplier grid.
                Ignored if ``target_multipliers`` is supplied.
            lambda_reg: Regulator coefficient :math:`\\lambda`. Applies to every relaxed
                point; the optional RP anchor always uses :math:`\\lambda = 0`.
            target_multipliers: Explicit iterable of multipliers. When provided, the
                method skips automatic grid generation and uses the supplied values verbatim.
            include_risk_parity: If ``True`` (default) the frontier prepends the pure
                risk parity solution so downstream plots can intercept the anchor directly.

        Returns:
            PortfolioFrontier: Object containing weights ``(n, k)``, realised returns,
            volatility proxy (standard deviation), and diagnostic metadata for each node.
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
            gamma_sigma_sq: Upper bound on the squared uncertainty radius.

        Returns:
            A tuple containing the portfolio weights, nominal return, and
            squared uncertainty radius.
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
        risk_value = float(result.risk) ** 2
        if risk_value > gamma_sigma_sq:
            logger.debug(
                "Capping squared estimation risk to gamma_sigma_sq (risk=%.6f, cap=%.6f).",
                risk_value,
                gamma_sigma_sq,
            )
            risk_value = float(gamma_sigma_sq)

        logger.info(
            f"Successfully solved for robust \\gamma-portfolio. "
            f"Nominal Return: {result.nominal_return:.4f}, Estimation Risk (squared): {risk_value:.4f}"
        )
        return w_series, result.nominal_return, risk_value

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
            """Return a deep copy of the current distribution.

            Returns:
                AssetsDistribution: Copied distribution instance.
            """
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
        """Proxy to :func:`pyvallocation.ensembles.assemble_portfolio_ensemble`.

        Args:
            specs: Sequence of ensemble specifications.
            **kwargs: Forwarded to :func:`assemble_portfolio_ensemble`.

        Returns:
            EnsembleResult: Aggregated ensemble output.
        """
        from .ensembles import assemble_portfolio_ensemble

        return assemble_portfolio_ensemble(specs, **kwargs)

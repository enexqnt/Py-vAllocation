"""Discrete allocation helpers transforming continuous weights into share counts."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from scipy import optimize
except ImportError as exc:  # pragma: no cover - SciPy is a hard dependency
    raise ImportError("scipy is required for pyvallocation.discrete_allocation") from exc

SeriesLike = Union[pd.Series, Mapping[str, float]]
LotSizeLike = Optional[Union[pd.Series, Mapping[str, int]]]


def _to_series(data: SeriesLike, name: str) -> pd.Series:
    """Coerce ``data`` into a non-empty pandas Series.

    Args:
        data: Series or mapping of asset -> value.
        name: Label used for validation error messages.

    Returns:
        pd.Series: Series with ``float`` dtype.

    Raises:
        ValueError: If ``data`` is empty.
        TypeError: If ``data`` is not a Series or mapping.
    """
    if isinstance(data, pd.Series):
        if data.empty:
            raise ValueError(f"`{name}` must contain at least one asset.")
        return data.astype(float)
    if isinstance(data, Mapping):
        if not data:
            raise ValueError(f"`{name}` mapping must contain at least one asset.")
        return pd.Series(data, dtype=float)
    raise TypeError(f"`{name}` must be a pandas Series or a mapping of asset -> value.")


def _lot_sizes_to_series(lot_sizes: LotSizeLike, index: Iterable[str]) -> pd.Series:
    """Normalize lot size inputs into a validated Series.

    Args:
        lot_sizes: Optional mapping/Series of integer lot sizes.
        index: Asset index that the resulting Series should align to.

    Returns:
        pd.Series: Integer lot sizes aligned to ``index`` with missing values set to 1.

    Raises:
        TypeError: If ``lot_sizes`` is not a mapping or Series.
        ValueError: If any lot sizes are non-positive.
    """
    if lot_sizes is None:
        return pd.Series(1, index=index, dtype=int)
    if isinstance(lot_sizes, pd.Series):
        series = lot_sizes.astype(int)
    elif isinstance(lot_sizes, Mapping):
        series = pd.Series(lot_sizes, dtype=int)
    else:
        raise TypeError("`lot_sizes` must be a mapping or pandas Series if provided.")
    series = series.reindex(index).fillna(1).astype(int)
    if (series <= 0).any():
        raise ValueError("All lot sizes must be positive integers.")
    return series


@dataclass
class DiscreteAllocationInput:
    """Container for validated discrete allocation inputs.

    Key fields:
        weights: Continuous target weights.
        latest_prices: Latest asset prices.
        total_value: Total portfolio value.
        lot_sizes: Optional lot sizes per asset.
    """

    weights: SeriesLike
    latest_prices: SeriesLike
    total_value: float
    lot_sizes: LotSizeLike = None

    def __post_init__(self) -> None:
        """Validate and normalize inputs after dataclass initialization.

        Raises:
            ValueError: If inputs are inconsistent or invalid.
        """
        weights = _to_series(self.weights, "weights").astype(float)
        latest_prices = _to_series(self.latest_prices, "latest_prices").astype(float)

        missing_prices = set(weights.index) - set(latest_prices.index)
        if missing_prices:
            missing_str = ", ".join(sorted(missing_prices))
            raise ValueError(f"Missing latest prices for assets: {missing_str}")

        weights = weights.loc[~np.isclose(weights, 0.0)]
        if weights.empty:
            raise ValueError("`weights` must contain at least one non-zero entry.")

        if (weights < 0).any():
            raise ValueError("Negative weights are not supported by the discrete allocation helpers.")

        latest_prices = latest_prices.reindex(weights.index)
        if (latest_prices <= 0).any():
            raise ValueError("All latest prices must be strictly positive.")

        if self.total_value <= 0:
            raise ValueError("`total_value` must be positive.")

        lot_sizes = _lot_sizes_to_series(self.lot_sizes, weights.index)

        total_weight = weights.sum()
        if not np.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("Weights must sum to a positive finite value.")
        weights = weights / total_weight

        # Assign back validated values
        self.weights = weights
        self.latest_prices = latest_prices
        self.lot_sizes = lot_sizes

    @property
    def asset_names(self) -> Sequence[str]:
        """Tuple of asset names inferred from the weights index.

        Returns:
            Sequence[str]: Asset labels.
        """
        return tuple(self.weights.index)


@dataclass
class DiscreteAllocationResult:
    """Discrete allocation output bundle.

    Key fields:
        shares: Non-zero share counts by asset.
        leftover_cash: Uninvested cash amount.
        achieved_weights: Realized weights from share allocation (relative to total portfolio value).
        tracking_error: L1 weight deviation (sum of absolute differences between target and achieved weights).
    """

    shares: Dict[str, int]
    leftover_cash: float
    achieved_weights: pd.Series
    tracking_error: float

    def as_series(self) -> pd.Series:
        """Return the allocated share counts as a pandas Series.

        Returns:
            pd.Series: Share counts indexed by asset.
        """
        return pd.Series(self.shares, dtype=int)


def _build_result(inputs: DiscreteAllocationInput, raw_shares: pd.Series) -> DiscreteAllocationResult:
    """Assemble a :class:`DiscreteAllocationResult` from raw share counts.

    Args:
        inputs: Validated allocation inputs.
        raw_shares: Raw share counts (possibly fractional).

    Returns:
        DiscreteAllocationResult: Normalized allocation summary.
    """
    shares = raw_shares.astype(int)
    values = shares * inputs.latest_prices
    portfolio_value = float(values.sum())
    leftover = float(inputs.total_value - portfolio_value)
    # Numerical noise guard
    if leftover < 0 and abs(leftover) < 1e-8:
        leftover = 0.0
    elif leftover < 0:
        raise RuntimeError("Allocation exceeded available funds by more than tolerance.")

    achieved = values / inputs.total_value

    achieved = achieved.reindex(inputs.weights.index, fill_value=0.0)
    tracking_error = float(np.sum(np.abs(inputs.weights - achieved)))

    non_zero = shares[shares != 0]
    shares_dict = {asset: int(count) for asset, count in non_zero.items()}

    return DiscreteAllocationResult(
        shares=shares_dict,
        leftover_cash=leftover,
        achieved_weights=achieved,
        tracking_error=tracking_error,
    )


def allocate_greedy(
    inputs: DiscreteAllocationInput,
    max_iterations: Optional[int] = None,
    *,
    fallback_mode: str = "auto",
    fallback_kwargs: Optional[Dict[str, Any]] = None,
) -> DiscreteAllocationResult:
    """Greedy rounding of weights into integer share counts.

    The allocator performs an initial floor pass followed by a deficit-driven
    topping up routine. If the greedy loop cannot make progress (e.g. due to
    a tight iteration cap), ``fallback_mode`` controls whether the MILP
    allocator should be tried as a secondary attempt.
    """

    mode = (fallback_mode or "none").lower()
    if mode not in {"auto", "milp", "none"}:
        raise ValueError("`fallback_mode` must be one of {'auto', 'milp', 'none'}.")

    weights = inputs.weights.sort_values(ascending=False)
    prices = inputs.latest_prices.loc[weights.index]
    lot_sizes = inputs.lot_sizes.loc[weights.index]
    cost_per_lot = prices * lot_sizes
    tolerance = 1e-12

    shares = pd.Series(0, index=weights.index, dtype=int)
    available_cash = float(inputs.total_value)

    # First pass: floor allocation
    for asset, weight in weights.items():
        price = prices[asset]
        lot_size = lot_sizes[asset]
        lot_cost = float(price * lot_size)
        if lot_cost > available_cash:
            continue
        target_cash = weight * inputs.total_value
        lots = int(np.floor(target_cash / lot_cost))
        if lots <= 0:
            continue
        cost = lots * lot_cost
        shares[asset] += lots * lot_size
        available_cash -= cost

    if shares.sum() == 0 and available_cash > 0:
        logger.warning(
            "Floor pass allocated zero shares. Minimum lot cost (%.2f) may exceed "
            "per-asset target values.",
            float(cost_per_lot.min()),
        )

    min_cost = float(cost_per_lot.min())
    if max_iterations is None:
        max_iterations = max(len(weights) * 100, 1)

    iteration = 0
    unaffordable: set[str] = set()
    last_deficits = weights.copy()
    terminated_via_break = False

    while available_cash + tolerance >= min_cost and iteration < max_iterations:
        values = shares * prices
        current_weights = values / inputs.total_value
        deficits = weights - current_weights

        deficits[deficits < 0] = 0.0
        if unaffordable:
            deficits.loc[list(unaffordable)] = 0.0

        last_deficits = deficits

        if deficits.max() <= tolerance:
            terminated_via_break = True
            break
        candidate = deficits.idxmax()
        price = prices[candidate]
        lot_size = lot_sizes[candidate]
        candidate_cost = float(cost_per_lot[candidate])

        if candidate_cost > available_cash + tolerance:
            unaffordable.add(candidate)
            if len(unaffordable) == len(weights):
                terminated_via_break = True
                break
            continue

        # Desired additional investment for this asset.
        desired_cash = deficits[candidate] * inputs.total_value
        possible_lots = int(np.floor((available_cash + tolerance) / candidate_cost))
        target_lots = int(np.floor(desired_cash / candidate_cost)) if candidate_cost > 0 else 0
        lots_to_buy = min(possible_lots, max(target_lots, 1))

        if lots_to_buy <= 0:
            unaffordable.add(candidate)
            if len(unaffordable) == len(weights):
                terminated_via_break = True
                break
            continue

        shares[candidate] += lots_to_buy * lot_size
        available_cash -= lots_to_buy * candidate_cost
        iteration += 1

        if candidate in unaffordable:
            unaffordable.remove(candidate)

    max_iterations_exhausted = max_iterations is not None and iteration >= max_iterations and not terminated_via_break

    should_try_fallback = False
    if mode in {"auto", "milp"}:
        still_cash_for_min_lot = available_cash + tolerance >= min_cost
        large_deficit_left = float(last_deficits.max()) > 1e-6
        allocated_nothing = shares.sum() == 0 and inputs.total_value + tolerance >= min_cost

        if max_iterations_exhausted or (still_cash_for_min_lot and large_deficit_left) or allocated_nothing:
            should_try_fallback = True

    if should_try_fallback:
        fallback_args = fallback_kwargs or {}
        try:
            return allocate_mip(inputs, **fallback_args)
        except Exception:
            if mode == "milp":
                raise

    return _build_result(inputs, shares)


def allocate_mip(
    inputs: DiscreteAllocationInput,
    cash_penalty: float = 1.0,
    max_cash: Optional[float] = None,
    solver_options: Optional[Dict[str, Union[int, float]]] = None,
) -> DiscreteAllocationResult:
    """Mixed-integer allocation minimizing L1 tracking error and leftover cash.

    Args:
        inputs: Validated allocation inputs.
        cash_penalty: Penalty on leftover cash in the objective.
        max_cash: Optional upper bound on leftover cash.
        solver_options: Optional SciPy MILP solver options.

    Returns:
        DiscreteAllocationResult: Allocation result with share counts.
    """

    if not hasattr(optimize, "milp"):
        raise RuntimeError(
            "scipy.optimize.milp is not available. Upgrade SciPy to >=1.11 to use the MILP allocator."
        )

    weights = inputs.weights
    prices = inputs.latest_prices
    lot_sizes = inputs.lot_sizes

    n = len(weights)
    if n == 0:
        raise ValueError("No assets to allocate.")

    price_per_lot = prices.values * lot_sizes.values
    target_values = weights.values * inputs.total_value

    # Decision variables: x (n integers), u (n continuous), r (cash)
    num_vars = 2 * n + 1
    c = np.concatenate([
        np.zeros(n),
        np.ones(n),
        [cash_penalty],
    ])

    max_shares_per_lot = np.where(price_per_lot > 0, np.floor(inputs.total_value / price_per_lot).astype(int), 0)
    max_shares_per_lot = np.clip(max_shares_per_lot + 1, 0, None)

    upper_cash = inputs.total_value if max_cash is None else min(max_cash, inputs.total_value)

    lb = np.zeros(num_vars)
    ub = np.concatenate([
        max_shares_per_lot.astype(float),
        np.full(n, inputs.total_value, dtype=float),
        [upper_cash],
    ])
    bounds = optimize.Bounds(lb, ub)

    integrality = np.concatenate([
        np.ones(n, dtype=int),
        np.zeros(n + 1, dtype=int),
    ])

    # Equality constraint: sum(price_per_lot * x) + r = total_value
    equality_matrix = np.zeros((1, num_vars))
    equality_matrix[0, :n] = price_per_lot
    equality_matrix[0, -1] = 1.0
    equality_constraint = optimize.LinearConstraint(
        equality_matrix,
        lb=np.array([inputs.total_value]),
        ub=np.array([inputs.total_value]),
    )

    # Inequalities for absolute error handling
    upper_matrix = np.zeros((n, num_vars))
    lower_matrix = np.zeros((n, num_vars))
    for i in range(n):
        upper_matrix[i, i] = price_per_lot[i]
        upper_matrix[i, n + i] = -1.0
        lower_matrix[i, i] = -price_per_lot[i]
        lower_matrix[i, n + i] = -1.0

    upper_constraint = optimize.LinearConstraint(
        upper_matrix,
        lb=-np.inf * np.ones(n),
        ub=target_values,
    )
    lower_constraint = optimize.LinearConstraint(
        lower_matrix,
        lb=-np.inf * np.ones(n),
        ub=-target_values,
    )

    options = solver_options or {}
    result = optimize.milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=[equality_constraint, upper_constraint, lower_constraint],
        options=options,
    )

    if not result.success:
        raise RuntimeError(f"MILP solver failed with status {result.status}: {result.message}")

    solution = result.x
    lots = np.rint(solution[:n]).astype(int)
    shares = pd.Series(lots * lot_sizes.values, index=weights.index, dtype=int)

    return _build_result(inputs, shares)


def discretize_weights(
    weights: SeriesLike,
    latest_prices: SeriesLike,
    total_value: float,
    *,
    method: str = "greedy",
    lot_sizes: LotSizeLike = None,
    **kwargs,
) -> DiscreteAllocationResult:
    """Routes to the requested discrete allocation algorithm.

    Args:
        weights: Target weights.
        latest_prices: Latest asset prices.
        total_value: Total portfolio value.
        method: ``"greedy"`` or ``"milp"``.
        lot_sizes: Optional lot sizes per asset.
        **kwargs: Extra keyword arguments forwarded to the allocator.

    Returns:
        DiscreteAllocationResult: Allocation result with share counts.
    """

    inputs = DiscreteAllocationInput(
        weights=weights,
        latest_prices=latest_prices,
        total_value=total_value,
        lot_sizes=lot_sizes,
    )

    method = method.lower()
    if method in {"greedy", "rounddown"}:
        return allocate_greedy(inputs, **kwargs)
    if method in {"milp", "mip", "integer", "lp"}:
        return allocate_mip(inputs, **kwargs)
    raise ValueError(f"Unknown discretization method '{method}'.")

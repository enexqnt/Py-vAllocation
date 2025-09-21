"""Discrete allocation helpers transforming continuous weights into share counts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    from scipy import optimize
except ImportError as exc:  # pragma: no cover - SciPy is a hard dependency
    raise ImportError("scipy is required for pyvallocation.discrete_allocation") from exc

SeriesLike = Union[pd.Series, Mapping[str, float]]
LotSizeLike = Optional[Union[pd.Series, Mapping[str, int]]]


def _to_series(data: SeriesLike, name: str) -> pd.Series:
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
    """Container for validated discrete allocation inputs."""

    weights: SeriesLike
    latest_prices: SeriesLike
    total_value: float
    lot_sizes: LotSizeLike = None

    def __post_init__(self) -> None:
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
        return tuple(self.weights.index)


@dataclass
class DiscreteAllocationResult:
    shares: Dict[str, int]
    leftover_cash: float
    achieved_weights: pd.Series
    tracking_error: float

    def as_series(self) -> pd.Series:
        return pd.Series(self.shares, dtype=int)


def _build_result(inputs: DiscreteAllocationInput, raw_shares: pd.Series) -> DiscreteAllocationResult:
    shares = raw_shares.astype(int)
    values = shares * inputs.latest_prices
    portfolio_value = float(values.sum())
    leftover = float(inputs.total_value - portfolio_value)
    # Numerical noise guard
    if leftover < 0 and abs(leftover) < 1e-8:
        leftover = 0.0
    elif leftover < 0:
        raise RuntimeError("Allocation exceeded available funds by more than tolerance.")

    if portfolio_value > 0:
        achieved = values / portfolio_value
    else:
        achieved = pd.Series(0.0, index=inputs.weights.index)

    achieved = achieved.reindex(inputs.weights.index, fill_value=0.0)
    tracking_error = float(np.sqrt(np.mean((inputs.weights - achieved) ** 2)))

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
) -> DiscreteAllocationResult:
    """Greedy rounding of weights into integer share counts."""

    weights = inputs.weights.sort_values(ascending=False)
    prices = inputs.latest_prices.loc[weights.index]
    lot_sizes = inputs.lot_sizes.loc[weights.index]

    shares = pd.Series(0, index=weights.index, dtype=int)
    available_cash = float(inputs.total_value)

    # First pass: floor allocation
    for asset, weight in weights.items():
        price = prices[asset]
        lot_size = lot_sizes[asset]
        cost_per_lot = price * lot_size
        if cost_per_lot > available_cash:
            continue
        target_cash = weight * inputs.total_value
        lots = int(np.floor(target_cash / cost_per_lot))
        if lots <= 0:
            continue
        cost = lots * cost_per_lot
        shares[asset] += lots * lot_size
        available_cash -= cost

    min_cost = float((prices * lot_sizes).min())
    if max_iterations is None:
        max_iterations = max(len(weights) * 100, 1)

    iteration = 0
    while available_cash + 1e-12 >= min_cost and iteration < max_iterations:
        values = shares * prices
        invested = values.sum()
        if invested <= 0:
            break
        current_weights = values / invested
        deficits = weights - current_weights
        deficits[deficits < 0] = 0.0
        if deficits.max() <= 1e-12:
            break
        candidate = deficits.idxmax()
        price = prices[candidate]
        lot_size = lot_sizes[candidate]
        cost_per_lot = price * lot_size
        if cost_per_lot > available_cash + 1e-12:
            deficits[candidate] = 0
            if deficits.max() <= 1e-12:
                break
            continue
        shares[candidate] += lot_size
        available_cash -= cost_per_lot
        iteration += 1

    return _build_result(inputs, shares)


def allocate_mip(
    inputs: DiscreteAllocationInput,
    cash_penalty: float = 1.0,
    max_cash: Optional[float] = None,
    solver_options: Optional[Dict[str, Union[int, float]]] = None,
) -> DiscreteAllocationResult:
    """Mixed-integer allocation minimizing L1 tracking error and leftover cash."""

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
    """Routes to the requested discrete allocation algorithm."""

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

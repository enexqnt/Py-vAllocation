from __future__ import annotations

import logging
import numbers
import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

logger = logging.getLogger(__name__)

Number = Union[int, float]
BoundsLike = Union[
    Tuple[Number, Number],
    Sequence[Tuple[Number, Number]],
    Dict[int, Tuple[Number, Number]],
]
RelBound = Tuple[int, int, Number]
EqRow = Tuple[Sequence[Number], Number]


@dataclass(frozen=True)
class Constraints:
    """Typed, immutable constraint specification for portfolio optimisers.

    All fields have sensible defaults so that ``Constraints()`` produces
    a long-only, fully-invested constraint set.

    Examples:
        >>> c = Constraints()                          # long-only, sum=1
        >>> c = Constraints(bounds=(0.0, 0.3))         # per-asset cap at 30%
        >>> c = Constraints(group_constraints={
        ...     "Equity": ([0, 1, 2], 0.2, 0.6),      # 20-60% in equities
        ... })
        >>> c = Constraints.from_dict({"long_only": True, "total_weight": 1.0})
    """
    long_only: bool = True
    total_weight: Optional[Number] = 1.0
    bounds: Optional[BoundsLike] = None
    relative_bounds: Optional[Sequence[RelBound]] = None
    group_constraints: Optional[Dict[str, Tuple[Sequence[int], Number, Number]]] = None
    additional_G_h: Optional[Sequence[Tuple[Sequence[Number], Number]]] = None
    additional_A_b: Optional[Sequence[EqRow]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Constraints":
        """Create a Constraints instance from a dictionary.

        Unrecognised keys are silently ignored for backward compatibility.
        """
        import dataclasses
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def to_matrices(self, n_assets: int) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray],
        Optional[np.ndarray], Optional[np.ndarray],
    ]:
        """Build CVXOPT-compatible (G, h, A, b) constraint matrices.

        Args:
            n_assets: Number of portfolio assets.

        Returns:
            Tuple of (G, h, A, b) arrays or None.
        """
        return build_G_h_A_b(
            n_assets,
            total_weight=self.total_weight,
            long_only=self.long_only,
            bounds=self.bounds,
            relative_bounds=self.relative_bounds,
            group_constraints=self.group_constraints,
            additional_G_h=self.additional_G_h,
            additional_A_b=self.additional_A_b,
        )


def _check_number(x: Number, name: str) -> None:
    """Validate that ``x`` is a finite real number.

    Args:
        x: Candidate numeric value.
        name: Parameter name for error messaging.

    Raises:
        TypeError: If ``x`` is not a real number.
        ValueError: If ``x`` is not finite.
    """
    if not isinstance(x, numbers.Real):
        logger.error("%s must be a real number, got %s", name, type(x))
        raise TypeError(f"{name} must be a real number, got {type(x)}")
    if not np.isfinite(x):
        logger.error("%s must be finite, got %s", name, x)
        raise ValueError(f"{name} must be finite, got {x}")


def build_G_h_A_b(
    n_assets: int,
    *,
    total_weight: Optional[Number] = 1.0,
    long_only: bool = True,
    bounds: Optional[BoundsLike] = None,
    relative_bounds: Optional[Sequence[RelBound]] = None,
    group_constraints: Optional[Dict[str, Tuple[Sequence[int], Number, Number]]] = None,
    additional_G_h: Optional[Sequence[Tuple[Sequence[Number], Number]]] = None,
    additional_A_b: Optional[Sequence[EqRow]] = None,
    return_none_if_empty: bool = True,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Assemble inequality and equality constraints for portfolio optimisers.

    The helper converts high-level constraint specifications (long-only,
    box bounds, pairwise relative bounds, and custom rows) into the ``G``,
    ``h``, ``A`` and ``b`` matrices expected by CVXOPT-based solvers.

    Args:
        n_assets: Number of portfolio weights to constrain.
        total_weight: Target sum of weights. ``None`` disables the equality row.
        long_only: When ``True`` append ``-I w <= 0`` so weights remain non-negative.
        bounds: Either a single ``(lower, upper)`` tuple applied to every asset,
            a sequence of per-asset tuples, or a mapping ``asset -> (lower, upper)``.
        relative_bounds: Sequence of triples ``(i, j, bound)`` implementing
            ``w_i - w_j <= bound`` style constraints.
        group_constraints: Mapping of group name to ``(indices, lower, upper)``
            tuples.  Each entry constrains the sum of weights in the group to
            lie between ``lower`` and ``upper``.
        additional_G_h: Extra inequality rows supplied as ``(row, rhs)`` pairs.
        additional_A_b: Extra equality rows supplied as ``(row, rhs)`` pairs.
        return_none_if_empty: When ``True`` return ``None`` instead of empty arrays.

    Returns:
        Tuple ``(G, h, A, b)`` suitable for CVXOPT/QP front-end functions. Each entry
        is ``None`` when no constraint of that type is required.
    """
    if not isinstance(n_assets, int) or n_assets <= 0:
        logger.error("n_assets must be a positive integer, got %s", n_assets)
        raise ValueError("n_assets must be a positive integer")

    if total_weight is not None:
        _check_number(total_weight, "total_weight")
        if total_weight == 0:
            logger.warning("total_weight=0 creates a degenerate all-zero portfolio.")

    G_rows: List[np.ndarray] = []
    h_vals: List[Number] = []
    A_rows: List[np.ndarray] = []
    b_vals: List[Number] = []

    if long_only:
        G_rows.append(-np.eye(n_assets))
        h_vals.extend([0.0] * n_assets)

    if bounds is not None:
        if isinstance(bounds, tuple):
            if len(bounds) != 2:
                logger.error(
                    "bounds tuple must be (lower, upper), got length %d", len(bounds)
                )
                raise ValueError("bounds tuple must be (lower, upper)")
            lower, upper = bounds
            if lower is not None:
                _check_number(lower, "lower bound")
            if upper is not None:
                _check_number(upper, "upper bound")
            if lower is not None and upper is not None and lower > upper:
                logger.error("lower bound %s greater than upper bound %s", lower, upper)
                raise ValueError("lower bound greater than upper bound")

            if lower is not None:
                G_rows.append(-np.eye(n_assets))
                h_vals.extend([-lower] * n_assets)
            if upper is not None:
                G_rows.append(np.eye(n_assets))
                h_vals.extend([upper] * n_assets)

        elif isinstance(bounds, dict):
            for idx, lu in bounds.items():
                if not (0 <= idx < n_assets):
                    logger.error(
                        "asset index %d out of range (0..%d)", idx, n_assets - 1
                    )
                    raise IndexError(
                        f"asset index {idx} out of range (0..{n_assets-1})"
                    )
                if len(lu) != 2:
                    logger.error(
                        "each bounds value must be (lower, upper), got length %d",
                        len(lu),
                    )
                    raise ValueError("each bounds value must be (lower, upper)")
                lower, upper = lu
                if lower is not None:
                    _check_number(lower, f"lower bound for asset {idx}")
                if upper is not None:
                    _check_number(upper, f"upper bound for asset {idx}")
                if lower is not None and upper is not None and lower > upper:
                    logger.error(
                        "asset %d: lower bound %s > upper bound %s", idx, lower, upper
                    )
                    raise ValueError(f"asset {idx}: lower bound > upper bound")
                if lower is not None:
                    row = np.zeros(n_assets)
                    row[idx] = -1
                    G_rows.append(row)
                    h_vals.append(-lower)
                if upper is not None:
                    row = np.zeros(n_assets)
                    row[idx] = 1
                    G_rows.append(row)
                    h_vals.append(upper)

        else:
            bounds_seq = list(bounds)  # type: ignore[arg-type]
            if len(bounds_seq) != n_assets:
                logger.error(
                    "bounds list length %d must equal n_assets %d",
                    len(bounds_seq),
                    n_assets,
                )
                raise ValueError("bounds list length must equal n_assets")
            for idx, (lower, upper) in enumerate(bounds_seq):
                if lower is not None:
                    _check_number(lower, f"lower bound for asset {idx}")
                if upper is not None:
                    _check_number(upper, f"upper bound for asset {idx}")
                if lower is not None and upper is not None and lower > upper:
                    logger.error(
                        "asset %d: lower bound %s > upper bound %s", idx, lower, upper
                    )
                    raise ValueError(f"asset {idx}: lower bound > upper bound")
                if lower is not None:
                    row = np.zeros(n_assets)
                    row[idx] = -1
                    G_rows.append(row)
                    h_vals.append(-lower)
                if upper is not None:
                    row = np.zeros(n_assets)
                    row[idx] = 1
                    G_rows.append(row)
                    h_vals.append(upper)

    if relative_bounds is not None:
        for triple in relative_bounds:
            if len(triple) != 3:
                logger.error(
                    "each relative_bounds entry must be (i, j, k), got length %d",
                    len(triple),
                )
                raise ValueError("each relative_bounds entry must be (i, j, k)")
            i, j, k = triple
            if not (0 <= i < n_assets and 0 <= j < n_assets):
                logger.error("relative_bounds indices %d,%d out of range", i, j)
                raise IndexError(f"relative_bounds indices {i},{j} out of range")
            if i == j:
                logger.error("relative_bounds: i and j must differ, got i=j=%d", i)
                raise ValueError(f"relative_bounds: i and j must differ (got i=j={i})")
            _check_number(k, "bound in relative_bounds")
            row = np.zeros(n_assets)
            row[i] = 1
            row[j] = -1
            G_rows.append(row)
            h_vals.append(float(k))

    if group_constraints is not None:
        for name, (indices, lo, hi) in group_constraints.items():
            idx = list(indices)
            if not all(0 <= i < n_assets for i in idx):
                raise IndexError(f"group '{name}' contains out-of-range asset indices.")
            _check_number(lo, f"lower bound for group '{name}'")
            _check_number(hi, f"upper bound for group '{name}'")
            if lo > hi:
                raise ValueError(f"group '{name}': lower bound {lo} > upper bound {hi}.")
            # sum(w_i for i in group) <= hi
            row_upper = np.zeros(n_assets)
            row_upper[idx] = 1.0
            G_rows.append(row_upper)
            h_vals.append(float(hi))
            # sum(w_i for i in group) >= lo  →  -sum(w_i) <= -lo
            row_lower = np.zeros(n_assets)
            row_lower[idx] = -1.0
            G_rows.append(row_lower)
            h_vals.append(float(-lo))

    if additional_G_h is not None:
        for row, rhs in additional_G_h:
            row_arr = np.asarray(row, dtype=float)
            if row_arr.size != n_assets:
                logger.error("additional_G_h row length %d mismatch", row_arr.size)
                raise ValueError("additional_G_h row length mismatch")
            _check_number(rhs, "rhs in additional_G_h")
            G_rows.append(row_arr)
            h_vals.append(rhs)

    if total_weight is not None:
        A_rows.append(np.ones(n_assets))
        b_vals.append(total_weight)

    if additional_A_b is not None:
        for row, rhs in additional_A_b:
            row_arr = np.asarray(row, dtype=float)
            if row_arr.size != n_assets:
                logger.error("additional_A_b row length %d mismatch", row_arr.size)
                raise ValueError("additional_A_b row length mismatch")
            _check_number(rhs, "rhs in additional_A_b")
            A_rows.append(row_arr)
            b_vals.append(rhs)

    if not long_only and bounds is None and not relative_bounds and not additional_G_h:
        warnings.warn(
            "No position bounds given and long_only=False - feasible set may be "
            "unbounded -> optimisation can fail.",
            UserWarning,
            stacklevel=2,
        )
        logger.warning(
            "No position bounds given and long_only=False - feasible set may be unbounded."
        )

    if (
        long_only
        and bounds is not None
        and isinstance(bounds, tuple)
        and bounds[0] is not None
        and bounds[0] == 0
    ):
        warnings.warn(
            "long_only=True already enforces w >= 0; supplying lower bound = 0 "
            "duplicates that constraint.",
            UserWarning,
            stacklevel=2,
        )

    def _stack(rows: List[np.ndarray]) -> Optional[np.ndarray]:
        """Stack constraint rows into a 2D array or return ``None`` if empty.

        Args:
            rows: List of row arrays.

        Returns:
            Optional[np.ndarray]: Stacked 2D array or ``None``.
        """
        if rows:
            return np.vstack(rows)
        return None if return_none_if_empty else np.zeros((0, n_assets))

    G = _stack(G_rows)
    h = (
        np.asarray(h_vals)
        if h_vals
        else (None if return_none_if_empty else np.zeros(0))
    )
    A = _stack(A_rows)
    b = (
        np.asarray(b_vals, float)
        if b_vals
        else (None if return_none_if_empty else np.zeros(0))
    )

    logger.debug("Built constraint matrices G, h, A, b.")
    return G, h, A, b

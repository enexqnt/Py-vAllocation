"""
Optimization module for portfolio construction.

Includes utilities for building constraint matrices and classes for
mean-variance and mean-CVaR portfolio optimization.

Uses cvxopt for quadratic and linear programming.
"""
# MeanCVaR and MeanVariance functions are adapted from fortituto-tech https://github.com/fortitudo-tech/fortitudo.tech

from __future__ import annotations

import logging
import numbers
import warnings
from copy import copy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from cvxopt import matrix, solvers, sparse
from cvxopt.solvers import lp, options

from .optional import HAS_PANDAS, pd

options["glpk"] = {"msg_lev": "GLP_MSG_OFF"}
options["show_progress"] = False
cvar_options = {}

logger = logging.getLogger(__name__)

Number = Union[int, float]
BoundsLike = Union[
    Tuple[Number, Number],
    Sequence[Tuple[Number, Number]],
    Dict[int, Tuple[Number, Number]],
]
RelBound = Tuple[int, int, Number]
EqRow = Tuple[Sequence[Number], Number]


def _check_number(x: Number, name: str) -> None:
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
    additional_G_h: Optional[Sequence[Tuple[Sequence[Number], Number]]] = None,
    additional_A_b: Optional[Sequence[EqRow]] = None,
    return_none_if_empty: bool = True,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    if not isinstance(n_assets, int) or n_assets <= 0:
        logger.error("n_assets must be a positive integer, got %s", n_assets)
        raise ValueError("n_assets must be a positive integer")

    if total_weight is not None:
        _check_number(total_weight, "total_weight")

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
            _check_number(k, "k in relative bound")
            if k < 0:
                logger.error("k in relative bound must be non-negative, got %s", k)
                raise ValueError("k in relative bound must be non-negative")
            row = np.zeros(n_assets)
            row[i] = 1
            row[j] = -k
            G_rows.append(row)
            h_vals.append(0.0)

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
            "No position bounds given and long_only=False – feasible set may be "
            "unbounded → optimisation can fail.",
            UserWarning,
            stacklevel=2,
        )
        logger.warning(
            "No position bounds given and long_only=False – feasible set may be unbounded."
        )

    if (
        long_only
        and bounds is not None
        and isinstance(bounds, tuple)
        and bounds[0] is not None
        and bounds[0] >= 0
    ):
        warnings.warn(
            "long_only=True already enforces w ≥ 0; supplying a non-negative lower "
            "bound duplicates that constraint.",
            UserWarning,
            stacklevel=2,
        )
        logger.warning(
            "long_only=True already enforces w ≥ 0; supplying a non-negative lower bound duplicates that constraint."
        )

    def _stack(rows: List[np.ndarray]) -> Optional[np.ndarray]:
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


class Optimization:
    """Base class containing utilities shared by optimizers."""

    _G: matrix
    _h: matrix
    _A: matrix
    _b: matrix
    _P: matrix
    _q: matrix
    _expected_return_row: matrix
    _I: int
    _mean: np.ndarray

    def _calculate_max_expected_return(self, feasibility_check: bool = False) -> float:
        c = (
            matrix(np.zeros(self._G.size[1]))
            if feasibility_check
            else self._expected_return_row.T
        )
        sol = solvers.lp(c, self._G, self._h, self._A, self._b, solver="glpk")
        if sol["status"] == "optimal":
            return -sol["primal objective"]
        if feasibility_check:
            logger.error("Constraints are infeasible.")
            raise ValueError(
                "Constraints are infeasible. Please specify feasible constraints."
            )
        logger.error("Expected return is unbounded.")
        raise ValueError(
            "Expected return is unbounded. Unable to compute efficient frontier."
        )

    def efficient_frontier(self, num_portfolios: int = 9) -> np.ndarray:
        frontier = np.full((self._I, num_portfolios), np.nan)
        frontier[:, 0] = self.efficient_portfolio(return_target=None)[:, 0]

        min_er = self._mean @ frontier[:, 0]
        max_er = self._calculate_max_expected_return()
        delta = (max_er - min_er) / (num_portfolios - 1)
        targets = min_er + delta * np.arange(1, num_portfolios)

        for k, target in enumerate(targets, start=1):
            frontier[:, k] = self.efficient_portfolio(return_target=target)[:, 0]
        logger.debug("Computed efficient frontier with %d portfolios.", num_portfolios)
        return frontier


class MeanVariance(Optimization):
    """Classical mean-variance optimization with flexible constraints."""

    def __init__(
        self,
        mean: np.ndarray,
        covariance_matrix: np.ndarray,
        G: Optional[np.ndarray] = None,
        h: Optional[np.ndarray] = None,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        v: Optional[np.ndarray] = None,
        *,
        tcost_lambda: Union[float, Sequence[float]] = 0.0,
        prev_weights: Optional[np.ndarray] = None,
    ):
        self._I = len(mean)
        self._mean = mean

        if np.isscalar(tcost_lambda):
            if tcost_lambda < 0:
                logger.error("tcost_lambda must be non-negative, got %s", tcost_lambda)
                raise ValueError("tcost_lambda must be non-negative")
            self._tcost_lambda = np.full(self._I, float(tcost_lambda))
        else:
            tcost_arr = np.asarray(tcost_lambda, float).flatten()
            if tcost_arr.shape != (self._I,):
                logger.error(
                    "tcost_lambda must have shape (%d,), got %s",
                    self._I,
                    tcost_arr.shape,
                )
                raise ValueError("tcost_lambda must match number of assets")
            if np.any(tcost_arr < 0):
                logger.error("tcost_lambda entries must be non-negative")
                raise ValueError("tcost_lambda entries must be non-negative")
            self._tcost_lambda = tcost_arr

        if np.any(self._tcost_lambda > 0):
            prev = (
                np.zeros(self._I)
                if prev_weights is None
                else np.asarray(prev_weights, float).flatten()
            )
            if prev.shape != (self._I,):
                logger.error(
                    "prev_weights must have shape (%d,), got %s", self._I, prev.shape
                )
                raise ValueError("prev_weights must match number of assets")
            self._prev_weights = prev
            self._aux = self._I
        else:
            self._prev_weights = np.zeros(self._I)
            self._aux = 0

        total_vars = self._I + self._aux

        self._expected_return_row = -matrix(np.hstack((mean, np.zeros(self._aux)))).T
        P_block = np.block(
            [
                [1000 * covariance_matrix, np.zeros((self._I, self._aux))],
                [np.zeros((self._aux, self._I + self._aux))],
            ]
        )
        self._P = matrix(P_block)
        if self._aux:
            q = np.hstack((np.zeros(self._I), self._tcost_lambda))
        else:
            q = np.zeros(self._I)
        self._q = matrix(q)
        self._v = (np.ones(self._I) if v is None else v).reshape(1, -1)

        if (G is None) ^ (h is None):
            logger.error("G and h must be provided together or both None")
            raise ValueError("G and h must be provided together or both None")
        if (A is None) ^ (b is None):
            logger.error("A and b must be provided together or both None")
            raise ValueError("A and b must be provided together or both None")

        if G is not None:
            G_base = np.asarray(G, float)
            if G_base.shape[1] != self._I:
                logger.error("G must have %d columns, got %d", self._I, G_base.shape[1])
                raise ValueError("G has incorrect shape")
            G_ext = np.hstack((G_base, np.zeros((G_base.shape[0], self._aux))))
            h_base = np.asarray(h, float)
        else:
            G_ext = np.zeros((1, total_vars))
            h_base = np.array([0.0])
        # trading cost constraints
        if self._aux:
            tc_G1 = np.hstack((np.eye(self._I), -np.eye(self._I)))
            tc_h1 = self._prev_weights
            tc_G2 = np.hstack((-np.eye(self._I), -np.eye(self._I)))
            tc_h2 = -self._prev_weights
            G_ext = np.vstack((G_ext, tc_G1, tc_G2))
            h_base = np.hstack((h_base, tc_h1, tc_h2))

        self._G = sparse(matrix(G_ext))
        self._h = matrix(h_base)

        if A is not None:
            A_base = np.asarray(A, float)
            if A_base.shape[1] != self._I:
                logger.error("A must have %d columns, got %d", self._I, A_base.shape[1])
                raise ValueError("A has incorrect shape")
            A_ext = np.hstack((A_base, np.zeros((A_base.shape[0], self._aux))))
            self._A = sparse(matrix(A_ext))
            self._b = matrix(b)
        else:
            A_ext = np.hstack((self._v, np.zeros((1, self._aux))))
            self._A = sparse(matrix(A_ext))
            self._b = matrix([1.0])

        _ = self._calculate_max_expected_return(feasibility_check=True)

    def efficient_portfolio(self, return_target: Optional[float] = None) -> np.ndarray:
        if return_target is None:
            sol = solvers.qp(self._P, self._q, self._G, self._h, self._A, self._b)
            return np.asarray(sol["x"][: self._I])

        G_ext = sparse([self._G, self._expected_return_row])
        h_ext = matrix([self._h, -return_target])
        sol = solvers.qp(self._P, self._q, G_ext, h_ext, self._A, self._b)
        return np.asarray(sol["x"][: self._I])


class MeanCVaR(Optimization):
    """Mean-CVaR optimizer using linear programming and Benders decomposition."""

    def __init__(
        self,
        R: np.ndarray,
        G: Optional[np.ndarray] = None,
        h: Optional[np.ndarray] = None,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        v: Optional[np.ndarray] = None,
        p: Optional[np.ndarray] = None,
        alpha: float = 0.95,
        *,
        tcost_lambda: Union[float, Sequence[float]] = 0.0,
        prev_weights: Optional[np.ndarray] = None,
        **kwargs: dict,
    ):
        self._S, self._I = R.shape

        self._set_options(kwargs.get("options", globals()["cvar_options"]))

        if p is None:
            self._p = np.ones((1, self._S)) / self._S
        else:
            p = np.asarray(p, float).reshape(1, -1)
            if p.shape[1] != self._S:
                logger.error("p must have length S, got %d", p.shape[1])
                raise ValueError("p must have length S")
            self._p = p

        self._mean = self._p @ R
        if not (isinstance(alpha, numbers.Real) and 0 < alpha < 1):
            logger.error("alpha must be a float in (0, 1), got %s", alpha)
            raise ValueError("alpha must be a float in (0, 1)")
        self._alpha = float(alpha)

        if np.isscalar(tcost_lambda):
            if tcost_lambda < 0:
                logger.error("tcost_lambda must be non-negative, got %s", tcost_lambda)
                raise ValueError("tcost_lambda must be non-negative")
            self._tcost_lambda = np.full(self._I, float(tcost_lambda))
        else:
            tcost_arr = np.asarray(tcost_lambda, float).flatten()
            if tcost_arr.shape != (self._I,):
                logger.error(
                    "tcost_lambda must have shape (%d,), got %s",
                    self._I,
                    tcost_arr.shape,
                )
                raise ValueError("tcost_lambda must match number of assets")
            if np.any(tcost_arr < 0):
                logger.error("tcost_lambda entries must be non-negative")
                raise ValueError("tcost_lambda entries must be non-negative")
            self._tcost_lambda = tcost_arr

        if np.any(self._tcost_lambda > 0):
            prev = (
                np.zeros(self._I)
                if prev_weights is None
                else np.asarray(prev_weights, float).flatten()
            )
            if prev.shape != (self._I,):
                logger.error(
                    "prev_weights must have shape (%d,), got %s", self._I, prev.shape
                )
                raise ValueError("prev_weights must match number of assets")
            self._prev_weights = prev
            self._aux = self._I
        else:
            self._prev_weights = np.zeros(self._I)
            self._aux = 0

        total_vars = self._I + self._aux + 2

        c_vec = np.hstack(
            (
                np.zeros(self._I),
                self._tcost_lambda if self._aux else np.array([]),
                [1.0, 1.0 / (1.0 - self._alpha)],
            )
        )
        self._c = matrix(c_vec)

        self._expected_return_row = matrix(
            np.hstack((-self._mean, np.zeros((1, self._aux + 2))))
        )

        v_vec = np.ones(self._I) if v is None else np.asarray(v, float)
        if v_vec.shape != (self._I,):
            logger.error("v must have shape (I,), got %s", v_vec.shape)
            raise ValueError("v must have shape (I,)")
        self._v = np.hstack((v_vec.reshape(1, -1), np.zeros((1, self._aux + 2))))

        if (G is None) ^ (h is None):
            logger.error("G and h must be provided together or both None")
            raise ValueError("G and h must be provided together or both None")

        if G is None:
            G_base = np.zeros((0, self._I))
            h_base = np.zeros(0)
        else:
            G_base = np.asarray(G, float)
            h_base = np.asarray(h, float).flatten()
            if G_base.shape[0] != h_base.size:
                logger.error(
                    "G and h have incompatible shapes: %s vs %s",
                    G_base.shape,
                    h_base.shape,
                )
                raise ValueError("G and h have incompatible shapes")

        G_ext = np.hstack((G_base, np.zeros((G_base.shape[0], self._aux + 2))))
        z_row = np.zeros(self._I + self._aux + 2)
        z_row[-1] = -1.0
        G_full = np.vstack((G_ext, z_row))
        h_full = np.hstack((h_base, [0.0]))

        if self._aux:
            tc_G1 = np.hstack(
                (np.eye(self._I), -np.eye(self._I), np.zeros((self._I, 2)))
            )
            tc_h1 = self._prev_weights
            tc_G2 = np.hstack(
                (-np.eye(self._I), -np.eye(self._I), np.zeros((self._I, 2)))
            )
            tc_h2 = -self._prev_weights
            G_full = np.vstack((G_full, tc_G1, tc_G2))
            h_full = np.hstack((h_full, tc_h1, tc_h2))

        self._G = sparse(matrix(G_full))
        self._h = matrix(h_full)

        if (A is None) ^ (b is None):
            logger.error("A and b must be provided together or both None")
            raise ValueError("A and b must be provided together or both None")

        if A is None:
            A_ext = np.hstack((self._v, np.zeros((1, self._aux + 2))))
            self._A = sparse(matrix(A_ext))
            self._b = matrix([1.0])
        else:
            A_base = np.asarray(A, float)
            if A_base.shape[1] != self._I:
                logger.error("A must have %d columns, got %d", self._I, A_base.shape[1])
                raise ValueError("A has incorrect shape")
            A_ext = np.hstack((A_base, np.zeros((A_base.shape[0], self._aux + 2))))
            self._A = sparse(matrix(A_ext))
            self._b = matrix(b)

        _ = self._calculate_max_expected_return(feasibility_check=True)

        self._R_scalar = getattr(self, "_R_scalar", 1000)
        self._demean = getattr(self, "_demean", True)
        if self._demean:
            self._losses = -self._R_scalar * (R - self._mean)
        else:
            self._losses = -self._R_scalar * R

    def _set_options(self, options: dict) -> None:
        self._demean = options.get("demean", True)
        if not isinstance(self._demean, bool):
            logger.error("demean must be a boolean, got %s", type(self._demean))
            raise ValueError("demean must be a boolean equal to True or False.")
        self._R_scalar = options.get("R_scalar", 1000)
        if not isinstance(self._R_scalar, (int, float)) or self._R_scalar <= 0:
            logger.error("R_scalar must be positive number, got %s", self._R_scalar)
            raise ValueError("R_scalar must be a positive integer or float.")
        self._maxiter = options.get("maxiter", 500)
        if not isinstance(self._maxiter, int) or self._maxiter < 100:
            logger.error("maxiter must be integer >= 100, got %s", self._maxiter)
            raise ValueError(
                "maxiter must be a positive integer greater than or equal to 100."
            )
        self._reltol = options.get("reltol", 1e-8)
        if not 1e-8 <= self._reltol <= 1e-4:
            logger.error("reltol must be in [1e-8, 1e-4], got %s", self._reltol)
            raise ValueError("reltol must be in [1e-8, 1e-4].")
        self._abstol = options.get("abstol", 1e-8)
        if not 1e-8 <= self._abstol <= 1e-4:
            logger.error("abstol must be in [1e-8, 1e-4], got %s", self._abstol)
            raise ValueError("abstol must be in [1e-8, 1e-4].")

    def _benders_algorithm(self, G: sparse, h: matrix) -> np.ndarray:
        eta = self._p @ self._losses
        p = 1
        solution, w, F_lower, G_benders, h_benders, eta, p = self._benders_main(
            G, h, eta, p
        )
        F_star = F_lower + self._c[-1] * (w - solution[-1])
        v = 1
        while self._benders_stopping_criteria(F_star, F_lower) and v <= self._maxiter:
            solution, w, F_lower, G_benders, h_benders, eta, p = self._benders_main(
                G_benders, h_benders, eta, p
            )
            F_star = min(F_lower + self._c[-1] * (w - solution[-1]), F_star)
            v += 1
        logger.debug("Benders algorithm completed in %d iterations.", v)
        return solution

    def _benders_main(
        self,
        G_benders: sparse,
        h_benders: matrix,
        eta: np.ndarray,
        p: float,
    ) -> Tuple[np.ndarray, float, float, sparse, matrix, np.ndarray, float]:
        new_row = np.hstack((eta, np.zeros((1, self._aux)), [[-p, -1]]))
        G_benders = sparse([G_benders, matrix(new_row)])
        h_benders = matrix([h_benders, 0])
        solution = np.array(
            lp(
                c=self._c, G=G_benders, h=h_benders, A=self._A, b=self._b, solver="glpk"
            )["x"]
        )
        eta, p = self._benders_cut(solution)
        w = eta @ solution[0 : self._I] - p * solution[self._I + self._aux]
        F_lower = self._c.T @ solution
        return solution, w, F_lower, G_benders, h_benders, eta, p

    def _benders_cut(self, solution: np.ndarray) -> Tuple[np.ndarray, float]:
        K = (self._losses @ solution[0 : self._I] >= solution[-2])[:, 0]
        eta = self._p[:, K] @ self._losses[K, :]
        p = np.sum(self._p[0, K])
        return eta, p

    def _benders_stopping_criteria(self, F_star: float, F_lower: float) -> bool:
        F_lower_abs = np.abs(F_lower)
        if F_lower_abs > 1e-10:
            return (F_star - F_lower) / F_lower_abs > self._reltol
        else:
            return (F_star - F_lower) > self._abstol

    def efficient_portfolio(self, return_target: Optional[float] = None) -> np.ndarray:
        if return_target is None:
            G = copy(self._G)
            h = copy(self._h)
        else:
            G = sparse([self._G, self._expected_return_row])
            h = matrix([self._h, -return_target])
        return self._benders_algorithm(G, h)[0 : self._I]


class RobustBayes(Optimization):
    """Robust Bayesian portfolio optimisation via second-order cone programming."""

    def __init__(
        self,
        mean: np.ndarray,
        covariance_matrix: np.ndarray,
        rho: float,
        gamma: float,
        G: Optional[np.ndarray] = None,
        h: Optional[np.ndarray] = None,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
    ) -> None:
        """Initialise the robust Bayesian optimiser.

        Parameters
        ----------
        rho : float
            Robustness level controlling the confidence ellipsoid. See Meucci
            (2005) eq.(23) for guidance on calibrating ``rho``.
        gamma : float
            Mean--variance risk aversion parameter.
        """

        self._I = len(mean)
        self._mean = np.asarray(mean, float).flatten()
        self._cov = np.asarray(covariance_matrix, float)
        self._sigma_sqrt = np.linalg.cholesky(
            self._cov + 1e-12 * np.eye(self._I)
        )
        self._rho = float(rho)
        self._gamma = float(gamma)
        self._lambda2 = 0.5 * self._gamma * (1.0 + np.sqrt(2 * self._rho / self._gamma))

        if (G is None) ^ (h is None):
            raise ValueError("G and h must be provided together or both None")

        if G is None:
            G_base = np.zeros((0, self._I))
            h_base = np.zeros(0)
        else:
            G_base = np.asarray(G, float)
            h_base = np.asarray(h, float).flatten()
            if G_base.shape[0] != h_base.size:
                raise ValueError("G and h have incompatible shapes")

        self._Gl = np.hstack((G_base, np.zeros((G_base.shape[0], 1))))
        self._hl = h_base

        if (A is None) ^ (b is None):
            raise ValueError("A and b must be provided together or both None")
        if A is None:
            A_ext = np.hstack((np.ones(self._I).reshape(1, -1), np.zeros((1, 1))))
            self._A = matrix(A_ext)
            self._b = matrix([1.0])
        else:
            A_base = np.asarray(A, float)
            if A_base.shape[1] != self._I:
                raise ValueError("A has incorrect shape")
            A_ext = np.hstack((A_base, np.zeros((A_base.shape[0], 1))))
            self._A = matrix(A_ext)
            self._b = matrix(b)

        # store linear constraints for compatibility with base methods
        self._G = matrix(self._Gl)
        self._h = matrix(self._hl)

        P_block = np.zeros((self._I + 1, self._I + 1))
        P_block[: self._I, : self._I] = 2 * self._lambda2 * self._cov
        self._P = matrix(P_block)
        self._q = matrix(np.hstack((-self._mean, np.sqrt(self._rho))))
        self._expected_return_row = matrix(np.hstack((-self._mean, 0.0))).T

        # verify feasibility and store attributes for parent utilities
        _ = self._calculate_max_expected_return(feasibility_check=True)

    def _robust_risk(self, w: np.ndarray) -> float:
        """Returns λ₂·wᵀΣ̂w  +  √ρ · ‖Σ̂½ w‖₂."""
        quad = self._lambda2 * float(w @ self._cov @ w)
        soc = np.sqrt(self._rho) * np.linalg.norm(self._sigma_sqrt @ w)
        return quad + soc

    def _cone_blocks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (G_soc, h_soc) implementing ‖Σ̂½ w‖₂ ≤ t."""
        top = np.hstack([np.zeros((1, self._I)), -np.ones((1, 1))])
        bot = np.hstack([self._sigma_sqrt, np.zeros((self._I, 1))])
        G_soc = np.vstack([top, bot])
        h_soc = np.zeros(self._I + 1)
        return G_soc, h_soc

    def _solve(self, G: matrix, h: matrix) -> np.ndarray:
        total_rows = G.size[0]
        dims = {"l": total_rows - (self._I + 1), "q": [self._I + 1], "s": []}
        sol = solvers.coneqp(self._P, self._q, G, h, dims=dims, A=self._A, b=self._b)
        if sol["status"] != "optimal":
            raise ValueError("SOCP did not converge")
        return np.asarray(sol["x"][: self._I])

    def _calculate_max_expected_return(self, feasibility_check: bool = False) -> float:
        """Maximise μ̂ᵀ w  subject to cone & linear constraints."""
        G_soc, h_soc = self._cone_blocks()
        G_tot = sparse(matrix(np.vstack([self._Gl, G_soc])))
        h_tot = matrix(np.hstack([self._hl, h_soc]))
        c = np.zeros(self._I + 1) if feasibility_check else np.hstack([-self._mean, 0.0])
        dims = {"l": G_tot.size[0] - (self._I + 1), "q": [self._I + 1], "s": []}
        sol = solvers.coneqp(
            self._P * 0,
            matrix(c),
            G_tot,
            h_tot,
            dims=dims,
            A=self._A,
            b=self._b,
        )
        if sol["status"] != "optimal":
            if feasibility_check:
                raise ValueError("Constraints are infeasible.")
            raise ValueError("Expected return is unbounded under cone constraints.")
        return -float(sol["primal objective"])

    def efficient_portfolio(self, return_target: Optional[float] = None) -> np.ndarray:
        Gl = self._Gl
        hl = self._hl
        if return_target is not None:
            Gl = np.vstack((Gl, -np.hstack((self._mean, 0.0))))
            hl = np.hstack((hl, -return_target))

        soc_block = np.vstack(
            (
                np.hstack((np.zeros((1, self._I)), -np.ones((1, 1)))),
                np.hstack((self._sigma_sqrt, np.zeros((self._I, 1)))),
            )
        )
        G = matrix(np.vstack((Gl, soc_block)))
        h = matrix(np.hstack((hl, np.zeros(self._I + 1))))
        return self._solve(G, h)

    @staticmethod
    def gamma_grid(gmin: float = 0.5, gmax: float = 20.0, n: int = 15) -> np.ndarray:
        """Log-spaced grid – tunable."""
        return np.geomspace(gmin, gmax, n)

    @staticmethod
    def frontier_by_gamma(
        mean,
        cov,
        rho,
        gamma_vec,
        G=None,
        h=None,
        A=None,
        b=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (weights matrix, worst-case returns) for a list of γ."""
        W = []
        mu_wc = []
        for g in gamma_vec:
            rb = RobustBayes(mean, cov, rho, g, G, h, A, b)
            w = rb.efficient_portfolio().flatten()
            W.append(w)
            mu_wc.append(mean @ w - rb._robust_risk(w))
        return np.column_stack(W), np.array(mu_wc)

    @staticmethod
    def frontier_by_risk(
        mean,
        cov,
        rho,
        gamma,
        R_vec,
        G=None,
        h=None,
        A=None,
        b=None,
    ):
        """Fix γ, maximise μ̂ subject to robust risk ≤ R̄."""
        weights, mu_wc = [], []
        base = RobustBayes(mean, cov, rho, gamma, G, h, A, b)
        G_soc, h_soc = base._cone_blocks()

        for R in R_vec:
            G_lin = copy(base._Gl)
            h_lin = copy(base._hl)

            quad_G = np.hstack([2 * base._lambda2 * base._cov, np.zeros((base._I, 1))])
            Gtot = np.vstack([G_lin, quad_G, G_soc])
            htot = np.hstack([h_lin, np.full(base._I, R), h_soc])

            dims = {"l": Gtot.shape[0] - (base._I + 1), "q": [base._I + 1], "s": []}

            sol = solvers.coneqp(
                base._P * 0,
                matrix(np.hstack([-base._mean, 0.0])),
                sparse(matrix(Gtot)),
                matrix(htot),
                dims=dims,
                A=base._A,
                b=base._b,
            )
            w = np.asarray(sol["x"][: base._I]).flatten()
            weights.append(w)
            mu_wc.append(base._mean @ w)
        return np.column_stack(weights), np.array(mu_wc)

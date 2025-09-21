"""Utilities for combining exposures from multiple portfolios or frontiers."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

ArrayLike = Union[np.ndarray, pd.DataFrame]


def _to_2d_array(sample_portfolios: ArrayLike) -> np.ndarray:
    arr = np.asarray(sample_portfolios, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Sample portfolios must be a 2D array of shape (assets, portfolios).")
    return arr


def average_exposures(sample_portfolios: ArrayLike, weights: Optional[Sequence[float]] = None) -> np.ndarray:
    exposures = _to_2d_array(sample_portfolios)
    num_samples = exposures.shape[1]
    if weights is None:
        weights_vector = np.full(num_samples, 1.0 / num_samples)
    else:
        weights_vector = np.asarray(weights, dtype=float)
        if weights_vector.shape != (num_samples,):
            raise ValueError("`weights` must have length equal to the number of sample portfolios.")
        weight_sum = weights_vector.sum()
        if weight_sum <= 0 or not np.isfinite(weight_sum):
            raise ValueError("`weights` must sum to a positive finite value.")
        weights_vector = weights_vector / weight_sum
    return exposures @ weights_vector


@contextmanager
def _temporary_solver_options(overrides: Optional[dict]):
    previous = solvers.options.copy()
    if overrides:
        solvers.options.update(overrides)
    try:
        yield
    finally:
        solvers.options.clear()
        solvers.options.update(previous)


def exposure_stacking(
    sample_portfolios: ArrayLike,
    L: int,
    *,
    solver_options: Optional[dict] = None,
) -> np.ndarray:
    exposures = _to_2d_array(sample_portfolios)
    _, num_samples = exposures.shape
    if L <= 0:
        raise ValueError("`L` must be a positive integer.")
    if L > num_samples:
        raise ValueError("`L` cannot exceed the number of sample portfolios.")

    partition_size = num_samples // L
    indices = np.arange(num_samples)
    partitions: List[np.ndarray] = []
    for part in range(L - 1):
        start = part * partition_size
        end = (part + 1) * partition_size
        partitions.append(indices[start:end])
    partitions.append(indices[(L - 1) * partition_size :])

    matrix_exposures = exposures.T
    gram = np.zeros((num_samples, num_samples))
    linear = np.zeros(num_samples)

    for subset in partitions:
        if subset.size == 0:
            continue
        masked = matrix_exposures.copy()
        masked[subset, :] = 0.0
        gram += masked @ masked.T
        summed = exposures[:, subset].sum(axis=1)
        linear += (masked @ summed) / subset.size

    qp_P = matrix(2.0 * gram)
    qp_q = matrix(-2.0 * linear.reshape(-1, 1))
    qp_A = matrix(np.ones((1, num_samples)))
    qp_b = matrix(np.array([[1.0]]))
    qp_G = matrix(-np.identity(num_samples))
    qp_h = matrix(np.zeros((num_samples, 1)))

    options = {"show_progress": False}
    if solver_options:
        options.update(solver_options)

    with _temporary_solver_options(options):
        solution = solvers.qp(qp_P, qp_q, qp_G, qp_h, qp_A, qp_b)

    if solution.get("status") != "optimal":
        raise RuntimeError(f"Exposure stacking QP failed (status={solution.get('status')}).")

    weights = np.squeeze(np.array(solution["x"]))
    return exposures @ weights


def _stack_frontiers(
    frontiers: Sequence[object],
    selections: Optional[Sequence[Optional[Iterable[int]]]] = None,
) -> Tuple[np.ndarray, Optional[List[str]]]:
    if not frontiers:
        raise ValueError("`frontiers` must contain at least one frontier.")

    if selections is None:
        selections = [None] * len(frontiers)
    if len(selections) != len(frontiers):
        raise ValueError("`selections` must match the number of frontiers.")

    stacked: List[np.ndarray] = []
    asset_names: Optional[List[str]] = None
    reference_dim: Optional[int] = None

    for frontier, selection in zip(frontiers, selections):
        if not hasattr(frontier, "weights"):
            raise TypeError("Frontier-like objects must expose a `weights` attribute.")
        weights = np.asarray(frontier.weights, dtype=float)
        if weights.ndim != 2:
            raise ValueError("Frontier `weights` must be a 2D array.")
        if selection is not None:
            selection_indices = np.array(list(selection), dtype=int)
            weights = weights[:, selection_indices]
        stacked.append(weights)

        current_names = list(getattr(frontier, "asset_names", []) or [])
        if asset_names is None:
            asset_names = current_names if current_names else None
            reference_dim = weights.shape[0]
        else:
            if current_names:
                if current_names != asset_names:
                    raise ValueError("All frontiers must share identical asset ordering.")
            elif reference_dim is not None and weights.shape[0] != reference_dim:
                raise ValueError("Frontiers without names must have matching asset counts.")

    combined = np.hstack(stacked)
    return combined, asset_names


def average_frontiers(
    frontiers: Sequence[object],
    selections: Optional[Sequence[Optional[Iterable[int]]]] = None,
    *,
    ensemble_weights: Optional[Sequence[float]] = None,
) -> pd.Series:
    samples, names = _stack_frontiers(frontiers, selections)
    averaged = average_exposures(samples, weights=ensemble_weights)
    if names:
        return pd.Series(averaged, index=names, name="Average Ensemble")
    return pd.Series(averaged, name="Average Ensemble")


def exposure_stack_frontiers(
    frontiers: Sequence[object],
    L: int,
    selections: Optional[Sequence[Optional[Iterable[int]]]] = None,
    *,
    solver_options: Optional[dict] = None,
) -> pd.Series:
    samples, names = _stack_frontiers(frontiers, selections)
    stacked = exposure_stacking(samples, L=L, solver_options=solver_options)
    if names:
        return pd.Series(stacked, index=names, name=f"Exposure Stacking (L={L})")
    return pd.Series(stacked, name=f"Exposure Stacking (L={L})")

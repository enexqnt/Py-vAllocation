"""
Helpers for working with weight matrices/series across the library.

The routines in this module centralise the logic that used to live inside
``pyvallocation.ensembles`` so other modules (frontiers, stress utilities,
discrete allocation, etc.) can reuse the exact same conversions and label
handling.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]

__all__ = [
    "ArrayLike",
    "ensure_samples_matrix",
    "wrap_exposure_vector",
    "normalize_weights",
]


def ensure_samples_matrix(
    sample_portfolios: ArrayLike,
    *,
    allow_empty: bool = False,
) -> Tuple[np.ndarray, Optional[List[str]], Optional[List[str]]]:
    """
    Convert sample portfolios into a ``(n_assets, n_samples)`` float array.

    Parameters
    ----------
    sample_portfolios
        Array/Series/DataFrame where columns correspond to portfolios.
    allow_empty
        When ``True`` the helper permits zero columns (useful for wiring tests).

    Returns
    -------
    tuple
        ``(matrix, asset_names, sample_names)``.
    """
    asset_names: Optional[List[str]] = None
    sample_names: Optional[List[str]] = None

    if isinstance(sample_portfolios, pd.DataFrame):
        asset_names = list(sample_portfolios.index)
        sample_names = list(sample_portfolios.columns)
        arr = sample_portfolios.to_numpy(dtype=float)
    elif isinstance(sample_portfolios, pd.Series):
        asset_names = list(sample_portfolios.index)
        sample_names = [sample_portfolios.name or "portfolio_0"]
        arr = sample_portfolios.to_numpy(dtype=float).reshape(-1, 1)
    else:
        arr = np.asarray(sample_portfolios, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise ValueError("Sample portfolios must broadcast to a 2D array (assets x portfolios).")
    if not allow_empty and arr.shape[1] == 0:
        raise ValueError("Sample matrix must contain at least one portfolio.")
    return arr, asset_names, sample_names


def wrap_exposure_vector(
    vector: Union[np.ndarray, Sequence[float]],
    asset_names: Optional[List[str]],
    *,
    label: Optional[str],
) -> Union[np.ndarray, pd.Series]:
    """Return a pandas Series when asset names are provided, else fall back to ndarray."""
    array = np.asarray(vector, dtype=float).reshape(-1)
    if asset_names is None:
        return array
    return pd.Series(array, index=asset_names, name=label)


def normalize_weights(
    weights: Optional[Union[Sequence[float], pd.Series]],
    num_samples: int,
    sample_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Normalise a weight vector while supporting pandas-labelled inputs.

    When ``weights`` is ``None`` the helper returns a uniform allocation.
    """
    if num_samples == 0:
        raise ValueError("No sample portfolios provided.")

    if weights is None:
        return np.full(num_samples, 1.0 / num_samples, dtype=float)

    if isinstance(weights, pd.Series):
        if sample_names is None:
            weights_vector = weights.to_numpy(dtype=float)
        else:
            reindexed = weights.reindex(sample_names)
            if reindexed.isna().any():
                missing = [name for name, val in zip(sample_names, reindexed) if pd.isna(val)]
                raise ValueError(f"Missing weights for samples: {missing}.")
            weights_vector = reindexed.to_numpy(dtype=float)
    else:
        weights_vector = np.asarray(weights, dtype=float)

    weights_vector = weights_vector.reshape(-1)
    if weights_vector.shape[0] != num_samples:
        raise ValueError("`weights` must have length equal to the number of sample portfolios.")
    total = float(np.sum(weights_vector))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("`weights` must sum to a positive finite value.")
    return weights_vector / total

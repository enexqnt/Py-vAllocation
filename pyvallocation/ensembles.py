"""
Ensemble utilities for blending portfolio weights.

The helpers in this module implement two complementary recipes that operate on
sample portfolios organised column-wise (``n_assets × n_samples``):

* :func:`average_exposures` – arithmetic averaging (optionally weighted) across
  a panel of sample portfolios.
* :func:`exposure_stacking` – the exposure-stacking quadratic programme first
  introduced by Vorobets :cite:p:`vorobets2024derivatives` and implemented in
  the GPL-3 licensed `fortitudo.tech <https://github.com/fortitudo-tech/fortitudo.tech>`_
  repository. The routine dampens idiosyncratic exposures while preserving the
  mean profile of the sample set.

Both routines can operate directly on NumPy/pandas objects and are wrapped by
:func:`average_frontiers` / :func:`exposure_stack_frontiers` to accept
:class:`~pyvallocation.portfolioapi.PortfolioFrontier` instances, ensuring
consistent integration with the rest of the library.

All functions preserve pandas indices when they are supplied, so users can move
between NumPy and pandas inputs without reshaping or re-labelling portfolios.

Example
-------

>>> from pyvallocation.ensembles import average_frontiers, exposure_stack_frontiers
>>> frontier_a, frontier_b = ...  # PortfolioFrontier instances
>>> average_frontiers([frontier_a, frontier_b])
AAA    0.48
BBB    0.52
Name: Average Ensemble, dtype: float64
>>> exposure_stack_frontiers([frontier_a, frontier_b], L=3)
AAA    0.44
BBB    0.56
Name: Exposure Stacking (L=3), dtype: float64
"""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    TYPE_CHECKING,
)

import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .portfolioapi import PortfolioFrontier

__all__ = [
    "average_exposures",
    "exposure_stacking",
    "average_frontiers",
    "exposure_stack_frontiers",
    "EnsembleSpec",
    "EnsembleResult",
    "assemble_portfolio_ensemble",
    "make_portfolio_spec",
]


def _to_2d_array(
    sample_portfolios: ArrayLike,
) -> Tuple[np.ndarray, Optional[List[str]], Optional[List[str]]]:
    """
    Convert sample portfolios into a ``(n_assets, n_samples)`` float array.

    Returns both asset names (row labels) and sample identifiers (column labels)
    when available to keep downstream I/O consistent.
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
        raise ValueError("Sample portfolios must broadcast to a 2D array (assets × portfolios).")
    return arr, asset_names, sample_names


def _wrap_exposure(vector: np.ndarray, asset_names: Optional[List[str]], *, label: Optional[str]) -> ArrayLike:
    """Return a pandas Series when asset names are provided, else fall back to ndarray."""
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if asset_names is None:
        return vector
    return pd.Series(vector, index=asset_names, name=label)


def _prepare_weights(
    weights: Optional[Sequence[float] | pd.Series],
    num_samples: int,
    sample_names: Optional[List[str]],
) -> np.ndarray:
    """Normalise weight vector while supporting pandas-labelled inputs."""
    if weights is None:
        if num_samples == 0:
            raise ValueError("No sample portfolios provided.")
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


def average_exposures(
    sample_portfolios: ArrayLike,
    weights: Optional[Union[Sequence[float], pd.Series]] = None,
) -> ArrayLike:
    """
    Compute the (possibly weighted) average exposure across multiple portfolios.

    The routine accepts any collection of sample weights arranged column-wise.
    When ``weights`` is omitted the average is uniform; otherwise ``weights``
    must supply one non-negative scalar per sample and is normalised to unity.

    Parameters
    ----------
    sample_portfolios :
        Array-like object whose columns represent sample portfolios.
    weights :
        Optional sequence or pandas Series of length ``n_samples`` providing
        relative importance for each column. When a Series is supplied its index
        is aligned to the sample column labels. The entries are automatically
        rescaled so that they sum to one.

    Returns
    -------
    ndarray or pandas.Series
        Averaged exposure vector with length equal to ``n_assets``. A pandas
        Series is returned when asset names are available on the input.

    Examples
    --------
    >>> import numpy as np
    >>> samples = np.array([[0.6, 0.3], [0.4, 0.7]])
    >>> average_exposures(samples)
    array([0.45, 0.55])
    >>> average_exposures(samples, weights=[1.0, 3.0])
    array([0.375, 0.625])
    """
    exposures, asset_names, sample_names = _to_2d_array(sample_portfolios)
    num_samples = exposures.shape[1]
    weights_vector = _prepare_weights(weights, num_samples, sample_names)
    averaged = exposures @ weights_vector
    return _wrap_exposure(averaged, asset_names, label="Average Exposure")


@contextmanager
def _temporary_solver_options(overrides: Optional[dict]):
    """Context manager that applies temporary CVXOPT solver options."""
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
) -> ArrayLike:
    """
    Compute exposure stacking weights following Vorobets :cite:p:`vorobets2024derivatives`.

    The algorithm partitions the set of sample portfolios into ``L`` buckets and
    solves a quadratic programme that minimises the sum of cross-validated
    residuals. Intuitively, the resulting allocation penalises weights that are
    idiosyncratic to any particular subset of samples, favouring stable signals.

    Parameters
    ----------
    sample_portfolios :
        Panel of sample portfolios organised column-wise.
    L :
        Number of cross-validation folds. Must satisfy ``1 <= L <= n_samples``.
    solver_options :
        Optional dictionary of CVXOPT solver overrides (e.g., ``{'maxiters': 100}``).

    Returns
    -------
    ndarray or pandas.Series
        Exposure-stacked portfolio of length ``n_assets``. A Series is returned
        when asset names are provided on the input.

    Notes
    -----
    This implementation adapts the open-source reference code from the
    `fortitudo.tech <https://github.com/fortitudo-tech/fortitudo.tech>`_ project
    (GPL-3.0) that accompanies Vorobets' original publication.

    Raises
    ------
    RuntimeError
        If the underlying quadratic programme does not terminate with status
        ``'optimal'``.
    """
    exposures, asset_names, _ = _to_2d_array(sample_portfolios)
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
        linear += (masked @ summed) / max(subset.size, 1)

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
    stacked = exposures @ weights
    return _wrap_exposure(stacked, asset_names, label=f"Exposure Stacking (L={L})")


def _stack_frontiers(
    frontiers: Sequence[object],
    selections: Optional[Sequence[Optional[Iterable[int]]]] = None,
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Extract portfolio samples from :class:`~pyvallocation.portfolioapi.PortfolioFrontier`.

    Parameters
    ----------
    frontiers :
        Iterable of frontier-like objects exposing a ``weights`` attribute with
        shape ``(n_assets, n_portfolios)``.
    selections :
        Optional sequence selecting a subset of columns per frontier. ``None``
        implies all portfolios.

    Returns
    -------
    tuple
        ``(samples, asset_names)`` where ``samples`` is a stacked 2-D array and
        ``asset_names`` propagates the first non-empty asset-name list, if any.
    """
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


def _series_from_vector(weights: np.ndarray, names: Optional[List[str]], label: str) -> pd.Series:
    """Helper returning a labelled Series with optional asset names."""
    if names:
        return pd.Series(weights, index=names, name=label)
    return pd.Series(weights, name=label)


def average_frontiers(
    frontiers: Sequence[object],
    selections: Optional[Sequence[Optional[Iterable[int]]]] = None,
    *,
    ensemble_weights: Optional[Sequence[float]] = None,
) -> pd.Series:
    """
    Average selected portfolios across multiple frontiers (column-wise).

    Parameters
    ----------
    frontiers :
        Sequence of frontier-like objects (typically
        :class:`~pyvallocation.portfolioapi.PortfolioFrontier` instances).
    selections :
        Optional per-frontier iterable selecting column indices. When omitted the
        entire frontier is used.
    ensemble_weights :
        Optional weights applied to the stacked sample matrix before averaging.
        Must have length equal to the total number of selected portfolios.

    Returns
    -------
    pandas.Series
        Averaged exposure vector with propagated asset labels when available.
    """
    samples, names = _stack_frontiers(frontiers, selections)
    averaged = average_exposures(samples, weights=ensemble_weights)
    return _series_from_vector(averaged, names, "Average Ensemble")


def exposure_stack_frontiers(
    frontiers: Sequence[object],
    L: int,
    selections: Optional[Sequence[Optional[Iterable[int]]]] = None,
    *,
    solver_options: Optional[dict] = None,
) -> pd.Series:
    """
    Apply exposure stacking across one or more frontiers.

    Parameters
    ----------
    frontiers :
        Sequence of frontier-like objects contributing sample portfolios.
    L :
        Number of stacking folds (as in :func:`exposure_stacking`).
    selections :
        Optional iterable specifying which column indices to draw from each
        frontier.
    solver_options :
        Optional dictionary of CVXOPT solver overrides.

    Returns
    -------
    pandas.Series
        Exposure-stacked weights with propagated asset labels.

    Notes
    -----
    The total number of selected portfolios must be at least ``L``. When
    ``selections`` is omitted the full frontier matrices are used, matching the
    layout of :attr:`~pyvallocation.portfolioapi.PortfolioFrontier.weights`.
    """
    samples, names = _stack_frontiers(frontiers, selections)
    stacked = exposure_stacking(samples, L=L, solver_options=solver_options)
    return _series_from_vector(stacked, names, f"Exposure Stacking (L={L})")


# --------------------------------------------------------------------------- #
# High-level ensemble orchestration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EnsembleSpec:
    """Descriptor for a single portfolio specification participating in an ensemble."""

    name: str
    frontier_factory: Callable[[], "PortfolioFrontier"]
    selector: Callable[["PortfolioFrontier"], Union[pd.Series, Tuple[Any, ...], np.ndarray]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    frontier_selection: Optional[Sequence[int]] = None


@dataclass
class EnsembleResult:
    """Container returned by :func:`assemble_portfolio_ensemble`."""

    frontiers: Dict[str, "PortfolioFrontier"]
    selections: pd.DataFrame
    ensembles: Dict[str, pd.Series]
    metadata: Dict[str, Dict[str, Any]]

    def get(self, name: str, default: Optional[pd.Series] = None) -> Optional[pd.Series]:
        return self.ensembles.get(name, default)

    @property
    def average(self) -> Optional[pd.Series]:
        return self.ensembles.get("average")

    @property
    def stacked(self) -> Optional[pd.Series]:
        return self.ensembles.get("stack")


def assemble_portfolio_ensemble(
    specs: Sequence[EnsembleSpec],
    *,
    ensemble: Union[str, Sequence[str], None] = "stack",
    combine: str = "selected",
    stack_folds: Optional[int] = None,
    ensemble_weights: Optional[Sequence[float]] = None,
    stack_kwargs: Optional[dict] = None,
) -> EnsembleResult:
    """
    Build multiple frontiers and collapse them into ensemble portfolios with a
    single call.

    Parameters
    ----------
    specs:
        Sequence of :class:`EnsembleSpec` instances describing how to generate and
        summarise each frontier.
    ensemble:
        ``"average"``, ``"stack"``, a sequence of the two, or ``None``. Defaults
        to ``"stack"`` for a stacked blend. Use ``("average", "stack")`` to obtain
        both.
    combine:
        ``"selected"`` (default) averages / stacks the representative portfolios
        extracted via each spec's selector. ``"frontier"`` operates directly on
        the underlying frontiers using :func:`average_frontiers` and
        :func:`exposure_stack_frontiers`.
    stack_folds:
        Number of folds for stacking. When omitted the helper picks
        ``min(3, number_of_portfolios)``.
    ensemble_weights:
        Optional weights applied during averaging (either over selected portfolios
        or the full frontier combination).
    stack_kwargs:
        Optional dictionary forwarded to the stacking solver
        (``solver_options`` argument).

    Returns
    -------
    EnsembleResult
        Rich result object containing the generated frontiers, the representative
        portfolios, and any requested ensemble allocations.
    """
    if not specs:
        raise ValueError("At least one EnsembleSpec must be provided.")

    frontiers: Dict[str, "PortfolioFrontier"] = {}
    selections: List[pd.Series] = []
    metadata: Dict[str, Dict[str, Any]] = {}

    for spec in specs:
        frontier = spec.frontier_factory()
        frontiers[spec.name] = frontier
        metadata[spec.name] = dict(spec.metadata or {})
        selected = _coerce_weights_series(frontier, spec.selector(frontier), spec.name)
        selections.append(selected)

    selection_df = pd.concat(selections, axis=1)
    selection_df.columns = [spec.name for spec in specs]

    ensemble_names = _normalise_ensemble_argument(ensemble)
    combine_mode = combine.lower()
    if combine_mode not in {"selected", "frontier"}:
        raise ValueError("`combine` must be either 'selected' or 'frontier'.")

    stack_kwargs = stack_kwargs or {}
    ensembles: Dict[str, pd.Series] = {}

    if ensemble_names:
        if combine_mode == "selected":
            total_portfolios = selection_df.shape[1]
            folds = _determine_stack_folds(stack_folds, total_portfolios)
            for entry in ensemble_names:
                if entry == "average":
                    ensembles["average"] = average_exposures(selection_df, weights=ensemble_weights)
                elif entry == "stack":
                    ensembles["stack"] = exposure_stacking(
                        selection_df,
                        L=folds,
                        solver_options=stack_kwargs or None,
                    )
                else:
                    raise ValueError(f"Unsupported ensemble option '{entry}'.")
        else:  # combine on full frontier matrices
            frontier_list = [frontiers[spec.name] for spec in specs]
            selections_norm = [
                None if spec.frontier_selection is None else list(spec.frontier_selection)
                for spec in specs
            ]
            total_portfolios = sum(
                _frontier_selection_size(frontier, selection)
                for frontier, selection in zip(frontier_list, selections_norm)
            )
            folds = _determine_stack_folds(stack_folds, total_portfolios)
            for entry in ensemble_names:
                if entry == "average":
                    ensembles["average"] = average_frontiers(
                        frontier_list,
                        selections=selections_norm,
                        ensemble_weights=ensemble_weights,
                    )
                elif entry == "stack":
                    ensembles["stack"] = exposure_stack_frontiers(
                        frontier_list,
                        L=folds,
                        selections=selections_norm,
                        solver_options=stack_kwargs or None,
                    )
                else:
                    raise ValueError(f"Unsupported ensemble option '{entry}'.")

    return EnsembleResult(
        frontiers=frontiers,
        selections=selection_df,
        ensembles=ensembles,
        metadata=metadata,
    )


def make_portfolio_spec(
    name: str,
    *,
    returns: Optional[pd.DataFrame] = None,
    probabilities: Optional[Union[pd.Series, Sequence[float], np.ndarray]] = None,
    preprocess: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    projection: Optional[Dict[str, Any]] = None,
    distribution: Optional["AssetsDistribution"] = None,
    distribution_factory: Optional[Callable[[], "AssetsDistribution"]] = None,
    use_scenarios: bool = False,
    mean_estimator: str = "sample",
    cov_estimator: str = "sample",
    mean_kwargs: Optional[Dict[str, Any]] = None,
    cov_kwargs: Optional[Dict[str, Any]] = None,
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
) -> EnsembleSpec:
    """
    Convenience constructor for :class:`EnsembleSpec` covering common workflows.

    Parameters
    ----------
    name:
        Spec identifier.
    returns:
        Historical scenario matrix (rows = scenarios, columns = assets).
    probabilities:
        Optional scenario weights aligned with ``returns``.
    preprocess:
        Optional callable applied to ``returns`` before estimation (e.g., convert
        compounded to simple returns).
    projection:
        Optional dictionary with projection settings. Recognised keys:

        * ``annualization_factor`` – passed to :func:`project_mean_covariance`.
        * ``log_to_simple`` / ``to_simple`` (bool) – apply :func:`log2simple`.
        * ``transform`` – callable ``transform(mu, Sigma) -> (mu, Sigma)``.
    distribution / distribution_factory:
        Supply an :class:`AssetsDistribution` directly (or a factory returning
        one) instead of estimating from data.
    use_scenarios:
        When ``True`` the distribution is built from scenarios rather than
        estimated moments.
    mean_estimator / cov_estimator:
        Names understood by :func:`pyvallocation.moments.estimate_moments`.
    mean_kwargs / cov_kwargs:
        Additional keyword arguments forwarded to the estimators.
    optimiser:
        Optimiser key (``"mean_variance"``, ``"cvar"``, ``"rrp"``, ``"robust"``)
        or a callable building a :class:`PortfolioFrontier` from a
        :class:`~pyvallocation.portfolioapi.PortfolioWrapper`.
    optimiser_kwargs:
        Keyword arguments for the optimiser. If it contains ``constraints`` they
        are passed to :meth:`PortfolioWrapper.set_constraints`.
    selector:
        How to extract the representative portfolio. Accepts strings
        (``"tangency"``, ``"min_risk"``, ``"max_return"``, ``"risk_target"``,
        ``"column"``) or a callable.
    selector_kwargs:
        Extra parameters for the selector (e.g., ``risk_free_rate`` for the
        tangency portfolio).
    frontier_selection:
        Column subset used when combining over entire frontiers.
    metadata:
        Optional dictionary persisted in the returned :class:`EnsembleResult`.
    """

    from .portfolioapi import AssetsDistribution, PortfolioWrapper

    mean_kwargs = dict(mean_kwargs or {})
    cov_kwargs = dict(cov_kwargs or {})
    optimiser_kwargs = dict(optimiser_kwargs or {})
    selector_kwargs = dict(selector_kwargs or {})

    frontier_selection_tuple = (
        None if frontier_selection is None else tuple(int(i) for i in frontier_selection)
    )

    def frontier_factory() -> "PortfolioFrontier":
        dist = _build_distribution(
            returns=returns,
            probabilities=probabilities,
            preprocess=preprocess,
            projection=projection,
            distribution=distribution,
            distribution_factory=distribution_factory,
            use_scenarios=use_scenarios,
            mean_estimator=mean_estimator,
            cov_estimator=cov_estimator,
            mean_kwargs=mean_kwargs,
            cov_kwargs=cov_kwargs,
        )
        wrapper = PortfolioWrapper(dist)
        local_kwargs = dict(optimiser_kwargs)
        constraints = local_kwargs.pop("constraints", None)
        if constraints is not None:
            wrapper.set_constraints(constraints)
        optimiser_callable = _resolve_optimiser(optimiser)
        return optimiser_callable(wrapper, **local_kwargs)

    selector_fn = _build_selector(selector, selector_kwargs, name)

    spec_metadata = dict(metadata or {})
    spec_metadata.setdefault("optimiser", optimiser if isinstance(optimiser, str) else optimiser.__name__)
    spec_metadata.setdefault("mean_estimator", mean_estimator)
    spec_metadata.setdefault("cov_estimator", cov_estimator)
    spec_metadata.setdefault("use_scenarios", use_scenarios)

    return EnsembleSpec(
        name=name,
        frontier_factory=frontier_factory,
        selector=selector_fn,
        metadata=spec_metadata,
        frontier_selection=frontier_selection_tuple,
    )


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _coerce_weights_series(
    frontier: "PortfolioFrontier",
    weights: Union[pd.Series, Tuple[Any, ...], np.ndarray, Sequence[float]],
    label: str,
) -> pd.Series:
    if isinstance(weights, tuple):
        weights = weights[0]
    if isinstance(weights, pd.Series):
        return weights.rename(label)
    arr = np.asarray(weights, dtype=float).reshape(-1)
    asset_names = getattr(frontier, "asset_names", None)
    if asset_names:
        return pd.Series(arr, index=asset_names, name=label)
    return pd.Series(arr, name=label)


def _normalise_ensemble_argument(
    ensemble: Union[str, Sequence[str], None]
) -> Tuple[str, ...]:
    if ensemble is None:
        return tuple()
    if isinstance(ensemble, str):
        ensemble = ensemble.strip().lower()
        if ensemble in {"", "none"}:
            return tuple()
        return (ensemble,)
    return tuple(item.strip().lower() for item in ensemble if item and item.strip())


def _determine_stack_folds(stack_folds: Optional[int], total: int) -> int:
    if total <= 0:
        raise ValueError("No portfolios available for stacking.")
    if stack_folds is None:
        stack_folds = min(3, total)
    stack_folds = int(max(1, min(stack_folds, total)))
    return stack_folds


def _frontier_selection_size(frontier: "PortfolioFrontier", selection: Optional[Sequence[int]]) -> int:
    if selection is None:
        return frontier.weights.shape[1]
    return len(list(selection))


def _resolve_optimiser(
    optimiser: Union[
        str,
        Callable[["PortfolioWrapper"], "PortfolioFrontier"],
        Callable[..., "PortfolioFrontier"],
    ]
) -> Callable[..., "PortfolioFrontier"]:
    if callable(optimiser):
        return optimiser
    key = optimiser.lower()

    def mean_variance(wrapper: "PortfolioWrapper", **kwargs):
        return wrapper.mean_variance_frontier(**kwargs)

    def mean_cvar(wrapper: "PortfolioWrapper", **kwargs):
        return wrapper.mean_cvar_frontier(**kwargs)

    def relaxed_rp(wrapper: "PortfolioWrapper", **kwargs):
        return wrapper.relaxed_risk_parity_frontier(**kwargs)

    def robust(wrapper: "PortfolioWrapper", **kwargs):
        return wrapper.robust_frontier(**kwargs)

    mapping = {
        "mean_variance": mean_variance,
        "mv": mean_variance,
        "mean-cvar": mean_cvar,
        "cvar": mean_cvar,
        "rrp": relaxed_rp,
        "relaxed_risk_parity": relaxed_rp,
        "robust": robust,
    }
    if key not in mapping:
        raise ValueError(f"Unknown optimiser '{optimiser}'.")
    return mapping[key]


def _build_selector(
    selector: Union[str, Callable[["PortfolioFrontier"], Any]],
    selector_kwargs: Dict[str, Any],
    label: str,
) -> Callable[["PortfolioFrontier"], pd.Series]:
    selector_kwargs = dict(selector_kwargs)

    if callable(selector):
        def _callable(frontier: "PortfolioFrontier") -> pd.Series:
            return selector(frontier, **selector_kwargs)

        return _callable

    key = selector.lower()

    if key in {"tangency", "max_sharpe", "sharpe"}:
        risk_free = selector_kwargs.pop("risk_free_rate", 0.0)

        def _tangency(frontier: "PortfolioFrontier") -> pd.Series:
            weights, *_ = frontier.get_tangency_portfolio(risk_free_rate=risk_free)
            return weights.rename(label)

        return _tangency

    if key in {"min_risk", "minimum_risk"}:

        def _min(frontier: "PortfolioFrontier") -> pd.Series:
            weights, *_ = frontier.get_min_risk_portfolio()
            return weights.rename(label)

        return _min

    if key in {"max_return", "maximum_return"}:

        def _max(frontier: "PortfolioFrontier") -> pd.Series:
            weights, *_ = frontier.get_max_return_portfolio()
            return weights.rename(label)

        return _max

    if key in {"risk_target", "max_return_subject_to_risk"}:
        if "max_risk" not in selector_kwargs:
            raise ValueError("`selector_kwargs` must include `max_risk` for 'risk_target'.")
        max_risk = selector_kwargs.pop("max_risk")

        def _risk(frontier: "PortfolioFrontier") -> pd.Series:
            weights, *_ = frontier.portfolio_at_risk_target(max_risk=max_risk)
            return weights.rename(label)

        return _risk

    if key in {"column", "index"}:
        column = selector_kwargs.pop("index", None)
        if column is None:
            raise ValueError("`selector_kwargs` must include `index` for 'column'.")

        def _column(frontier: "PortfolioFrontier") -> pd.Series:
            weights = frontier.weights[:, int(column)]
            return _series_from_vector(weights, frontier.asset_names, label)

        return _column

    raise ValueError(f"Unknown selector '{selector}'.")


def _build_distribution(
    *,
    returns: Optional[pd.DataFrame],
    probabilities: Optional[Union[pd.Series, Sequence[float], np.ndarray]],
    preprocess: Optional[Callable[[pd.DataFrame], pd.DataFrame]],
    projection: Optional[Dict[str, Any]],
    distribution: Optional["AssetsDistribution"],
    distribution_factory: Optional[Callable[[], "AssetsDistribution"]],
    use_scenarios: bool,
    mean_estimator: str,
    cov_estimator: str,
    mean_kwargs: Dict[str, Any],
    cov_kwargs: Dict[str, Any],
) -> "AssetsDistribution":
    from .portfolioapi import AssetsDistribution
    from .moments import estimate_moments
    from pyvallocation.utils.validation import ensure_psd_matrix

    if distribution_factory is not None:
        return distribution_factory()
    if distribution is not None:
        return distribution
    if returns is None:
        raise ValueError("Either `returns` or an explicit `distribution` must be supplied.")

    data = returns.copy()
    if preprocess is not None:
        data = preprocess(data)

    probs_series = _normalise_probabilities(probabilities, data.index) if probabilities is not None else None

    if use_scenarios:
        return AssetsDistribution(
            scenarios=data,
            probabilities=None if probs_series is None else probs_series.to_numpy(),
        )

    mu, sigma = estimate_moments(
        data,
        p=probs_series,
        mean_estimator=mean_estimator,
        cov_estimator=cov_estimator,
        mean_kwargs=mean_kwargs,
        cov_kwargs=cov_kwargs,
    )
    if projection:
        mu, sigma = _apply_projection(mu, sigma, projection)
    sigma_psd = ensure_psd_matrix(np.asarray(sigma, dtype=float))
    sigma_psd = sigma_psd + np.eye(sigma_psd.shape[0]) * 1e-10
    if isinstance(sigma, pd.DataFrame):
        sigma = pd.DataFrame(sigma_psd, index=sigma.index, columns=sigma.columns)
    else:
        sigma = sigma_psd
    return AssetsDistribution(mu=mu, cov=sigma)


def _normalise_probabilities(
    probabilities: Union[pd.Series, Sequence[float], np.ndarray],
    index: pd.Index,
) -> pd.Series:
    if isinstance(probabilities, pd.Series):
        probs = probabilities.reindex(index)
        if probs.isna().any():
            missing = probs[probs.isna()].index.tolist()
            raise ValueError(f"Probabilities missing for scenarios: {missing}.")
        return probs
    probs_arr = np.asarray(probabilities, dtype=float).reshape(-1)
    if probs_arr.shape[0] != len(index):
        raise ValueError("`probabilities` length must match number of scenarios.")
    return pd.Series(probs_arr, index=index)


def _apply_projection(
    mu: pd.Series,
    sigma: pd.DataFrame,
    projection: Dict[str, Any],
) -> Tuple[pd.Series, pd.DataFrame]:
    from pyvallocation.utils.projection import log2simple, project_mean_covariance

    options = dict(projection)
    if "annualization_factor" in options:
        af = options.pop("annualization_factor")
        mu, sigma = project_mean_covariance(mu, sigma, annualization_factor=af)
    if options.pop("log_to_simple", False) or options.pop("to_simple", False):
        mu, sigma = log2simple(mu, sigma)
    transform = options.pop("transform", None)
    if transform is not None:
        mu, sigma = transform(mu, sigma)
    if options:
        raise ValueError(f"Unknown projection options: {', '.join(options.keys())}")
    return mu, sigma

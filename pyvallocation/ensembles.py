"""
Ensemble utilities for blending portfolio weights.

The helpers in this module implement two complementary recipes that operate on
sample portfolios organised column-wise (``n_assets x n_samples``):

* :func:`average_exposures` - arithmetic averaging (optionally weighted) across
  a panel of sample portfolios.
* :func:`exposure_stacking` - the exposure-stacking quadratic programme first
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

from .probabilities import resolve_probabilities
from .utils.weights import (
    ArrayLike,
    ensure_samples_matrix,
    normalize_weights,
    wrap_exposure_vector,
)

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .portfolioapi import PortfolioFrontier

__all__ = [
    "average_exposures",
    "exposure_stacking",
    "stack_portfolios",
    "average_frontiers",
    "exposure_stack_frontiers",
    "risk_percentile_selections",
    "EnsembleSpec",
    "EnsembleResult",
    "assemble_portfolio_ensemble",
    "make_portfolio_spec",
]


def average_exposures(
    sample_portfolios: ArrayLike,
    weights: Optional[Union[Sequence[float], pd.Series]] = None,
) -> ArrayLike:
    """
    Compute the (possibly weighted) average exposure across multiple portfolios.

    The routine accepts any collection of sample weights arranged column-wise.
    When ``weights`` is omitted the average is uniform; otherwise ``weights``
    must supply one non-negative scalar per sample and is normalised to unity.

    Args:
        sample_portfolios: Array-like object whose columns represent sample portfolios.
        weights: Optional sequence or pandas Series of length ``n_samples`` providing
            relative importance for each column. When a Series is supplied its index
            is aligned to the sample column labels. The entries are automatically
            rescaled so that they sum to one.

    Returns:
        np.ndarray or pd.Series: Averaged exposure vector with length ``n_assets``.
        A pandas Series is returned when asset names are available on the input.

    Examples:
        >>> import numpy as np
        >>> samples = np.array([[0.6, 0.3], [0.4, 0.7]])
        >>> average_exposures(samples)
        array([0.45, 0.55])
        >>> average_exposures(samples, weights=[1.0, 3.0])
        array([0.375, 0.625])
    """
    exposures, asset_names, sample_names = ensure_samples_matrix(sample_portfolios)
    num_samples = exposures.shape[1]
    weights_vector = normalize_weights(weights, num_samples, sample_names)
    averaged = exposures @ weights_vector
    return wrap_exposure_vector(averaged, asset_names, label="Average Exposure")


@contextmanager
def _temporary_solver_options(overrides: Optional[dict]):
    """Context manager that applies temporary CVXOPT solver options.

    Args:
        overrides: Optional dictionary of solver option overrides.

    Yields:
        None
    """
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

    Args:
        sample_portfolios: Panel of sample portfolios organised column-wise.
        L: Number of cross-validation folds. Must satisfy ``1 <= L <= n_samples``.
        solver_options: Optional dictionary of CVXOPT solver overrides
            (e.g., ``{'maxiters': 100}``).

    Returns:
        np.ndarray or pd.Series: Exposure-stacked portfolio of length ``n_assets``.
        A Series is returned when asset names are provided on the input.

    Notes:
        This implementation adapts the open-source reference code from
        `fortitudo.tech <https://github.com/fortitudo-tech/fortitudo.tech>`_
        (GPL-3.0) that accompanies Vorobets' original publication.

    Raises:
        RuntimeError: If the underlying quadratic programme does not terminate
        with status ``'optimal'``.
    """
    exposures, asset_names, _ = ensure_samples_matrix(sample_portfolios)
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
    return wrap_exposure_vector(stacked, asset_names, label=f"Exposure Stacking (L={L})")


def stack_portfolios(
    portfolios: Sequence[Any],
    *,
    selections: Optional[Sequence[Optional[Iterable[int]]]] = None,
    L: int = 3,
    solver_options: Optional[dict] = None,
) -> pd.Series:
    """
    Stack a mixture of individual portfolios and/or frontiers.

    Each entry can be a pandas Series/NumPy vector (single portfolio) or a
    :class:`~pyvallocation.portfolioapi.PortfolioFrontier` (optionally paired with
    a selection of frontier columns via ``selections``).
    """
    matrix, asset_names = _collect_samples(portfolios, selections)
    if asset_names:
        samples = pd.DataFrame(matrix, index=asset_names)
    else:
        samples = matrix
    return exposure_stacking(samples, L=L, solver_options=solver_options)


def _merge_asset_names(
    existing: Optional[List[str]],
    candidate: Optional[Sequence[str]],
    dimension: int,
) -> Optional[List[str]]:
    """Validate/merge asset name lists across multiple samples.

    Args:
        existing: Current asset name list (or ``None`` if unset).
        candidate: Candidate asset name list inferred from a new sample.
        dimension: Expected number of assets for unlabelled samples.

    Returns:
        list[str] or None: Consolidated asset name list.

    Raises:
        ValueError: If asset labels conflict or dimensions mismatch.
    """
    candidate_list = list(candidate) if candidate else None
    if candidate_list:
        if existing is None:
            return candidate_list
        if existing != candidate_list:
            raise ValueError("All portfolios must share identical asset ordering.")
        return existing
    if existing is not None and len(existing) != dimension:
        raise ValueError("Portfolios without labels must match the established asset dimension.")
    return existing


def _collect_samples(
    entries: Sequence[Any],
    selections: Optional[Sequence[Optional[Iterable[int]]]] = None,
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Collect portfolios/frontiers into a single sample matrix.

    Args:
        entries: Sequence of portfolio vectors or ``PortfolioFrontier`` objects.
        selections: Optional per-entry column selections (for frontiers).

    Returns:
        Tuple[np.ndarray, Optional[list[str]]]: Stacked ``(n_assets, n_samples)`` matrix
        and optional asset labels.
    """
    if not entries:
        raise ValueError("At least one portfolio/frontier must be provided.")

    if selections is None:
        selections_iter = [None] * len(entries)
    else:
        if len(selections) != len(entries):
            raise ValueError("`selections` must match the number of entries supplied.")
        selections_iter = list(selections)

    stacked: List[np.ndarray] = []
    asset_names: Optional[List[str]] = None

    for entry, selection in zip(entries, selections_iter):
        if hasattr(entry, "to_samples"):
            cols: Optional[Iterable[int]]
            if selection is None:
                # default to minimum-risk column to avoid mixing risk levels
                if hasattr(entry, "risks"):
                    cols = [int(np.argmin(entry.risks))]
                else:
                    cols = [0]
            else:
                cols_list = list(selection)
                if len(cols_list) != 1:
                    raise ValueError("Provide exactly one column index per frontier to avoid mixing risk levels.")
                cols = [int(cols_list[0])]

            matrix, names = entry.to_samples(columns=cols, as_frame=False)
            asset_names = _merge_asset_names(asset_names, names, matrix.shape[0])
            stacked.append(matrix)
            continue

        matrix, names, _ = ensure_samples_matrix(entry)
        asset_names = _merge_asset_names(asset_names, names, matrix.shape[0])
        stacked.append(matrix)

    combined = np.hstack(stacked)
    if combined.size == 0:
        raise ValueError("No portfolio samples available for stacking.")
    return combined, asset_names


def _stack_frontiers(
    frontiers: Sequence[Any],
    selections: Optional[Sequence[Optional[Iterable[int]]]] = None,
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Alias for :func:`_collect_samples` with frontier inputs.

    Args:
        frontiers: Sequence of frontier-like objects.
        selections: Optional per-frontier column selections.

    Returns:
        Tuple[np.ndarray, Optional[list[str]]]: Stacked sample matrix and asset labels.
    """
    return _collect_samples(frontiers, selections)


def _series_from_vector(weights: np.ndarray, names: Optional[List[str]], label: str) -> pd.Series:
    """Helper returning a labelled Series with optional asset names.

    Args:
        weights: Weight vector to wrap.
        names: Optional asset labels.
        label: Series name.

    Returns:
        pd.Series: Labelled weight vector.
    """
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
    Average one portfolio from each frontier (aligned risk level).

    Args:
        frontiers: Sequence of frontier-like objects (typically
            :class:`~pyvallocation.portfolioapi.PortfolioFrontier` instances).
        selections: Optional per-frontier iterable selecting a **single** column
            index. When omitted, each frontier contributes its minimum-risk
            portfolio to avoid mixing risk levels.
        ensemble_weights: Optional weights applied to the stacked sample matrix
            before averaging. Must have length equal to the total number of
            selected portfolios.

    Returns:
        pd.Series: Averaged exposure vector with propagated asset labels when available.
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

    Args:
        frontiers: Sequence of frontier-like objects contributing sample portfolios.
        L: Number of stacking folds (as in :func:`exposure_stacking`).
        selections: Optional iterable specifying **one** column index per frontier.
            When omitted, each frontier contributes its minimum-risk portfolio.
        solver_options: Optional dictionary of CVXOPT solver overrides.

    Returns:
        pd.Series: Exposure-stacked weights with propagated asset labels.

    Notes:
        The total number of selected portfolios must be at least ``L``. When
        ``selections`` is omitted the full frontier matrices are used, matching the
        layout of :attr:`~pyvallocation.portfolioapi.PortfolioFrontier.weights`.
    """
    samples, names = _stack_frontiers(frontiers, selections)
    stacked = exposure_stacking(samples, L=L, solver_options=solver_options)
    return _series_from_vector(stacked, names, f"Exposure Stacking (L={L})")


def risk_percentile_selections(
    frontiers: Sequence["PortfolioFrontier"],
    percentile: float,
    *,
    risk_label: Optional[str] = None,
) -> List[List[int]]:
    """Return per-frontier column selections aligned by risk percentile.

    Args:
        frontiers: Sequence of frontier objects.
        percentile: Percentile on ``[0, 1]`` or ``[0, 100]``.
        risk_label: Optional risk label to align on.

    Returns:
        list[list[int]]: Column selections per frontier.
    """
    selections: List[List[int]] = []
    for frontier in frontiers:
        idx = frontier.index_at_risk_percentile(percentile, risk_label=risk_label)
        selections.append([idx])
    return selections


# --------------------------------------------------------------------------- #
# High-level ensemble orchestration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EnsembleSpec:
    """Descriptor for a single portfolio specification participating in an ensemble.

    Key fields:
        name: Spec identifier.
        frontier_factory: Callable returning a :class:`PortfolioFrontier`.
        selector: Callable extracting a representative portfolio.
        metadata: Optional metadata attached to the resulting ensemble output.
        frontier_selection: Optional subset of frontier columns for full-frontier blends.
    """

    name: str
    frontier_factory: Callable[[], "PortfolioFrontier"]
    selector: Callable[["PortfolioFrontier"], Union[pd.Series, Tuple[Any, ...], np.ndarray]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    frontier_selection: Optional[Sequence[int]] = None


@dataclass
class EnsembleResult:
    """Container returned by :func:`assemble_portfolio_ensemble`.

    Key fields:
        frontiers: Mapping of spec name to frontier object.
        selections: DataFrame of representative portfolios.
        ensembles: Mapping of ensemble label to weight Series.
        metadata: Per-spec metadata dictionaries.
    """

    frontiers: Dict[str, "PortfolioFrontier"]
    selections: pd.DataFrame
    ensembles: Dict[str, pd.Series]
    metadata: Dict[str, Dict[str, Any]]

    def get(self, name: str, default: Optional[pd.Series] = None) -> Optional[pd.Series]:
        """Return the ensemble weights by name (or ``default`` if missing).

        Args:
            name: Ensemble key (e.g. ``"average"`` or ``"stack"``).
            default: Value returned when ``name`` is not present.

        Returns:
            Optional[pd.Series]: Requested ensemble weights, if available.
        """
        return self.ensembles.get(name, default)

    @property
    def average(self) -> Optional[pd.Series]:
        """Convenience accessor for the average ensemble (if computed).

        Returns:
            Optional[pd.Series]: Average ensemble weights.
        """
        return self.ensembles.get("average")

    @property
    def stacked(self) -> Optional[pd.Series]:
        """Convenience accessor for the stacked ensemble (if computed).

        Returns:
            Optional[pd.Series]: Stacked ensemble weights.
        """
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

    Args:
        specs: Sequence of :class:`EnsembleSpec` instances describing how to generate and
            summarise each frontier.
        ensemble: ``"average"``, ``"stack"``, a sequence of the two, or ``None``.
            Defaults to ``"stack"`` for a stacked blend. Use ``("average", "stack")`` to
            obtain both.
        combine: ``"selected"`` (default) averages/stacks the representative portfolios
            extracted via each spec's selector. ``"frontier"`` operates directly on the
            underlying frontiers using :func:`average_frontiers` and
            :func:`exposure_stack_frontiers`.
        stack_folds: Number of folds for stacking. When omitted the helper picks
            ``min(3, number_of_portfolios)``.
        ensemble_weights: Optional weights applied during averaging (either over selected
            portfolios or the full frontier combination).
        stack_kwargs: Optional dictionary forwarded to the stacking solver
            (``solver_options`` argument).

    Returns:
        EnsembleResult: Rich result object containing the generated frontiers,
        representative portfolios, and any requested ensemble allocations.
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
            # enforce single column per frontier to avoid mixing risk levels
            cleaned_selections = []
            for frontier, sel in zip(frontier_list, selections_norm):
                if sel is None:
                    cleaned_selections.append([int(np.argmin(frontier.risks))])
                elif len(sel) != 1:
                    raise ValueError(
                        "frontier_selection must provide exactly one column index per frontier when combine='frontier'."
                    )
                else:
                    cleaned_selections.append([int(sel[0])])

            total_portfolios = sum(
                _frontier_selection_size(frontier, selection)
                for frontier, selection in zip(frontier_list, cleaned_selections)
            )
            folds = _determine_stack_folds(stack_folds, total_portfolios)
            for entry in ensemble_names:
                if entry == "average":
                    ensembles["average"] = average_frontiers(
                        frontier_list,
                        selections=cleaned_selections,
                        ensemble_weights=ensemble_weights,
                    )
                elif entry == "stack":
                    ensembles["stack"] = exposure_stack_frontiers(
                        frontier_list,
                        L=folds,
                        selections=cleaned_selections,
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

    Args:
        name: Spec identifier.
        returns: Historical scenario matrix (rows = scenarios, columns = assets).
        probabilities: Optional scenario weights aligned with ``returns``.
        preprocess: Optional callable applied to ``returns`` before estimation
            (e.g., convert compounded to simple returns).
        projection: Optional dictionary with projection settings. Recognised keys:
            ``annualization_factor`` (for :func:`project_mean_covariance`),
            ``log_to_simple``/``to_simple`` (apply :func:`log2simple`),
            and ``transform`` (callable ``transform(mu, Sigma) -> (mu, Sigma)``).
        distribution / distribution_factory: Supply an :class:`AssetsDistribution`
            directly (or a factory returning one) instead of estimating from data.
        use_scenarios: When ``True`` the distribution is built from scenarios
            rather than estimated moments.
        mean_estimator / cov_estimator: Names understood by
            :func:`pyvallocation.moments.estimate_moments`.
        mean_kwargs / cov_kwargs: Additional keyword arguments forwarded to the estimators.
        optimiser: Optimiser key (``"mean_variance"``, ``"cvar"``, ``"rrp"``, ``"robust"``)
            or a callable building a :class:`PortfolioFrontier` from a
            :class:`~pyvallocation.portfolioapi.PortfolioWrapper`.
        optimiser_kwargs: Keyword arguments for the optimiser. If it contains ``constraints``
            they are passed to :meth:`PortfolioWrapper.set_constraints`.
        selector: How to extract the representative portfolio. Accepts strings
            (``"tangency"``, ``"min_risk"``, ``"max_return"``, ``"risk_target"``,
            ``"risk_match"``, ``"risk_percentile"``, ``"column"``) or a callable.
        selector_kwargs: Extra parameters for the selector (e.g., ``risk_free_rate`` for tangency).
        frontier_selection: Column subset used when combining over entire frontiers.
        metadata: Optional dictionary persisted in the returned :class:`EnsembleResult`.

    Returns:
        EnsembleSpec: Spec object that encapsulates the distribution, optimiser, selector,
        and associated metadata.
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
        """Build a frontier using the configured distribution and optimiser.

        Returns:
            PortfolioFrontier: Optimised frontier instance.
        """
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
    """Coerce selector output into a labelled Series.

    Args:
        frontier: Frontier that provides asset labels.
        weights: Selector output (Series, ndarray, or tuple from frontier methods).
        label: Series name to assign.

    Returns:
        pd.Series: Weight vector aligned to asset labels when available.
    """
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
    """Normalize the ensemble selection argument to a tuple of lower-case keys.

    Args:
        ensemble: ``None``, a string key, or a sequence of keys.

    Returns:
        tuple[str, ...]: Normalized ensemble keys.
    """
    if ensemble is None:
        return tuple()
    if isinstance(ensemble, str):
        ensemble = ensemble.strip().lower()
        if ensemble in {"", "none"}:
            return tuple()
        return (ensemble,)
    return tuple(item.strip().lower() for item in ensemble if item and item.strip())


def _determine_stack_folds(stack_folds: Optional[int], total: int) -> int:
    """Return a valid number of stacking folds given available portfolios.

    Args:
        stack_folds: Requested fold count (or ``None`` for auto).
        total: Number of available portfolios.

    Returns:
        int: Validated number of folds.
    """
    if total <= 0:
        raise ValueError("No portfolios available for stacking.")
    if stack_folds is None:
        stack_folds = min(3, total)
    stack_folds = int(max(1, min(stack_folds, total)))
    return stack_folds


def _frontier_selection_size(frontier: "PortfolioFrontier", selection: Optional[Sequence[int]]) -> int:
    """Return the number of columns selected from a frontier.

    Args:
        frontier: Frontier instance.
        selection: Optional column indices.

    Returns:
        int: Number of columns included.
    """
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
    """Map a optimiser key/callable to a frontier factory.

    Args:
        optimiser: Callable or string key (e.g. ``"mean_variance"``).

    Returns:
        Callable[..., PortfolioFrontier]: Frontier builder.
    """
    if callable(optimiser):
        return optimiser
    key = optimiser.lower()

    def mean_variance(wrapper: "PortfolioWrapper", **kwargs):
        """Build a mean-variance frontier.

        Args:
            wrapper: Portfolio wrapper instance.
            **kwargs: Forwarded optimisation keyword arguments.

        Returns:
            PortfolioFrontier: Mean-variance frontier.
        """
        return wrapper.variance_frontier(**kwargs)

    def mean_cvar(wrapper: "PortfolioWrapper", **kwargs):
        """Build a mean-CVaR frontier.

        Args:
            wrapper: Portfolio wrapper instance.
            **kwargs: Forwarded optimisation keyword arguments.

        Returns:
            PortfolioFrontier: Mean-CVaR frontier.
        """
        return wrapper.cvar_frontier(**kwargs)

    def relaxed_rp(wrapper: "PortfolioWrapper", **kwargs):
        """Build a relaxed risk parity frontier.

        Args:
            wrapper: Portfolio wrapper instance.
            **kwargs: Forwarded optimisation keyword arguments.

        Returns:
            PortfolioFrontier: Relaxed risk parity frontier.
        """
        return wrapper.relaxed_risk_parity_frontier(**kwargs)

    def robust(wrapper: "PortfolioWrapper", **kwargs):
        """Build a robust frontier across uncertainty penalties.

        Args:
            wrapper: Portfolio wrapper instance.
            **kwargs: Forwarded optimisation keyword arguments.

        Returns:
            PortfolioFrontier: Robust frontier.
        """
        return wrapper.robust_lambda_frontier(**kwargs)

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
    """Resolve a selector string/callable into a frontier-to-weights function.

    Args:
        selector: Selector key or callable.
        selector_kwargs: Keyword arguments forwarded to the selector.
        label: Label used for the resulting Series.

    Returns:
        Callable[[PortfolioFrontier], pd.Series]: Selector function.
    """
    selector_kwargs = dict(selector_kwargs)

    if callable(selector):
        def _callable(frontier: "PortfolioFrontier") -> pd.Series:
            """Call a custom selector with user-supplied kwargs.

            Args:
                frontier: Portfolio frontier instance.

            Returns:
                pd.Series: Selected portfolio weights.
            """
            return selector(frontier, **selector_kwargs)

        return _callable

    key = selector.lower()

    if key in {"tangency", "max_sharpe", "sharpe"}:
        risk_free = selector_kwargs.pop("risk_free_rate", 0.0)

        def _tangency(frontier: "PortfolioFrontier") -> pd.Series:
            """Select the tangency portfolio using ``risk_free_rate``.

            Args:
                frontier: Portfolio frontier instance.

            Returns:
                pd.Series: Tangency portfolio weights.
            """
            weights, *_ = frontier.get_tangency_portfolio(risk_free_rate=risk_free)
            return weights.rename(label)

        return _tangency

    if key in {"min_risk", "minimum_risk"}:

        def _min(frontier: "PortfolioFrontier") -> pd.Series:
            """Select the minimum-risk portfolio.

            Args:
                frontier: Portfolio frontier instance.

            Returns:
                pd.Series: Minimum-risk portfolio weights.
            """
            risk_label = selector_kwargs.pop("risk_label", None)
            weights, *_ = frontier.get_min_risk_portfolio(risk_label=risk_label)
            return weights.rename(label)

        return _min

    if key in {"max_return", "maximum_return"}:

        def _max(frontier: "PortfolioFrontier") -> pd.Series:
            """Select the maximum-return portfolio.

            Args:
                frontier: Portfolio frontier instance.

            Returns:
                pd.Series: Maximum-return portfolio weights.
            """
            weights, *_ = frontier.get_max_return_portfolio()
            return weights.rename(label)

        return _max

    if key in {"risk_target", "max_return_subject_to_risk"}:
        if "max_risk" not in selector_kwargs:
            raise ValueError("`selector_kwargs` must include `max_risk` for 'risk_target'.")
        max_risk = selector_kwargs.pop("max_risk")
        risk_label = selector_kwargs.pop("risk_label", None)

        def _risk(frontier: "PortfolioFrontier") -> pd.Series:
            """Select the max-return portfolio under a risk limit.

            Args:
                frontier: Portfolio frontier instance.

            Returns:
                pd.Series: Selected portfolio weights.
            """
            resolved_label = risk_label
            if resolved_label is None and "Volatility" in frontier.alternate_risks:
                if "Estimation Risk" in (frontier.risk_measure or ""):
                    resolved_label = "Volatility"
            weights, *_ = frontier.portfolio_at_risk_target(max_risk=max_risk, risk_label=resolved_label)
            return weights.rename(label)

        return _risk

    if key in {"risk_match", "risk_nearest"}:
        if "target_risk" not in selector_kwargs:
            raise ValueError("`selector_kwargs` must include `target_risk` for 'risk_match'.")
        target_risk = selector_kwargs.pop("target_risk")
        risk_label = selector_kwargs.pop("risk_label", None)

        def _risk_match(frontier: "PortfolioFrontier") -> pd.Series:
            """Select the portfolio closest to ``target_risk``.

            Args:
                frontier: Portfolio frontier instance.

            Returns:
                pd.Series: Selected portfolio weights.
            """
            resolved_label = risk_label
            if resolved_label is None and "Volatility" in frontier.alternate_risks:
                if "Estimation Risk" in (frontier.risk_measure or ""):
                    resolved_label = "Volatility"
            weights, *_ = frontier.portfolio_closest_risk(target_risk, risk_label=resolved_label)
            return weights.rename(label)

        return _risk_match

    if key in {"risk_percentile", "risk_pct", "percentile"}:
        if "percentile" not in selector_kwargs:
            raise ValueError("`selector_kwargs` must include `percentile` for 'risk_percentile'.")
        percentile = selector_kwargs.pop("percentile")
        risk_label = selector_kwargs.pop("risk_label", None)

        def _risk_pct(frontier: "PortfolioFrontier") -> pd.Series:
            """Select the portfolio at a risk percentile.

            Args:
                frontier: Portfolio frontier instance.

            Returns:
                pd.Series: Selected portfolio weights.
            """
            resolved_label = risk_label
            if resolved_label is None and "Volatility" in frontier.alternate_risks:
                if "Estimation Risk" in (frontier.risk_measure or ""):
                    resolved_label = "Volatility"
            weights, *_ = frontier.portfolio_at_risk_percentile(percentile, risk_label=resolved_label)
            return weights.rename(label)

        return _risk_pct

    if key in {"column", "index"}:
        column = selector_kwargs.pop("index", None)
        if column is None:
            raise ValueError("`selector_kwargs` must include `index` for 'column'.")

        def _column(frontier: "PortfolioFrontier") -> pd.Series:
            """Select a portfolio by column index.

            Args:
                frontier: Portfolio frontier instance.

            Returns:
                pd.Series: Selected portfolio weights.
            """
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
    """Construct an :class:`AssetsDistribution` from inputs or factories.

    Returns:
        AssetsDistribution: Distribution derived from scenarios or estimated moments.
    """
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

    if probabilities is not None:
        if isinstance(probabilities, pd.Series):
            probs_series_aligned = probabilities.reindex(data.index)
            if probs_series_aligned.isna().any():
                missing = probs_series_aligned[probs_series_aligned.isna()].index.tolist()
                raise ValueError(f"Probabilities missing for scenarios: {missing}.")
            probs_array = resolve_probabilities(
                probs_series_aligned.to_numpy(dtype=float),
                len(data),
                name="probabilities",
            )
            probs_series = pd.Series(probs_array, index=data.index)
        else:
            probs_array = resolve_probabilities(
                probabilities,
                len(data),
                name="probabilities",
            )
            probs_series = pd.Series(probs_array, index=data.index)
    else:
        probs_series = None

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


def _apply_projection(
    mu: pd.Series,
    sigma: pd.DataFrame,
    projection: Dict[str, Any],
) -> Tuple[pd.Series, pd.DataFrame]:
    """Apply projection options (annualisation, log/simple transform, custom).

    Args:
        mu: Mean vector.
        sigma: Covariance matrix.
        projection: Projection options dictionary.

    Returns:
        Tuple[pd.Series, pd.DataFrame]: Projected mean and covariance.
    """
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

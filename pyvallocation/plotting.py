"""Plotting utilities for efficient frontiers and portfolio summaries."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np

from .portfolioapi import PortfolioFrontier

if TYPE_CHECKING:  # pragma: no cover - typing helper
    import pandas as pd


def _require_matplotlib():
    """
    Import :mod:`matplotlib.pyplot` and fall back to the Agg backend when GUI
    toolkits are unavailable (e.g. headless CI environments).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - guarded by tests
        raise ImportError(
            "`plot_frontiers` requires `matplotlib`. Install it via `pip install matplotlib`."
        ) from exc

    try:
        fig = plt.figure()
    except Exception:  # pragma: no cover - backend fallback
        try:
            plt.switch_backend("Agg")
        except Exception:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt  # type: ignore[no-redef]

        fig = plt.figure()

    plt.close(fig)
    return plt


def _normalize_frontiers(
    frontiers: Union[PortfolioFrontier, Sequence[PortfolioFrontier], Mapping[str, PortfolioFrontier]],
    labels: Optional[Sequence[str]],
) -> List[Tuple[Optional[str], PortfolioFrontier]]:
    """Normalize frontier inputs into a list of ``(label, frontier)`` pairs.

    Args:
        frontiers: Single frontier, sequence, or mapping of label -> frontier.
        labels: Optional sequence of labels aligned to the frontier sequence.

    Returns:
        list[tuple[str | None, PortfolioFrontier]]: Normalized list of frontiers.
    """
    if isinstance(frontiers, Mapping):
        result: List[Tuple[Optional[str], PortfolioFrontier]] = []
        for name, frontier in frontiers.items():
            if not isinstance(frontier, PortfolioFrontier):
                raise TypeError("`frontiers` must contain `PortfolioFrontier` instances.")
            result.append((str(name), frontier))
        return result

    if isinstance(frontiers, PortfolioFrontier):
        frontier_list: Sequence[PortfolioFrontier] = [frontiers]
    else:
        frontier_list = list(frontiers)

    if labels is not None and len(labels) != len(frontier_list):
        raise ValueError("`labels` length must match the number of frontiers supplied.")

    result: List[Tuple[Optional[str], PortfolioFrontier]] = []
    for idx, frontier in enumerate(frontier_list):
        if not isinstance(frontier, PortfolioFrontier):
            raise TypeError("`frontiers` must contain `PortfolioFrontier` instances.")
        label = labels[idx] if labels is not None else None
        result.append((label, frontier))
    return result


def _project_to_3d(
    mean: np.ndarray,
    cov: np.ndarray,
    scenarios: Optional[np.ndarray] = None,
    *,
    components: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Project mean/cov (and optionally scenarios) into 3D using PCA directions.

    Args:
        mean: Mean vector of length ``n_assets``.
        cov: Covariance matrix of shape ``(n_assets, n_assets)``.
        scenarios: Optional scenario matrix shaped ``(n_scenarios, n_assets)``.
        components: Optional projection matrix with 3 columns. When ``None``,
            the top-3 eigenvectors of ``cov`` are used.

    Returns:
        Tuple containing the projected mean, covariance, projected scenarios (or ``None``),
        and the projection matrix used.
    """
    mean = np.asarray(mean, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("`cov` must be a square matrix.")
    if cov.shape[0] != mean.size:
        raise ValueError("`mean` length must match `cov` dimensions.")
    cov = 0.5 * (cov + cov.T)

    n_assets = mean.size
    if n_assets >= 3:
        if components is None:
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            components = eigvecs[:, order[:3]]
        mean_3d = components.T @ mean
        cov_3d = components.T @ cov @ components
        if scenarios is not None:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                scen_3d = scenarios @ components
        else:
            scen_3d = None
        return mean_3d, cov_3d, scen_3d, components

    mean_3d = np.zeros(3)
    mean_3d[:n_assets] = mean
    cov_3d = np.zeros((3, 3))
    cov_3d[:n_assets, :n_assets] = cov
    scen_3d = None
    if scenarios is not None:
        scen_3d = np.zeros((scenarios.shape[0], 3))
        scen_3d[:, :n_assets] = scenarios
    components = np.eye(n_assets, 3)
    return mean_3d, cov_3d, scen_3d, components


def _ellipsoid_mesh(
    cov: np.ndarray,
    *,
    n_std: float = 1.0,
    n_points: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate an ellipsoid mesh representing covariance uncertainty.

    Args:
        cov: 3x3 covariance matrix.
        n_std: Ellipsoid radius in standard deviations. Defaults to ``1.0``.
        n_points: Mesh resolution along each angular dimension.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: ``(x, y, z)`` mesh arrays.
    """
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    radii = n_std * np.sqrt(eigvals + 1e-16)

    u = np.linspace(0.0, 2.0 * np.pi, n_points)
    v = np.linspace(0.0, np.pi, n_points)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    coords = np.stack([x, y, z], axis=0).reshape(3, -1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        coords = eigvecs @ coords
    x, y, z = coords.reshape(3, n_points, n_points)
    return x, y, z


def _resolve_highlight(
    frontier: PortfolioFrontier,
    marker: str,
    *,
    risk_vector: np.ndarray,
    risk_free_rate: Optional[float],
) -> Optional[Tuple[str, float, float, "pd.Series", int]]:
    """Resolve a highlighted portfolio point for plotting.

    Args:
        frontier: Frontier instance.
        marker: Highlight key (``min_risk``, ``max_return``, ``tangency``).
        risk_vector: Risk values associated with the frontier.
        risk_free_rate: Risk-free rate used for tangency selection.

    Returns:
        Optional[tuple]: Highlight label, risk, return, weights, and column index.
    """
    if marker == "min_risk":
        idx = int(np.argmin(risk_vector))
        target_risk = float(risk_vector[idx])
        target_return = float(frontier.returns[idx])
        weights = frontier._to_pandas(frontier.weights[:, idx], "Min Risk Portfolio")
        name = "Min Risk"
    elif marker == "max_return":
        idx = int(np.argmax(frontier.returns))
        target_risk = float(risk_vector[idx])
        target_return = float(frontier.returns[idx])
        weights = frontier._to_pandas(frontier.weights[:, idx], "Max Return Portfolio")
        name = "Max Return"
    elif marker == "tangency":
        if risk_free_rate is None:
            raise ValueError("Highlighting the tangency portfolio requires `risk_free_rate`.")
        if np.all(np.isclose(risk_vector, 0)):
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe_ratios = (frontier.returns - risk_free_rate) / risk_vector
        sharpe_ratios[~np.isfinite(sharpe_ratios)] = -np.inf
        idx = int(np.argmax(sharpe_ratios))
        if not np.isfinite(frontier.returns[idx]) or not np.isfinite(risk_vector[idx]):
            return None
        target_risk = float(risk_vector[idx])
        target_return = float(frontier.returns[idx])
        weights = frontier._to_pandas(
            frontier.weights[:, idx],
            f"Tangency Portfolio (rf={risk_free_rate:.2%})",
        )
        name = f"Tangency (rf={risk_free_rate:.2%})"
    else:
        raise ValueError(f"Unknown highlight '{marker}'.")

    if not (np.isfinite(target_return) and np.isfinite(target_risk)):
        return None
    return name, target_risk, target_return, weights, idx


def plot_frontiers(
    frontiers: Union[PortfolioFrontier, Sequence[PortfolioFrontier], Mapping[str, PortfolioFrontier]],
    *,
    ax=None,
    labels: Optional[Sequence[str]] = None,
    highlight: Iterable[str] = ("min_risk", "max_return"),
    risk_free_rate: Optional[float] = None,
    legend: bool = True,
    line_kwargs: Optional[Mapping[str, object]] = None,
    marker_kwargs: Optional[Mapping[str, Mapping[str, object]]] = None,
    scatter_kwargs: Optional[Mapping[str, object]] = None,
    show_points: bool = False,
    percent_axes: bool = False,
    overlay_risk_labels: Optional[Sequence[str]] = None,
    risk_label: Optional[str] = None,
    return_label: str = "Expected Return",
    highlight_metadata_keys: Optional[Sequence[str]] = None,
    metadata_value_formatter: Optional[Callable[[str, object], str]] = None,
):
    """Plot one or more efficient frontiers.

    Args:
        frontiers: Frontier instance(s) to be visualised.
        ax: Optional matplotlib axis. If omitted, a new figure and axis are created.
        labels: Optional labels associated with each frontier.
        highlight: Iterable of portfolio markers to emphasise. Valid entries are
            ``"min_risk"``, ``"max_return"`` and ``"tangency"``.
        risk_free_rate: Required when highlighting the tangency portfolio.
        legend: Whether to render a legend.
        line_kwargs: Keyword arguments passed to ``Axes.plot`` for the frontier lines.
        marker_kwargs: Mapping from highlight name to keyword arguments for the
            corresponding scatter points.
        scatter_kwargs: Global keyword arguments applied to all highlight markers.
        show_points: When ``True``, scatter all frontier nodes (useful for reports).
        percent_axes: Format both axes as percentages (assumes decimal inputs).
        overlay_risk_labels: Optional risk labels to overlay as dashed diagnostic lines
            (e.g., raw CVaR before convexification).
        risk_label: Axis label for the risk dimension. If supplied, the plotting
            uses the corresponding risk grid (primary or ``alternate_risks``) from
            each frontier. When omitted, the primary risk is used.
        return_label: Axis label for expected returns.
        highlight_metadata_keys: Optional iterable of metadata field names to append
            to highlight labels when the underlying :class:`PortfolioFrontier`
            exposes ``metadata``.
        metadata_value_formatter: Optional callable used to render metadata values
            in highlight labels. Receives ``(key, value)`` and must return a string.

    Returns:
        The matplotlib ``Axes`` containing the plot. The axis is also populated
        with a ``_frontier_highlights`` attribute containing the highlighted points.
    """

    plt = _require_matplotlib()
    highlight = tuple(highlight) if highlight else tuple()
    normalized = _normalize_frontiers(frontiers, labels)

    if ax is None:
        _, ax = plt.subplots()

    line_kwargs = dict(line_kwargs or {})
    scatter_kwargs = dict(scatter_kwargs or {})
    marker_kwargs = marker_kwargs or {}
    highlight_records = []
    metadata_keys = tuple(highlight_metadata_keys) if highlight_metadata_keys else tuple()

    if metadata_value_formatter is None:
        def _default_metadata_formatter(key: str, value: object) -> str:
            """Format metadata key/value pairs for highlight labels.

            Args:
                key: Metadata key.
                value: Metadata value.

            Returns:
                str: Formatted key/value token.
            """
            if isinstance(value, (float, np.floating)):
                return f"{key}={value:.4f}"
            return f"{key}={value}"
    else:
        _default_metadata_formatter = metadata_value_formatter

    overlay_labels = tuple(overlay_risk_labels or ())

    for supplied_label, frontier in normalized:
        risk_vector = frontier._risk_vector(risk_label)
        label = supplied_label or (risk_label or frontier.risk_measure)
        line, = ax.plot(risk_vector, frontier.returns, label=label, **line_kwargs)
        colour = line.get_color()
        if show_points:
            ax.scatter(risk_vector, frontier.returns, color=colour, s=12, alpha=0.6, zorder=line.get_zorder() + 1)
        for overlay in overlay_labels:
            try:
                overlay_risk = frontier._risk_vector(overlay)
            except KeyError:
                continue
            ax.plot(
                overlay_risk,
                frontier.returns,
                linestyle="--",
                color=colour,
                alpha=0.6,
                label=f"{label} ({overlay})",
            )

        for marker in highlight:
            resolved = _resolve_highlight(
                frontier,
                marker,
                risk_vector=risk_vector,
                risk_free_rate=risk_free_rate,
            )
            if resolved is None:
                continue
            display_name, risk_value, return_value, weights, idx = resolved
            metadata_entry = (
                frontier.metadata[idx] if frontier.metadata and idx < len(frontier.metadata) else None
            )

            style = {
                "color": colour,
                "s": 60,
                "zorder": line.get_zorder() + 1,
            }
            style.update(scatter_kwargs)
            style.update(marker_kwargs.get(marker, {}))

            highlight_label = f"{label} - {display_name}"
            if metadata_entry and metadata_keys:
                tokens: List[str] = []
                for key in metadata_keys:
                    if key not in metadata_entry:
                        continue
                    value = metadata_entry[key]
                    if value is None:
                        continue
                    formatted = _default_metadata_formatter(key, value)
                    if formatted:
                        tokens.append(formatted)
                if tokens:
                    highlight_label = f"{highlight_label} ({', '.join(tokens)})"

            ax.scatter(risk_value, return_value, label=highlight_label, **style)
            highlight_records.append(
                {
                    "frontier": label,
                    "type": display_name,
                    "risk": risk_value,
                    "return": return_value,
                    "risk_measure": risk_label or frontier.risk_measure,
                    "weights": weights,
                    "index": idx,
                    "metadata": metadata_entry,
                }
            )

    if not risk_label:
        unique_measures = {frontier.risk_measure for _, frontier in normalized}
        risk_label = unique_measures.pop() if len(unique_measures) == 1 else "Risk"

    ax.set_xlabel(risk_label)
    ax.set_ylabel(return_label)
    if percent_axes:
        try:
            from matplotlib.ticker import FuncFormatter
            fmt = FuncFormatter(lambda x, _pos: f"{x:.1%}")
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
        except Exception:  # pragma: no cover - optional formatting
            pass
    ax.grid(True, alpha=0.3)

    if legend:
        ax.legend()

    ax._frontier_highlights = highlight_records  # type: ignore[attr-defined]
    return ax


def plot_frontiers_grid(
    frontiers: Union[PortfolioFrontier, Sequence[PortfolioFrontier], Mapping[str, PortfolioFrontier]],
    *,
    by: Optional[Callable[[Optional[str], PortfolioFrontier], str]] = None,
    labels: Optional[Sequence[str]] = None,
    cols: Optional[int] = None,
    figsize: Tuple[float, float] = (12.0, 4.5),
    sharex: bool = False,
    sharey: bool = False,
    highlight: Iterable[str] = ("min_risk", "max_return"),
    risk_free_rate: Optional[float] = None,
    legend: bool = True,
    legend_policy: str = "auto",
    **kwargs,
):
    """Plot groups of efficient frontiers on a grid.

    Args:
        frontiers: Single frontier, sequence, or mapping of frontiers.
        by: Optional grouping key function ``(label, frontier) -> group``.
        labels: Optional labels aligned to the frontier sequence.
        cols: Number of columns in the grid. Defaults to ``len(groups)``.
        figsize: Figure size in inches.
        sharex: Whether to share x-axes across subplots.
        sharey: Whether to share y-axes across subplots.
        highlight: Iterable of highlight markers (e.g. ``min_risk``).
        risk_free_rate: Risk-free rate for tangency highlights.
        legend: Whether to show legends.
        legend_policy: ``auto`` (default), ``all``, or ``none``.
        **kwargs: Forwarded to :func:`plot_frontiers`.

    Returns:
        matplotlib.figure.Figure: Figure containing the grid of plots.
    """

    plt = _require_matplotlib()
    normalized = _normalize_frontiers(frontiers, labels)
    grouper = by or (lambda supplied, fr: fr.risk_measure or "Risk")

    grouped: "OrderedDict[str, OrderedDict[str, PortfolioFrontier]]" = OrderedDict()
    for supplied_label, frontier in normalized:
        key = str(grouper(supplied_label, frontier))
        bucket = grouped.setdefault(key, OrderedDict())
        bucket[supplied_label or frontier.risk_measure] = frontier

    if not grouped:
        raise ValueError("No frontiers supplied.")

    cols = cols or len(grouped)
    rows = (len(grouped) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=sharex, sharey=sharey)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (group_name, mapping) in zip(axes_flat, grouped.items()):
        show_legend = legend
        if legend_policy == "auto":
            show_legend = legend and len(grouped) == 1
        elif legend_policy == "none":
            show_legend = False
        elif legend_policy == "each":
            show_legend = legend
        else:
            raise ValueError("`legend_policy` must be 'auto', 'each', or 'none'.")

        plot_frontiers(
            mapping,
            ax=ax,
            highlight=highlight,
            risk_free_rate=risk_free_rate,
            legend=show_legend,
            **kwargs,
        )
        ax.set_title(group_name)

    for ax in axes_flat[len(grouped) :]:
        ax.axis("off")

    fig.tight_layout()
    return fig, axes_flat


def plot_weights(
    weights,
    *,
    ax=None,
    top_n: Optional[int] = None,
    title: Optional[str] = None,
    kind: str = "barh",
    stacked: bool = False,
    percent_axes: bool = True,
    legend: bool = True,
):
    """Plot portfolio weights (Series or DataFrame) with minimal setup.

    Args:
        weights: Weight Series or DataFrame.
        ax: Optional matplotlib axis to draw on.
        top_n: If set, plot only the largest ``top_n`` weights.
        title: Optional plot title.
        kind: Plot kind (e.g., ``barh`` or ``bar``).
        stacked: Whether to stack DataFrame columns.
        percent_axes: If ``True`` format axes as percentages.
        legend: Whether to show legend for DataFrame inputs.

    Returns:
        matplotlib.axes.Axes: Axis with the plot.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    import pandas as pd  # local import keeps plotting optional

    if isinstance(weights, pd.Series):
        series = weights.sort_values(ascending=True) if kind == "barh" else weights.sort_values(ascending=False)
        if top_n is not None:
            series = series.tail(top_n) if kind == "barh" else series.head(top_n)
        series.plot(kind=kind, ax=ax, legend=False)
    elif isinstance(weights, pd.DataFrame):
        data = weights.copy()
        data.plot(kind=kind, ax=ax, stacked=stacked, legend=legend)
    else:
        raise TypeError("`weights` must be a pandas Series or DataFrame.")

    if percent_axes:
        try:
            from matplotlib.ticker import FuncFormatter
            fmt = FuncFormatter(lambda x, _pos: f"{x:.1%}")
            ax.xaxis.set_major_formatter(fmt)
            if kind == "bar":
                ax.yaxis.set_major_formatter(fmt)
        except Exception:  # pragma: no cover
            pass

    if title:
        ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    return ax


def plot_robust_path(
    frontier: PortfolioFrontier,
    *,
    param: str = "lambda",
    risk_label: Optional[str] = None,
    overlay_risk_label: Optional[str] = None,
    ax=None,
    cmap: str = "viridis",
    percent_axes: bool = False,
    show_points: bool = True,
):
    """Plot robust frontier with colour indicating parameter impact.

    Args:
        frontier: Robust frontier instance with metadata.
        param: Metadata key to color by (default ``"lambda"``).
        risk_label: Risk label for the x-axis.
        overlay_risk_label: Optional risk label to overlay as a dashed line.
        ax: Optional matplotlib axis to draw on.
        cmap: Colormap name for parameter coloring.
        percent_axes: Whether to format axes as percentages.
        show_points: Whether to show individual frontier nodes.

    Returns:
        matplotlib.axes.Axes: Axis with the plot.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    risk_vec = frontier._risk_vector(risk_label)
    values = None
    if frontier.metadata:
        values = np.array([meta.get(param) for meta in frontier.metadata], dtype=float)
    if values is None or not np.all(np.isfinite(values)):
        values = np.linspace(0.0, 1.0, len(risk_vec))

    sc = ax.scatter(risk_vec, frontier.returns, c=values, cmap=cmap, s=30, zorder=3)
    if show_points:
        ax.plot(risk_vec, frontier.returns, color="black", alpha=0.3, zorder=2)
    if overlay_risk_label is not None:
        try:
            overlay = frontier._risk_vector(overlay_risk_label)
            ax.plot(overlay, frontier.returns, linestyle="--", alpha=0.6, color="tab:gray")
        except KeyError:
            pass

    ax.set_xlabel(risk_label or frontier.risk_measure)
    ax.set_ylabel("Expected Return")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(param)

    if percent_axes:
        try:
            from matplotlib.ticker import FuncFormatter
            fmt = FuncFormatter(lambda x, _pos: f"{x:.1%}")
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
        except Exception:  # pragma: no cover
            pass
    return ax


def plot_param_impact(
    frontier: PortfolioFrontier,
    *,
    param: str = "lambda",
    risk_label: Optional[str] = None,
    percent_axes: bool = False,
    ax=None,
):
    """Plot parameter vs return/risk in a two-panel figure.

    Args:
        frontier: Frontier with metadata containing parameter values.
        param: Metadata key to plot along the x-axis.
        risk_label: Risk label for the risk panel.
        percent_axes: Whether to format y-axes as percentages.
        ax: Optional tuple of axes or a figure axis to reuse.

    Returns:
        Tuple[matplotlib.figure.Figure, tuple]: Figure and axes tuple.
    """
    plt = _require_matplotlib()
    if frontier.metadata is None:
        raise ValueError("Frontier metadata missing parameter values.")
    values = np.array([meta.get(param) for meta in frontier.metadata], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Metadata does not contain finite '{param}' values.")
    order = np.argsort(values)
    values = values[order]
    returns = np.asarray(frontier.returns, dtype=float)[order]
    risks = frontier._risk_vector(risk_label)[order]
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    else:
        fig = ax.figure
        axes = ax

    ax_return, ax_risk = axes
    ax_return.plot(values, returns, marker="o")
    ax_return.set_xlabel(param)
    ax_return.set_ylabel("Expected Return")
    ax_return.grid(True, alpha=0.3)

    ax_risk.plot(values, risks, marker="o")
    ax_risk.set_xlabel(param)
    ax_risk.set_ylabel(risk_label or frontier.risk_measure)
    ax_risk.grid(True, alpha=0.3)

    if percent_axes:
        try:
            from matplotlib.ticker import FuncFormatter
            fmt = FuncFormatter(lambda x, _pos: f"{x:.1%}")
            ax_return.yaxis.set_major_formatter(fmt)
            ax_risk.yaxis.set_major_formatter(fmt)
        except Exception:  # pragma: no cover
            pass

    fig.tight_layout()
    return fig, axes


def plot_robust_surface(
    frontier: PortfolioFrontier,
    *,
    param: str = "lambda",
    risk_label: Optional[str] = None,
    ax=None,
):
    """3D scatter of parameter vs risk vs return for robust frontiers.

    Args:
        frontier: Frontier with metadata containing parameter values.
        param: Metadata key to use for the x-axis and color.
        risk_label: Risk label for the y-axis.
        ax: Optional matplotlib 3D axis to reuse.

    Returns:
        matplotlib.axes.Axes: 3D axis with the scatter plot.
    """
    plt = _require_matplotlib()
    if frontier.metadata is None:
        raise ValueError("Frontier metadata missing parameter values.")
    values = np.array([meta.get(param) for meta in frontier.metadata], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Metadata does not contain finite '{param}' values.")
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    ax.scatter(values, frontier._risk_vector(risk_label), frontier.returns, c=values, cmap="viridis")
    ax.set_xlabel(param)
    ax.set_ylabel(risk_label or frontier.risk_measure)
    ax.set_zlabel("Expected Return")
    return ax


def plot_assumptions_3d(
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    scenarios: Optional[np.ndarray] = None,
    uncertainty_cov: Optional[np.ndarray] = None,
    n_std: float = 1.0,
    uncertainty_std: float = 1.0,
    max_points: int = 2500,
    titles: Optional[Tuple[str, str]] = None,
):
    """Plot mean/covariance assumptions (and optional mean uncertainty) in 3D.

    The function projects high-dimensional inputs onto the first three
    principal components implied by ``cov`` and renders:
    - a covariance ellipsoid around the mean,
    - optional scenario clouds,
    - and, when provided, a second panel for the mean-uncertainty ellipsoid.
    """
    plt = _require_matplotlib()

    scenarios_arr = None
    if scenarios is not None:
        scenarios_arr = np.asarray(scenarios, dtype=float)
        if scenarios_arr.ndim != 2:
            raise ValueError("`scenarios` must be a 2D array.")
        if scenarios_arr.shape[1] != np.asarray(mean, dtype=float).size:
            raise ValueError("`scenarios` columns must match `mean` length.")
        if scenarios_arr.shape[0] > max_points:
            idx = np.linspace(0, scenarios_arr.shape[0] - 1, max_points).astype(int)
            scenarios_arr = scenarios_arr[idx]

    cov_arr = np.asarray(cov, dtype=float)
    mean_3d, cov_3d, scen_3d, components = _project_to_3d(mean, cov_arr, scenarios_arr)

    if uncertainty_cov is None:
        fig = plt.figure(figsize=(6.5, 5.0))
        ax_main = fig.add_subplot(111, projection="3d")
        axes = (ax_main,)
    else:
        fig = plt.figure(figsize=(12.0, 5.0))
        ax_main = fig.add_subplot(121, projection="3d")
        ax_unc = fig.add_subplot(122, projection="3d")
        axes = (ax_main, ax_unc)

    def _style_axis(ax, title: str, labels: Tuple[str, str, str]):
        """Apply consistent formatting to 3D axes.

        Args:
            ax: Matplotlib 3D axis.
            title: Axis title.
            labels: Tuple of axis labels.
        """
        ax.set_title(title)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        ax.grid(True, alpha=0.2)
        ax.view_init(elev=22, azim=35)
        try:  # matplotlib >= 3.3
            ax.set_box_aspect((1, 1, 1))
        except Exception:  # pragma: no cover
            pass

    def _plot_principal_axes(ax, mean_vec, cov_mat, *, color: str, alpha: float):
        """Draw principal axis lines for a covariance ellipsoid.

        Args:
            ax: Matplotlib 3D axis.
            mean_vec: Mean vector in 3D.
            cov_mat: 3x3 covariance matrix.
            color: Line color.
            alpha: Line opacity.
        """
        cov_mat = 0.5 * (cov_mat + cov_mat.T)
        eigvals, eigvecs = np.linalg.eigh(cov_mat)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 0.0, None)
        eigvecs = eigvecs[:, order]
        lengths = np.sqrt(eigvals + 1e-16)
        for idx in range(3):
            vec = eigvecs[:, idx] * lengths[idx]
            xs = [mean_vec[0] - vec[0], mean_vec[0] + vec[0]]
            ys = [mean_vec[1] - vec[1], mean_vec[1] + vec[1]]
            zs = [mean_vec[2] - vec[2], mean_vec[2] + vec[2]]
            ax.plot(xs, ys, zs, color=color, alpha=alpha, linewidth=1.4)

    def _plot_distribution(ax, mean_vec, cov_mat, scen_vec, title, *, cov_color, cov_alpha, labels):
        """Plot mean/covariance ellipsoid with optional scenarios.

        Args:
            ax: Matplotlib 3D axis.
            mean_vec: Mean vector in 3D.
            cov_mat: 3x3 covariance matrix.
            scen_vec: Optional scenario projections.
            title: Subplot title.
            cov_color: Ellipsoid color.
            cov_alpha: Ellipsoid opacity.
            labels: Axis labels.
        """
        x, y, z = _ellipsoid_mesh(cov_mat, n_std=n_std, n_points=32)
        ax.plot_surface(
            x + mean_vec[0],
            y + mean_vec[1],
            z + mean_vec[2],
            color=cov_color,
            alpha=cov_alpha,
            linewidth=0.0,
            shade=False,
        )
        if scen_vec is not None:
            ax.scatter(
                scen_vec[:, 0],
                scen_vec[:, 1],
                scen_vec[:, 2],
                s=8,
                alpha=0.15,
                color="0.35",
            )
        ax.scatter(mean_vec[0], mean_vec[1], mean_vec[2], color="crimson", s=50)
        _plot_principal_axes(ax, mean_vec, cov_mat, color=cov_color, alpha=0.6)
        _style_axis(ax, title, labels)

        try:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch

            handles = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson", markersize=6, label="Mean"),
                Patch(facecolor=cov_color, alpha=cov_alpha, label="Covariance (1σ)"),
            ]
            if scen_vec is not None:
                handles.insert(
                    0,
                    Line2D([0], [0], marker=".", color="0.35", linestyle="", label="Scenarios"),
                )
            ax.legend(handles=handles, loc="upper left", frameon=True)
        except Exception:  # pragma: no cover
            pass

    titles = titles or ("Return distribution", "Mean uncertainty")
    eigvals = np.linalg.eigvalsh(cov_arr)
    eigvals = np.clip(eigvals, 0.0, None)
    order = np.argsort(eigvals)[::-1]
    explained = eigvals[order]
    total = float(explained.sum()) if explained.sum() > 0 else 1.0
    explained = explained[:3] / total
    if explained.size < 3:
        explained = np.pad(explained, (0, 3 - explained.size), constant_values=0.0)
    labels = (
        f"PC1 ({explained[0]:.0%})",
        f"PC2 ({explained[1]:.0%})",
        f"PC3 ({explained[2]:.0%})",
    )
    _plot_distribution(
        ax_main,
        mean_3d,
        cov_3d,
        scen_3d,
        titles[0],
        cov_color="#377eb8",
        cov_alpha=0.22,
        labels=labels,
    )

    if uncertainty_cov is not None:
        mean_u, cov_u, _, _ = _project_to_3d(mean, uncertainty_cov, None, components=components)
        x_u, y_u, z_u = _ellipsoid_mesh(cov_u, n_std=uncertainty_std, n_points=32)
        ax_unc.plot_surface(
            x_u + mean_u[0],
            y_u + mean_u[1],
            z_u + mean_u[2],
            color="#ff7f00",
            alpha=0.35,
            linewidth=0.0,
            shade=False,
        )
        ax_unc.scatter(mean_u[0], mean_u[1], mean_u[2], color="crimson", s=50)
        _plot_principal_axes(ax_unc, mean_u, cov_u, color="#ff7f00", alpha=0.7)
        _style_axis(ax_unc, titles[1], labels)
        try:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch

            handles = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson", markersize=6, label="Mean"),
                Patch(facecolor="#ff7f00", alpha=0.35, label="Mean uncertainty (1σ)"),
            ]
            ax_unc.legend(handles=handles, loc="upper left", frameon=True)
        except Exception:  # pragma: no cover
            pass

    fig.tight_layout()
    return fig, axes


def plot_frontier_report(
    frontier: PortfolioFrontier,
    *,
    selection: str = "min_risk",
    selection_kwargs: Optional[Mapping[str, object]] = None,
    risk_label: Optional[str] = None,
    overlay_risk_labels: Optional[Sequence[str]] = None,
    show_points: bool = True,
    percent_axes: bool = True,
    weights: Optional["pd.Series"] = None,
    weights_title: str = "Weights",
    figsize: Tuple[float, float] = (10.0, 4.0),
):
    """Compact report plot: frontier plus selected weights bar.

    Args:
        frontier: Frontier to plot.
        selection: Selector key (e.g. ``min_risk``, ``tangency``).
        selection_kwargs: Keyword arguments for the selector.
        risk_label: Optional risk label for the frontier x-axis.
        overlay_risk_labels: Optional list of risk labels to overlay.
        show_points: Whether to draw frontier nodes.
        percent_axes: Whether to format axes as percentages.
        weights: Optional pre-selected weights (skip selector).
        weights_title: Title for the weights subplot.
        figsize: Figure size in inches.

    Returns:
        matplotlib.figure.Figure: Figure containing the report plot.
    """
    plt = _require_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.4, 1.0]})
    ax_frontier, ax_weights = axes

    selection_key = selection.lower()
    selection_kwargs = dict(selection_kwargs or {})

    selected_weights = weights
    selected_risk = None
    selected_return = None

    if selected_weights is None:
        if selection_key in {"min_risk", "minimum_risk"}:
            selected_weights, selected_return, selected_risk = frontier.get_min_risk_portfolio(
                risk_label=risk_label
            )
        elif selection_key in {"max_return", "maximum_return"}:
            selected_weights, selected_return, selected_risk = frontier.get_max_return_portfolio()
        elif selection_key in {"tangency", "max_sharpe", "sharpe"}:
            rf = float(selection_kwargs.pop("risk_free_rate", 0.0))
            selected_weights, selected_return, selected_risk = frontier.get_tangency_portfolio(rf)
        elif selection_key in {"risk_target", "max_return_subject_to_risk"}:
            max_risk = float(selection_kwargs.pop("max_risk"))
            selected_weights, selected_return, selected_risk = frontier.portfolio_at_risk_target(
                max_risk=max_risk, risk_label=risk_label
            )
        elif selection_key in {"return_target", "min_risk_subject_to_return"}:
            min_return = float(selection_kwargs.pop("min_return"))
            selected_weights, selected_return, selected_risk = frontier.portfolio_at_return_target(
                min_return=min_return, risk_label=risk_label
            )
        elif selection_key in {"risk_percentile", "risk_pct", "percentile"}:
            pct = float(selection_kwargs.pop("percentile"))
            selected_weights, selected_return, selected_risk = frontier.portfolio_at_risk_percentile(
                pct, risk_label=risk_label
            )
        elif selection_key in {"risk_match", "risk_nearest"}:
            target_risk = float(selection_kwargs.pop("target_risk"))
            selected_weights, selected_return, selected_risk = frontier.portfolio_closest_risk(
                target_risk, risk_label=risk_label
            )
        elif selection_key in {"index", "column"}:
            idx = int(selection_kwargs.pop("index"))
            selected_weights = frontier._to_pandas(frontier.weights[:, idx], "Selected")
            selected_return = float(frontier.returns[idx])
            selected_risk = float(frontier._risk_vector(risk_label)[idx])
        else:
            raise ValueError(f"Unknown selection '{selection}'.")

    highlight = (selection_key,) if selection_key in {"min_risk", "max_return", "tangency"} else ()
    plot_frontiers(
        frontier,
        ax=ax_frontier,
        highlight=highlight,
        risk_free_rate=selection_kwargs.get("risk_free_rate"),
        show_points=show_points,
        percent_axes=percent_axes,
        risk_label=risk_label,
        overlay_risk_labels=overlay_risk_labels,
    )
    if selected_risk is not None and selected_return is not None and not highlight:
        ax_frontier.scatter(selected_risk, selected_return, color="black", s=60, zorder=5)

    plot_weights(
        selected_weights,
        ax=ax_weights,
        title=weights_title,
        kind="barh",
        percent_axes=percent_axes,
    )

    fig.tight_layout()
    return fig, axes


__all__ = [
    "plot_frontiers",
    "plot_frontiers_grid",
    "plot_weights",
    "plot_frontier_report",
    "plot_robust_path",
    "plot_param_impact",
    "plot_robust_surface",
    "plot_assumptions_3d",
]

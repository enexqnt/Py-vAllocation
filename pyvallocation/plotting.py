"""Plotting utilities for efficient frontiers and portfolio summaries."""

from __future__ import annotations

from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np

from .portfolioapi import PortfolioFrontier

if TYPE_CHECKING:  # pragma: no cover - typing helper
    import pandas as pd


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - guarded by tests
        raise ImportError(
            "`plot_frontiers` requires `matplotlib`. Install it via `pip install matplotlib`."
        ) from exc
    return plt


def _normalize_frontiers(
    frontiers: Union[PortfolioFrontier, Sequence[PortfolioFrontier], Mapping[str, PortfolioFrontier]],
    labels: Optional[Sequence[str]],
) -> List[Tuple[Optional[str], PortfolioFrontier]]:
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


def _resolve_highlight(
    frontier: PortfolioFrontier,
    marker: str,
    *,
    risk_free_rate: Optional[float],
) -> Optional[Tuple[str, float, float, "pd.Series", int]]:
    if marker == "min_risk":
        idx = int(np.argmin(frontier.risks))
        target_risk = float(frontier.risks[idx])
        target_return = float(frontier.returns[idx])
        weights = frontier._to_pandas(frontier.weights[:, idx], "Min Risk Portfolio")
        name = "Min Risk"
    elif marker == "max_return":
        idx = int(np.argmax(frontier.returns))
        target_risk = float(frontier.risks[idx])
        target_return = float(frontier.returns[idx])
        weights = frontier._to_pandas(frontier.weights[:, idx], "Max Return Portfolio")
        name = "Max Return"
    elif marker == "tangency":
        if risk_free_rate is None:
            raise ValueError("Highlighting the tangency portfolio requires `risk_free_rate`.")
        if np.all(np.isclose(frontier.risks, 0)):
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe_ratios = (frontier.returns - risk_free_rate) / frontier.risks
        sharpe_ratios[~np.isfinite(sharpe_ratios)] = -np.inf
        idx = int(np.argmax(sharpe_ratios))
        if not np.isfinite(frontier.returns[idx]) or not np.isfinite(frontier.risks[idx]):
            return None
        target_risk = float(frontier.risks[idx])
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
        risk_label: Axis label for the risk dimension. If omitted, use the common
            risk measure when all frontiers agree, otherwise default to ``"Risk"``.
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
            if isinstance(value, (float, np.floating)):
                return f"{key}={value:.4f}"
            return f"{key}={value}"
    else:
        _default_metadata_formatter = metadata_value_formatter

    for supplied_label, frontier in normalized:
        label = supplied_label or frontier.risk_measure
        line, = ax.plot(frontier.risks, frontier.returns, label=label, **line_kwargs)
        colour = line.get_color()

        for marker in highlight:
            resolved = _resolve_highlight(frontier, marker, risk_free_rate=risk_free_rate)
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
                    "risk_measure": frontier.risk_measure,
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
    ax.grid(True, alpha=0.3)

    if legend:
        ax.legend()

    ax._frontier_highlights = highlight_records  # type: ignore[attr-defined]
    return ax


__all__ = ["plot_frontiers"]

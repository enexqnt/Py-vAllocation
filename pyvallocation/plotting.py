"""Plotting utilities for efficient frontiers and portfolio summaries."""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union, TYPE_CHECKING

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
) -> Optional[Tuple[str, float, float, "pd.Series"]]:
    if marker == "min_risk":
        weights, target_return, target_risk = frontier.get_min_risk_portfolio()
        name = "Min Risk"
    elif marker == "max_return":
        weights, target_return, target_risk = frontier.get_max_return_portfolio()
        name = "Max Return"
    elif marker == "tangency":
        if risk_free_rate is None:
            raise ValueError("Highlighting the tangency portfolio requires `risk_free_rate`.")
        weights, target_return, target_risk = frontier.get_tangency_portfolio(risk_free_rate)
        name = f"Tangency (rf={risk_free_rate:.2%})"
    else:
        raise ValueError(f"Unknown highlight '{marker}'.")

    if not (np.isfinite(target_return) and np.isfinite(target_risk)):
        return None
    return name, float(target_risk), float(target_return), weights


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

    for supplied_label, frontier in normalized:
        label = supplied_label or frontier.risk_measure
        line, = ax.plot(frontier.risks, frontier.returns, label=label, **line_kwargs)
        colour = line.get_color()

        for marker in highlight:
            resolved = _resolve_highlight(frontier, marker, risk_free_rate=risk_free_rate)
            if resolved is None:
                continue
            display_name, risk_value, return_value, weights = resolved

            style = {
                "color": colour,
                "s": 60,
                "zorder": line.get_zorder() + 1,
            }
            style.update(scatter_kwargs)
            style.update(marker_kwargs.get(marker, {}))

            ax.scatter(risk_value, return_value, label=f"{label} – {display_name}", **style)
            highlight_records.append(
                {
                    "frontier": label,
                    "type": display_name,
                    "risk": risk_value,
                    "return": return_value,
                    "risk_measure": frontier.risk_measure,
                    "weights": weights,
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

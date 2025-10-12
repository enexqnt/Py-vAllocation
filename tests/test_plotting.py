import numpy as np
import pandas as pd
import pytest

from pyvallocation.plotting import plot_frontiers
from pyvallocation.portfolioapi import PortfolioFrontier


def _sample_frontier() -> PortfolioFrontier:
    weights = np.array(
        [
            [0.6, 0.4, 0.2],
            [0.4, 0.6, 0.8],
        ]
    )
    returns = np.array([0.05, 0.07, 0.09])
    risks = np.array([0.08, 0.1, 0.14])
    return PortfolioFrontier(
        weights=weights,
        returns=returns,
        risks=risks,
        risk_measure="Volatility",
        asset_names=["AAA", "BBB"],
    )


def test_portfolio_frontier_to_frame_preserves_names():
    frontier = _sample_frontier()

    full_frame = frontier.to_frame()
    pd.testing.assert_index_equal(full_frame.index, pd.Index(["AAA", "BBB"]))
    assert list(full_frame.columns) == [0, 1, 2]

    custom = frontier.to_frame(columns=[1], column_labels=["Target"])
    pd.testing.assert_index_equal(custom.index, pd.Index(["AAA", "BBB"]))
    assert list(custom.columns) == ["Target"]
    np.testing.assert_allclose(custom.iloc[:, 0], frontier.weights[:, 1])


def test_portfolio_frontier_to_frame_validates_labels_length():
    frontier = _sample_frontier()
    with pytest.raises(ValueError):
        frontier.to_frame(columns=[0, 2], column_labels=["Only one label"])


def test_plot_frontiers_highlight_records_capture_weights():
    plt = pytest.importorskip("matplotlib.pyplot")
    frontier = _sample_frontier()

    ax = plot_frontiers(frontier, highlight=("min_risk", "max_return"))
    highlights = getattr(ax, "_frontier_highlights")

    assert {entry["type"] for entry in highlights} == {"Min Risk", "Max Return"}
    for entry in highlights:
        assert isinstance(entry["weights"], pd.Series)
        pd.testing.assert_index_equal(entry["weights"].index, pd.Index(["AAA", "BBB"]))

    assert ax.get_xlabel() == "Volatility"
    assert ax.get_ylabel() == "Expected Return"

    plt.close(ax.figure)


def test_plot_frontiers_requires_risk_free_for_tangency():
    pytest.importorskip("matplotlib.pyplot")
    frontier = _sample_frontier()
    with pytest.raises(ValueError):
        plot_frontiers(frontier, highlight=("tangency",))


def test_plot_frontiers_metadata_in_highlight_labels():
    plt = pytest.importorskip("matplotlib.pyplot")
    metadata = [
        {"lambda_reg": 0.0, "target_multiplier": None},
        {"lambda_reg": 0.2, "target_multiplier": 1.2},
        {"lambda_reg": 0.2, "target_multiplier": 1.4},
    ]
    frontier = PortfolioFrontier(
        weights=np.array([[0.5, 0.4, 0.3], [0.5, 0.6, 0.7]]),
        returns=np.array([0.04, 0.06, 0.08]),
        risks=np.array([0.07, 0.09, 0.12]),
        risk_measure="Volatility",
        asset_names=["AAA", "BBB"],
        metadata=metadata,
    )

    ax = plot_frontiers(
        frontier,
        highlight=("max_return",),
        highlight_metadata_keys=("target_multiplier", "lambda_reg"),
    )
    _, legend_labels = ax.get_legend_handles_labels()
    joined = " ".join(legend_labels)
    assert "target_multiplier" in joined
    assert "lambda_reg" in joined

    highlights = getattr(ax, "_frontier_highlights")
    assert highlights[0]["metadata"]["target_multiplier"] == 1.4
    assert highlights[0]["index"] == 2

    plt.close(ax.figure)

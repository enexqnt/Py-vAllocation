import numpy as np
import pandas as pd

from pyvallocation.utils.performance import performance_report, scenario_pnl


def test_scenario_pnl_preserves_labels_series():
    scenarios = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.005],
            "B": [0.015, 0.0, -0.01],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    weights = pd.Series({"A": 0.6, "B": 0.4}, name="tangency")
    pnl = scenario_pnl(weights, scenarios)
    assert isinstance(pnl, pd.Series)
    assert pnl.name == "tangency"
    pd.testing.assert_index_equal(pnl.index, scenarios.index)


def test_scenario_pnl_dataframe_multiple_allocations():
    scenarios = pd.DataFrame({"A": [0.02, -0.01], "B": [0.0, 0.01]})
    weights = pd.DataFrame(
        {"w1": [0.5, 0.5], "w2": [0.8, 0.2]},
        index=["A", "B"],
    )
    pnl = scenario_pnl(weights, scenarios)
    assert isinstance(pnl, pd.DataFrame)
    assert pnl.shape == (2, 2)
    pd.testing.assert_index_equal(pnl.index, scenarios.index)
    pd.testing.assert_index_equal(pnl.columns, weights.columns)


def test_performance_report_matches_manual_metrics():
    scenarios = np.array(
        [
            [0.05, -0.04],
            [-0.02, 0.01],
            [0.01, 0.02],
        ]
    )
    weights = np.array([0.4, 0.6])
    pnl = scenarios @ weights
    probs = np.array([0.2, 0.3, 0.5])

    report = performance_report(weights, scenarios, probabilities=probs, alpha=0.95)

    mean_expected = float(np.dot(probs, pnl))
    demeaned = pnl - mean_expected
    stdev_expected = float(np.sqrt(np.dot(probs, demeaned**2)))

    worst_idx = np.argsort(pnl)[0]
    var_expected = -pnl[worst_idx]
    cvar_expected = -np.dot(probs[pnl <= pnl[worst_idx]], pnl[pnl <= pnl[worst_idx]]) / probs[pnl <= pnl[worst_idx]].sum()

    assert np.isclose(report["mean"], mean_expected)
    assert np.isclose(report["stdev"], stdev_expected)
    assert np.isclose(report["VaR95"], var_expected)
    assert np.isclose(report["CVaR95"], cvar_expected)
    assert np.isclose(
        report["ENS"],
        1.0 / np.sum(probs**2),
    )


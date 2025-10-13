import numpy as np
import pandas as pd

from pyvallocation.stress import (
    entropy_pooling_stress,
    exp_decay_stress,
    kernel_focus_stress,
    linear_map,
    stress_test,
)


def _sample_data():
    scenarios = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.015, 0.005, -0.004],
            "B": [0.02, -0.01, 0.0, 0.01, -0.005],
            "C": [-0.005, 0.004, 0.006, -0.003, 0.002],
        }
    )
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2}, name="baseline")
    return scenarios, weights


def test_stress_test_nominal_only():
    R, w = _sample_data()
    df = stress_test(w, R)
    expected_cols = {
        "return_nom",
        "stdev_nom",
        "VaR95_nom",
        "CVaR95_nom",
        "ENS_nom",
    }
    assert set(df.columns) == expected_cols
    assert df.index.tolist() == ["baseline"]


def test_exp_decay_stress_reduces_ens():
    R, w = _sample_data()
    df = exp_decay_stress(w, R, half_life=2)
    assert (df["ENS_stress"] < df["ENS_nom"]).all()
    assert df["KL_q_p"].iloc[0] > 0.0


def test_linear_map_scaling():
    R, w = _sample_data()
    df = stress_test(w, R, transform=linear_map(scale=2.0))
    ratio = df["stdev_stress"] / df["stdev_nom"]
    np.testing.assert_allclose(ratio.values, np.full_like(ratio.values, 2.0), rtol=1e-6)


def test_kernel_focus_probabilities():
    R, w = _sample_data()
    focus = R["A"].abs()
    df = kernel_focus_stress(w, R, focus_series=focus, target=focus.max())
    assert df["ENS_stress"].iloc[0] <= df["ENS_nom"].iloc[0] + 1e-12


def test_entropy_pooling_stress_custom_probabilities():
    R, w = _sample_data()
    posterior = np.array([0.05, 0.1, 0.15, 0.3, 0.4])
    df = entropy_pooling_stress(w, R, posterior_probabilities=posterior)
    np.testing.assert_allclose(df["ENS_stress"], 1 / np.sum(posterior**2))
    assert df["KL_q_p"].iloc[0] > 0.0


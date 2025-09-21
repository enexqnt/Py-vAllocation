import numpy as np
import pandas as pd

from pyvallocation.utils import projection


def test_project_scenarios_defaults_to_uniform_probabilities():
    R = np.ones((4, 2)) * 0.05
    sims = projection.project_scenarios(R, investment_horizon=3, n_simulations=5)
    assert sims.shape == (5, 2)
    np.testing.assert_allclose(sims, 0.15)


def test_project_scenarios_accepts_column_vector_probabilities():
    series = pd.Series([0.01, 0.02, 0.03], name="daily")
    probabilities = np.array([[1.0], [2.0], [3.0]])
    sims = projection.project_scenarios(
        series,
        investment_horizon=1,
        p=probabilities,
        n_simulations=4,
    )
    assert isinstance(sims, pd.Series)
    assert sims.name == "daily"
    assert len(sims) == 4
    assert np.all(sims.isin(series.values))

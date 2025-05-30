import pytest
import numpy as np
import pandas as pd
from pyvallocation.utils.data_helpers import pandas_to_numpy_returns, numpy_weights_to_pandas_series

def test_pandas_to_numpy_returns_log():
    dates = pd.date_range("2020-01-01", periods=3)
    df = pd.DataFrame({"A": [100, 110, 121], "B": [50, 55, 60]}, index=dates)
    R = pandas_to_numpy_returns(df, return_calculation_method="log")
    expected = np.log(np.array([[110/100, 55/50], [121/110, 60/55]]))
    np.testing.assert_allclose(R, expected)

def test_pandas_to_numpy_returns_simple_and_fill_methods():
    dates = pd.date_range("2020-01-01", periods=4)
    df = pd.DataFrame({"A": [100, np.nan, 121, np.nan], "B": [50, 55, np.nan, 66]}, index=dates)
    R = pandas_to_numpy_returns(df, return_calculation_method="simple", fill_na_method="bfill")
    assert R.shape[0] == 3
    assert not np.isnan(R).any()

def test_pandas_to_numpy_returns_invalid_df():
    with pytest.raises(ValueError):
        pandas_to_numpy_returns("not a df")

def test_numpy_weights_to_pandas_series_basic():
    weights = np.array([0.2, 0.8])
    names = ["X", "Y"]
    s = numpy_weights_to_pandas_series(weights, names)
    assert list(s.index) == names
    assert np.allclose(s.values, weights)

def test_numpy_weights_to_pandas_series_invalid_inputs():
    weights = np.array([[0.2, 0.8]])
    with pytest.raises(ValueError):
        numpy_weights_to_pandas_series(weights, ["X", "Y"])
    with pytest.raises(ValueError):
        numpy_weights_to_pandas_series(np.array([0.2, 0.8]), ["X", 1])
    with pytest.raises(ValueError):
        numpy_weights_to_pandas_series(np.array([0.2]), ["X", "Y"])

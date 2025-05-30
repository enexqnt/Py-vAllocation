from typing import TYPE_CHECKING, Optional, List
import numpy as np

if TYPE_CHECKING:
    import pandas as pd

PANDAS_AVAILABLE = False
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pass

def pandas_to_numpy_returns(
    dataframe: 'pd.DataFrame',
    price_columns: Optional[List[str]] = None,
    date_column: Optional[str] = None,
    return_calculation_method: str = 'log',
    fill_na_method: str = 'ffill',
) -> np.ndarray:
    """Convert a pandas DataFrame of prices to a numpy array of returns."""
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for `pandas_to_numpy_returns` function, but it is not installed.")
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("`dataframe` must be a pandas DataFrame.")

    df = dataframe.copy()

    if date_column:
        if df.index.name != date_column:
            if date_column in df.columns:
                try:
                    df[date_column] = pd.to_datetime(df[date_column])
                    df = df.set_index(date_column)
                except Exception as e:
                    raise ValueError(f"Failed to set date column '{date_column}' as index: {e}")
            else:
                raise ValueError(f"Date column '{date_column}' not found in DataFrame columns or as index name.")
        # else: index name matches date_column, no action needed

    if df.index.nlevels > 1:  # Handle MultiIndex if present after set_index
        df = df.reset_index(level=list(range(1, df.index.nlevels)), drop=True)

    if price_columns:
        missing_cols = [col for col in price_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Price columns not found in DataFrame: {missing_cols}")
        price_data = df[price_columns]
    else:
        price_data = df.select_dtypes(include=np.number)
        if price_data.empty:
            raise ValueError("No numeric columns found for price data.")

    if return_calculation_method == 'log':
        returns = np.log(price_data / price_data.shift(1))
    elif return_calculation_method == 'simple':
        returns = price_data.pct_change(fill_method=None)
    else:
        raise ValueError("`return_calculation_method` must be 'log' or 'simple'.")

    returns = returns.iloc[1:]  # First row will be NaN due to shift/pct_change

    if fill_na_method == 'ffill':
        returns = returns.ffill()
    elif fill_na_method == 'bfill':
        returns = returns.bfill()
    elif fill_na_method == 'zero':
        returns = returns.fillna(0)
    elif fill_na_method == 'drop':
        returns = returns.dropna()
    else:
        raise ValueError("`fill_na_method` must be 'ffill', 'bfill', 'zero', or 'drop'.")

    returns = returns.fillna(0)  # Fill any remaining NaNs with 0 after chosen method

    return returns.to_numpy()


def numpy_weights_to_pandas_series(weights: np.ndarray, asset_names: List[str]) -> 'pd.Series':
    """Convert a 1D numpy array of weights to a pandas Series with asset names as index."""
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for `numpy_weights_to_pandas_series` function, but it is not installed.")
    if not isinstance(weights, np.ndarray) or weights.ndim != 1:
        raise ValueError("`weights` must be a 1D NumPy array.")
    if not isinstance(asset_names, list) or not all(isinstance(name, str) for name in asset_names):
        raise ValueError("`asset_names` must be a list of strings.")
    if weights.shape[0] != len(asset_names):
        raise ValueError("Length of `weights` must match length of `asset_names`.")

    return pd.Series(weights, index=asset_names)

.. _data_helpers:

pyvallocation.utils.data\_helpers module
========================================

This module provides helper functions for converting between pandas DataFrames and NumPy arrays,
primarily for handling financial time series data and portfolio weights. These utilities
facilitate seamless data manipulation for various asset allocation and portfolio optimization tasks.

.. automodule:: pyvallocation.utils.data_helpers
   :members:
   :show-inheritance:
   :undoc-members:

Functions
---------

.. autofunction:: pandas_to_numpy_returns
   :no-index:

.. autofunction:: numpy_weights_to_pandas_series
   :no-index:

Examples
--------

Here are some examples demonstrating the usage of the functions in this module.

**1. Converting Pandas DataFrame to NumPy Returns**

This example shows how to convert a DataFrame of asset prices into a NumPy array of returns.

.. code-block:: python

    import pandas as pd
    import numpy as np
    from pyvallocation.utils.data_helpers import pandas_to_numpy_returns

    # Sample DataFrame of prices
    data = {
        'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
        'AssetA': [100, 101, 102, 103],
        'AssetB': [50, 50.5, 51, 51.5]
    }
    df_prices = pd.DataFrame(data)

    # Convert to log returns
    log_returns = pandas_to_numpy_returns(
        dataframe=df_prices,
        date_column='Date',
        price_columns=['AssetA', 'AssetB'],
        return_calculation_method='log'
    )
    print("Log Returns:")
    print(log_returns)

    # Convert to simple returns, filling NaNs with zero
    simple_returns = pandas_to_numpy_returns(
        dataframe=df_prices,
        date_column='Date',
        return_calculation_method='simple',
        fill_na_method='zero'
    )
    print("\nSimple Returns (NaNs filled with zero):")
    print(simple_returns)

**2. Converting NumPy Weights to Pandas Series**

This example demonstrates how to convert a NumPy array of portfolio weights into a pandas Series
with meaningful asset names.

.. code-block:: python

    import numpy as np
    from pyvallocation.utils.data_helpers import numpy_weights_to_pandas_series

    # Sample NumPy array of weights
    weights_array = np.array([0.4, 0.3, 0.2, 0.1])
    asset_names = ['StockA', 'StockB', 'BondC', 'Gold']

    # Convert to pandas Series
    weights_series = numpy_weights_to_pandas_series(
        weights=weights_array,
        asset_names=asset_names
    )
    print("Portfolio Weights Series:")
    print(weights_series)

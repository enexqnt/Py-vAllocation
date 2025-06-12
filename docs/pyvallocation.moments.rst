Statistical Moments Module
==========================

.. automodule:: pyvallocation.moments
   :members:
   :show-inheritance:
   :undoc-members:

Overview
--------

The Moments module provides advanced statistical moment calculation and analysis tools specifically designed for financial portfolio management. It offers comprehensive methods to compute and interpret statistical moments that are crucial for understanding asset and portfolio characteristics.

Key Features
------------

Moment Calculation
^^^^^^^^^^^^^^^^^^

- First moment (mean)
- Second moment (variance)
- Higher-order moments (skewness, kurtosis)
- Moment-based risk analysis

Statistical Techniques
^^^^^^^^^^^^^^^^^^^^^^

- Moment estimation
- Moment-based distribution fitting
- Portfolio moment analysis

Example Usage
-------------

.. code-block:: python

   from pyvallocation.moments import calculate_moments
   from pyvallocation.portfolioapi import Portfolio

   # Create a portfolio
   portfolio = Portfolio()
   portfolio.add_assets(['AAPL', 'GOOGL', 'MSFT'])

   # Calculate portfolio moments
   moments = calculate_moments(portfolio)
   
   # Analyze statistical properties
   mean_return = moments.first_moment
   portfolio_variance = moments.second_moment

Key Moment Metrics
------------------

1. Mean Return
2. Portfolio Variance
3. Skewness
4. Kurtosis
5. Moment-based Risk Measures

Related Modules
---------------

- :doc:`pyvallocation.probabilities`: For probabilistic analysis
- :doc:`pyvallocation.portfolioapi`: For portfolio management
- :doc:`pyvallocation.optimization`: For portfolio optimization using moment insights

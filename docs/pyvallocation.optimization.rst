Portfolio Optimization
======================

.. automodule:: pyvallocation.optimization
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The Optimization module provides advanced techniques for portfolio weight allocation, focusing on maximizing returns while managing risk through various sophisticated strategies.

Optimization Strategies
-----------------------

Mean-Variance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Markowitz Portfolio Theory implementation
- Efficient frontier calculation
- Risk-return trade-off analysis

Advanced Techniques
^^^^^^^^^^^^^^^^^^^

- Black-Litterman model integration
- Robust optimization methods
- Constraint-based allocation

Example Usage
-------------

.. code-block:: python

   from pyvallocation.optimization import optimize_portfolio
   from pyvallocation.portfolioapi import Portfolio

   # Create portfolio
   portfolio = Portfolio()
   portfolio.add_assets(['AAPL', 'GOOGL', 'MSFT'])

   # Optimize portfolio weights
   optimized_weights = optimize_portfolio(
       portfolio, 
       method='mean-variance',
       risk_tolerance=0.5
   )

Key Optimization Methods
------------------------

1. Mean-Variance Optimization
2. Black-Litterman Model
3. Robust Portfolio Optimization
4. Constrained Optimization

Performance Metrics
-------------------

- Expected Return
- Portfolio Variance
- Sharpe Ratio
- Maximum Drawdown

Related Modules
---------------

- :doc:`pyvallocation.portfolioapi`: For portfolio management
- :doc:`pyvallocation.views`: For incorporating investment views
- :doc:`pyvallocation.moments`: For statistical analysis

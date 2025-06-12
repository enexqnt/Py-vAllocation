Portfolio Views
================

.. automodule:: pyvallocation.views
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The Views module provides a flexible framework for incorporating investment perspectives and constraints into portfolio allocation strategies. It allows investors to express their market insights and apply specific investment constraints.

Key Features
------------

- Flexible view specification
- Constraint management
- View-based portfolio adjustment

View Types
----------

Absolute Views
^^^^^^^^^^^^^^

- Direct predictions about asset returns
- Confidence level specification
- Quantitative market insights

Relative Views
^^^^^^^^^^^^^^

- Comparative asset performance expectations
- Sector or asset class comparisons
- Relative value strategies

Example Usage
-------------

.. code-block:: python

   from pyvallocation.views import ViewSpecification
   from pyvallocation.portfolioapi import Portfolio

   # Create a view specification
   view = ViewSpecification()
   
   # Add an absolute view on specific assets
   view.add_absolute_view('AAPL', expected_return=0.1, confidence=0.75)
   
   # Add a relative view between assets
   view.add_relative_view('AAPL', 'GOOGL', relative_return=0.05)

   # Apply views to portfolio optimization
   optimized_portfolio = view.apply_to_portfolio(portfolio)

Constraint Management
---------------------

- Asset allocation limits
- Sector exposure constraints
- Risk-based restrictions
- Liquidity considerations

Advanced View Techniques
------------------------

- Black-Litterman view integration
- Bayesian view adjustment
- Probabilistic view modeling

Related Modules
---------------

- :doc:`pyvallocation.optimization`: For portfolio optimization
- :doc:`pyvallocation.portfolioapi`: For portfolio management
- :doc:`pyvallocation.bayesian`: For probabilistic view modeling

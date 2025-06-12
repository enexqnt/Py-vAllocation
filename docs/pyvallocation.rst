Py-vAllocation Core Module
==========================

The core module provides the foundational classes and functions for portfolio allocation and optimization.
This page provides an overview and links to the submodules.

Submodules
----------

Portfolio Allocation
^^^^^^^^^^^^^^^^^^^^

- :doc:`pyvallocation.portfolioapi`: Core portfolio management functionality
- :doc:`pyvallocation.optimization`: Advanced portfolio optimization techniques
- :doc:`pyvallocation.views`: Flexible views and constraints management

Probabilistic Methods
^^^^^^^^^^^^^^^^^^^^^

- :doc:`pyvallocation.bayesian`: Bayesian portfolio allocation methods
- :doc:`pyvallocation.probabilities`: Probabilistic modeling tools
- :doc:`pyvallocation.moments`: Statistical moments and distribution analysis

Utility Functions
^^^^^^^^^^^^^^^^^

- :doc:`pyvallocation.utils`: Comprehensive utility functions for data handling, validation, and projection

Key Features
------------

- Sophisticated portfolio allocation strategies
- Advanced optimization techniques
- Bayesian and probabilistic modeling
- Flexible views and constraints management
- Robust utility functions for financial data processing

Getting Started
---------------

To begin using Py-vAllocation, import the desired modules and explore the documentation for detailed usage instructions.

.. code-block:: python

   import pyvallocation as pva
   
   # Example of using portfolio API
   portfolio = pva.portfolioapi.Portfolio()
   
   # Optimization example
   optimized_weights = pva.optimization.optimize_portfolio(portfolio)

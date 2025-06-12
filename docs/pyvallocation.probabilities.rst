Probabilistic Methods
=====================

.. automodule:: pyvallocation.probabilities
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The Probabilities module provides advanced probabilistic tools for financial modeling, risk assessment, and portfolio analysis. It offers sophisticated methods to quantify and manage uncertainty in investment strategies.

Probabilistic Techniques
------------------------

Distributional Analysis
^^^^^^^^^^^^^^^^^^^^^^^

- Probability density estimation
- Distribution fitting
- Tail risk analysis

Stochastic Modeling
^^^^^^^^^^^^^^^^^^^

- Monte Carlo simulations
- Scenario generation
- Probabilistic forecasting

Example Usage
-------------

.. code-block:: python

   from pyvallocation.probabilities import ProbabilityAnalysis
   from pyvallocation.portfolioapi import Portfolio

   # Create portfolio
   portfolio = Portfolio()
   portfolio.add_assets(['AAPL', 'GOOGL', 'MSFT'])

   # Perform probabilistic analysis
   prob_analysis = ProbabilityAnalysis(portfolio)
   
   # Generate return scenarios
   scenarios = prob_analysis.generate_scenarios(
       num_simulations=10000,
       confidence_level=0.95
   )

Key Probabilistic Methods
-------------------------

1. Scenario Generation
2. Risk Probability Estimation
3. Distributional Mapping
4. Uncertainty Quantification

Risk Metrics
------------

- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Probability of Drawdown
- Extreme Value Analysis

Related Modules
---------------

- :doc:`pyvallocation.moments`: For statistical moment analysis
- :doc:`pyvallocation.bayesian`: For probabilistic inference
- :doc:`pyvallocation.optimization`: For probabilistic portfolio optimization

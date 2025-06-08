.. _views:

pyvallocation.views module
==========================

This module provides tools for incorporating investor views into asset allocation
frameworks, primarily through entropy pooling and Black-Litterman models. It
supports flexible views on various moments of asset returns.

.. automodule:: pyvallocation.views
   :show-inheritance:
   :undoc-members:

Functions
---------

.. autofunction:: _entropy_pooling_dual_objective

.. autofunction:: _dual_objective

.. autofunction:: entropy_pooling

Classes
-------

.. autoclass:: FlexibleViewsProcessor
   :members:
   :show-inheritance:

.. autoclass:: BlackLittermanProcessor
   :members:
   :show-inheritance:

   .. automethod:: get_posterior
      :no-index:

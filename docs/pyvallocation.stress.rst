Stress Testing Utilities
========================

The :mod:`pyvallocation.stress` module offers user-friendly wrappers around the
single-period scenario engine so allocations can be evaluated under alternative
probability measures or transformed scenarios.

Highlights
----------

- :func:`pyvallocation.stress.stress_test` - master entry point combining probability
  tilts and linear scenario transformations, returning tidy DataFrames.
- :func:`pyvallocation.stress.exp_decay_stress` - historical half-life stress out of the box.
- :func:`pyvallocation.stress.kernel_focus_stress` - Gaussian-kernel focusing on a target regime.
- :func:`pyvallocation.stress.entropy_pooling_stress` - plug posterior probabilities from entropy pooling.
- :func:`pyvallocation.stress.linear_map` - helper to build mean/scale/factor shocks.

The :ref:`stress testing tutorial <stress_testing_tutorial>` demonstrates these
functions alongside the :mod:`pyvallocation.utils.performance` helpers.

Reference
---------

.. automodule:: pyvallocation.stress
   :members:
   :undoc-members:
   :show-inheritance:

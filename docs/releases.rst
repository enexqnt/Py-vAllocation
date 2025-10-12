Release Notes
=============

.. contents::
   :local:
   :depth: 1


Version 0.3.0 (2025-02-25)
--------------------------

- Hardened the discrete allocation engine with iteration safeguards, automatic
  MILP fallback, and richer unit test coverage for lot sizes and failure modes.
- Eliminated ``numpy.matlib`` deprecation noise by wrapping non-linear
  shrinkage imports in targeted warning filters.
- Ran a documentation deep clean: HTML build is warning-free and ``linkcheck``
  now passes thanks to durable DOI handling and updated references.
- Expanded the quickstart tutorial narrative so discrete trade conversion steps
  explain the new fallback behaviour.

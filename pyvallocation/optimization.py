r"""
Portfolio-Optimisation Toolbox
==============================

This package groups three **single-period convex allocation models** and
exposes an identical, numerically stable API for each of them.  All back-end
calls rely exclusively on **CVXOPT 1.3+**; hence global optimality is
guaranteed provided the solver returns a status flag *"optimal"*.

.. list-table::
   :header-rows: 1
   :widths: 18 40 18 18

   * - Acronym
     - Risk measure / Objective functional :math:`f(w)`
     - Cone class
     - CVXOPT routine
   * - ``MV``
     - :math:`\tfrac12\,w^{\top}\Sigma w`
     - PSD
     - - :py:func:`cvxopt.solvers.qp`
   * - ``CVaR``\ :sub:`\alpha`
     - :math:`\operatorname{CVaR}_{\alpha}\bigl(-R\,w\bigr)`
     - LP
     - :py:func:`cvxopt.solvers.conelp`
   * - ``RB``
     - :math:`\displaystyle
       \max_{\mu\in\mathcal U}\bigl[-w^{\top}\mu + \lambda\|S^{1/2}w\|_2\bigr]`
     - SOC
     - :py:func:`cvxopt.solvers.conelp`

Global symbols
--------------
* :math:`N`  -- number of risky assets.
* :math:`T`  -- number of Monte-Carlo or historical scenarios.
* :math:`R\in\mathbb R^{T\times N}` -- scenario matrix of *excess* returns.
* :math:`p\in\Delta^{T}` -- probability vector, :math:`\mathbf1^{\top}p=1`.
* :math:`\mu:=R^{\top}p`,   :math:`\Sigma:=(R-\mu^{\top})^{\top}\!\mathrm{diag}(p)(R-\mu^{\top})`.
* :math:`w\in\mathbb R^{N}` -- portfolio after re-balancing,  
  :math:`\mathbf1^{\top}w=1`.

Optional affine trading rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. math::
   G\,w\;\le\;h, \qquad A\,w\;=\;b,

encode leverage caps, tracking constraints, minimum/maximum holdings, *etc.*

Transaction-cost primitives
---------------------------
* *Quadratic impact* : :math:`\Lambda=\operatorname{diag}(\lambda)`   (QP only)

  .. math:: (w-w_0)^{\top}\Lambda\,(w-w_0).

* *Proportional turnover* : :math:`c^{+},c^{-}\ge0`   (LP/SOCP)

  .. math::
     \sum_{i=1}^{N}\bigl(c^{+}_iu^{+}_i+c^{-}_iu^{-}_i\bigr),
     \qquad
     w = w_0 + u^{+}-u^{-},\;u^{+},u^{-}\ge0.

Primary references
------------------
Markowitz (1952); Rockafellar & Uryasev (2000); Meucci (2005);
Lobo *et al.* (2007).

Credits 
------------------
MeanVariance and MeanCVaR classes are adapted from fortitudo-tech package (https://github.com/fortitudo-tech/fortitudo.tech)

"""

from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Final

import numpy as np
import numpy.typing as npt
from cvxopt import matrix, solvers

from .bayesian import _cholesky_pd

# ------------------------------------------------------------------ #
# Solver settings & logging
# ------------------------------------------------------------------ #
solvers.options.update({"glpk": {"msg_lev": "GLP_MSG_OFF"}, "show_progress": False})
_LOGGER: Final = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _check_shapes(**arrays: npt.NDArray[np.floating]) -> None:
    """Raise ``ValueError`` if any supplied arrays have mismatched shapes."""
    shapes = {k: v.shape for k, v in arrays.items()}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"Shape mismatch: {shapes}")


def _quadratic_turnover(
    P: np.ndarray,
    q: np.ndarray,
    w0: npt.NDArray[np.floating],
    lambdas: npt.NDArray[np.floating],
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Inject the quadratic impact term
    :math:`(w-w_0)^{\top}\Lambda(w-w_0)` into the Hessian/gradient pair
    expected by *CVXOPT*.

    The factor two stems from the *1/2* convention adopted internally by
    :py:func:`cvxopt.solvers.qp`.
    """
    _check_shapes(w0=w0, lambdas=lambdas)
    lambda_diag = np.diag(lambdas)
    return P + 2 * lambda_diag, q - 2 * lambda_diag @ w0


def _linear_turnover_blocks(
    N: int,
    T: int,
    w0: npt.NDArray[np.floating],
    costs: npt.NDArray[np.floating],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Construct the *LP* blocks that realise proportional turnover costs.

    Returns
    -------
    c_cost : np.ndarray
        Objective coefficients for :math:`u^{+},u^{-}` (concatenated ``[c,c]``).
    A_trade : np.ndarray
        Coefficient matrix that enforces the inventory identity
        :math:`w = w_0 + u^{+}-u^{-}`.
    b_trade : np.ndarray
        Right-hand side (simply ``w0``).
    """
    _check_shapes(w0=w0, costs=costs)
    c_cost = np.hstack((costs, costs))
    A_trade = np.concatenate(
        (np.eye(N), np.zeros((N, 1 + T)), -np.eye(N), np.eye(N)), axis=1
    )
    return c_cost, A_trade, w0

# ------------------------------------------------------------------ #
# Result container
# ------------------------------------------------------------------ #
@dataclass(frozen=True)
class OptimizationResult:
    r"""
    Immutable result object returned by :class:`RobustOptimizer`.

    It stores the optimal **post-trade weights**, the associated
    point-estimate return and a model-specific **risk proxy**
    (\sigma* for mean-variance, CVaR* for mean-CVaR, or the robust radius *t**).
    """
    weights: npt.NDArray[np.floating]
    nominal_return: float
    risk: float


@dataclass(frozen=True)
class RelaxedRiskParityResult:
    r"""
    Result container for the relaxed risk parity SOCP.

    Attributes
    ----------
    weights :
        Optimal long-only allocations :math:`x^{\\star}` (shape ``(n,)``).
    marginal_risk :
        Marginal risk vector :math:`\\zeta^{\\star} = \\Sigma x^{\\star}`.
    psi :
        Optimal average-risk proxy :math:`\\psi^{\\star}`.
    gamma :
        Optimal ARC floor :math:`\\gamma^{\\star}`; enforces the lower risk contribution
        bound :math:`x_i \\zeta_i \\ge \\gamma^2`.
    rho :
        Regulator slack variable :math:`\\rho^{\\star}` that upper bounds the diagonal
        risk penalty :math:`\\lambda\\,x^{\\top}\\Theta x`.
    objective :
        Optimal value of :math:`\\psi - \\gamma`, i.e. the gap between the average-risk
        ceiling and the ARC floor.
    target_return :
        Effective return constraint :math:`\\mu^{\\top}x \\ge R` imposed at optimality.
        May differ from the requested target when clipping was required.
    max_return :
        Feasible bound on the return target under the supplied constraints. ``None`` if
        no target was requested.
    """
    weights: npt.NDArray[np.floating]
    marginal_risk: npt.NDArray[np.floating]
    psi: float
    gamma: float
    rho: float
    objective: float
    target_return: float | None
    max_return: float | None

# ------------------------------------------------------------------ #
# Base class
# ------------------------------------------------------------------ #
class _BaseOptimization(ABC):
    r"""
    Abstract helper that stores **data & affine constraints** common to all
    models.

    Implementations must

    1. call :py:func:`_init_constraints` early in ``__init__``;
    2. finish with :py:func:`_finalise_expected_row` (passing ``extra_dims``).

    The latter pre-computes a padded row
    :math:`[-\mu^{\top},\,0,\dots,0]` used by all efficient-frontier routines.
    """

    _I: int
    _mean: np.ndarray
    _G: matrix | None
    _h: matrix | None
    _A: matrix | None
    _b: matrix | None
    _expected_row: matrix

    # ---- init helpers --------------------------------------------- #
    def _init_constraints(
        self,
        mean: npt.ArrayLike,
        G: Optional[npt.ArrayLike],
        h: Optional[npt.ArrayLike],
        A: Optional[npt.ArrayLike],
        b: Optional[npt.ArrayLike],
    ) -> None:
        self._mean = np.asarray(mean, float)
        self._I = self._mean.size
        self._G = matrix(G) if G is not None else None
        self._h = matrix(h) if h is not None else None
        self._A = matrix(A) if A is not None else None
        self._b = matrix(b) if b is not None else None

    def _finalise_expected_row(self, extra: int) -> None:
        """Pad ``-\mu`` with *extra* zeros for later frontier sweeps."""
        self._expected_row = -matrix(
            np.hstack((self._mean, np.zeros(extra, float)))
        ).T

    # ---- utilities ------------------------------------------------ #
    @staticmethod
    def _assert_optimal(sol: dict, kind: str) -> None:
        if sol["status"] != "optimal":
            raise RuntimeError(f"{kind} solver failed (status='{sol['status']}').")

    def _max_expected_return(self) -> float:
        r"""
        Solve a single auxiliary LP to *maximise* :math:`\mu^{\top}w` subject to
        the current constraint set.  Used to anchor the **right end** of every
        efficient frontier.
        """
        sol = solvers.lp(
            self._expected_row.T, self._G, self._h, self._A, self._b, solver="glpk"
        )
        self._assert_optimal(sol, "LP")
        return -sol["primal objective"]

# ------------------------------------------------------------------ #
# Frontier mix-in
# ------------------------------------------------------------------ #
class _FrontierMixin:
    r"""
    Provides :pymeth:`_frontier` - a generic **linear interpolation** between
    the minimum-risk portfolio and the *return-maximiser* obtained from
    :py:func:`_BaseOptimization._max_expected_return`.
    """

    def _frontier(
        self,
        first: npt.NDArray[np.floating],
        fn: callable,
        num: int,
        mean: np.ndarray,
        max_ret: float,
    ) -> np.ndarray:
        min_ret = float(mean @ first)
        if num < 2 or np.isclose(min_ret, max_ret):
            _LOGGER.warning("Frontier collapses to a single point.")
            return first[:, None]
        grid = np.linspace(min_ret, max_ret, num)
        return np.column_stack([first] + [fn(t) for t in grid[1:]])

# =================================================================== #
# 1.  MEAN-VARIANCE
# =================================================================== #
class MeanVariance(_FrontierMixin, _BaseOptimization):
    r"""
    **Classic mean-variance programme** a la Markowitz (1952).

    Problem statement
    -----------------
    .. math::
       \begin{aligned}
         \min_{w}\;&\tfrac12\,w^{\top}\Sigma w
           + \tfrac12\,(w-w_0)^{\top}\Lambda(w-w_0) \\[3pt]
         \text{s.t. }&
           \mu^{\top}w \;\ge\; \tau, \quad
           \mathbf1^{\top}w = 1, \quad
           G w \le h,\; A w = b.
       \end{aligned}

    * ``\tau`` is supplied on the fly via :py:meth:`efficient_portfolio`.
    * ``\Lambda`` (quadratic impact) is optional; if omitted the model degenerates to
      the textbook QP with *no* trading costs.

    Notes
    -----
    The QP is **strictly convex** whenever ``\Sigma`` > 0 or at least one
    positive \lambda_i is present, hence the solution is unique.

    Parameters
    ----------
    mean, covariance :
        Mean vector :math:`\mu` and covariance matrix :math:`\Sigma`.
    G, h, A, b :
        Optional affine constraints as defined in the module docstring.
    initial_weights, market_impact_costs :
        ``w0`` and diag-elements of ``\Lambda`` - must be passed *together*.
    """

    def __init__(
        self,
        mean: npt.ArrayLike,
        covariance: npt.ArrayLike,
        G: Optional[npt.ArrayLike] = None,
        h: Optional[npt.ArrayLike] = None,
        A: Optional[npt.ArrayLike] = None,
        b: Optional[npt.ArrayLike] = None,
        *,
        initial_weights: Optional[npt.NDArray[np.floating]] = None,
        market_impact_costs: Optional[npt.NDArray[np.floating]] = None,
    ):
        self._cov = np.asarray(covariance, float)
        super()._init_constraints(mean, G, h, A, b)

        P, q = 2 * self._cov, np.zeros(self._I)
        if initial_weights is not None and market_impact_costs is not None:
            P, q = _quadratic_turnover(P, q, initial_weights, market_impact_costs)
        self._P, self._q = matrix(P), matrix(q)
        self._finalise_expected_row(0)

    # ---------------------------------------------------------------- #
    # core solver
    # ---------------------------------------------------------------- #
    def _solve_target(self, return_target: float | None = None) -> np.ndarray:
        r"""
        Solve the QP **once** for a given target :math:`\tau`.

        Passing ``None`` yields the *minimum-variance* solution, i.e. the
        left-most point on the efficient frontier.
        """
        G, h = self._G, self._h
        if return_target is not None:
            G = self._expected_row if G is None else matrix([G, self._expected_row])
            h = matrix([-return_target]) if h is None else matrix(
                np.append(np.asarray(h).ravel(), -return_target)
            )
        sol = solvers.qp(self._P, self._q, G, h, self._A, self._b)
        self._assert_optimal(sol, "QP")
        return np.asarray(sol["x"]).ravel()

    efficient_portfolio = _solve_target

    def efficient_frontier(self, num_portfolios: int) -> np.ndarray:
        """
        Return an ``(N, num_portfolios)`` array whose columns trace the Markowitz
        efficient set between the variance minimiser and the return maximiser.
        """
        first = self._solve_target(None)
        max_ret = self._max_expected_return()
        return self._frontier(first, self._solve_target, num_portfolios, self._mean, max_ret)

# =================================================================== #
# 2.  MEAN-CVaR
# =================================================================== #
class MeanCVaR(_FrontierMixin, _BaseOptimization):
    r"""
    **Rockafellar-Uryasev CVaR optimisation** with optional proportional costs.

    Formulation
    -----------
    For level :math:`\alpha\in(0,1)` define the *conditional value-at-risk*

    .. math::
       \operatorname{CVaR}_{\alpha}(Z)=
       \min_{c\in\mathbb R}\;
         c+\frac1\alpha\,\mathbb E[(Z-c)_{+}].

    Substituting :math:`Z=-R\,w` and applying the *sample average*
    approximation produces the *LP*

    .. math::
       \begin{aligned}
       \min_{w,c,\xi}\quad
           & c + \tfrac1\alpha\,p^{\top}\xi
           + c^{\top}(u^{+}+u^{-}) \\[4pt]
       \text{s.t.}\quad
           & \xi \;\ge\; -R\,w - c\mathbf1, \\[2pt]
           & \xi \;\ge\; 0, \\
           & w = w_0 + u^{+}-u^{-},\;u^{+},u^{-}\ge 0, \\
           & \mathbf1^{\top}w = 1,\; G w \le h,\; A w = b.
       \end{aligned}

    Parameters
    ----------
    R, p :
        Scenario matrix and probabilities.
    alpha :
        Tail probability :math:`\alpha` (e.g. ``0.05`` = 95 % CVaR).
    initial_weights, proportional_costs :
        Activate linear turnover frictions *iff* both are given.
    """

    def __init__(
        self,
        R: npt.ArrayLike,
        p: npt.ArrayLike,
        alpha: float,
        G: Optional[npt.ArrayLike] = None,
        h: Optional[npt.ArrayLike] = None,
        A: Optional[npt.ArrayLike] = None,
        b: Optional[npt.ArrayLike] = None,
        *,
        initial_weights: Optional[npt.NDArray[np.floating]] = None,
        proportional_costs: Optional[npt.NDArray[np.floating]] = None,
    ):
        R, p = np.asarray(R, float), np.asarray(p, float)
        T, N = R.shape
        self._alpha = float(alpha)
        self._has_costs = initial_weights is not None and proportional_costs is not None
        super()._init_constraints(p @ R, G, h, A, b)

        base_len = N + 1 + T
        extra_len = 2 * N if self._has_costs else 0
        # objective
        turnover_cost = (
            _linear_turnover_blocks(N, T, initial_weights, proportional_costs)[0]
            if self._has_costs
            else []
        )
        self._c = matrix(np.hstack((np.zeros(N), 1.0, p / alpha, turnover_cost)))

        # --- inequality blocks -------------------------------------- #
        G_blocks, h_blocks = [], []
        # 1) \xi >= -R w - c
        G_blocks += [
            np.concatenate(
                (-R, -np.ones((T, 1)), -np.eye(T), np.zeros((T, extra_len))), axis=1
            ),
            # 2) \xi >= 0
            np.concatenate(
                (np.zeros((T, N + 1)), -np.eye(T), np.zeros((T, extra_len))), axis=1
            ),
        ]
        h_blocks += [np.zeros(T), np.zeros(T)]

        # 3) turnover cost cone u^+,u^- >= 0
        if self._has_costs:
            G_blocks.append(
                np.concatenate((np.zeros((2 * N, base_len)), -np.eye(2 * N)), axis=1)
            )
            h_blocks.append(np.zeros(2 * N))

        # 4) user-supplied G w <= h
        if G is not None:
            G_blocks.append(
                np.concatenate((G, np.zeros((G.shape[0], 1 + T + extra_len))), axis=1)
            )
            h_blocks.append(h)

        self._G, self._h = matrix(np.vstack(G_blocks)), matrix(np.hstack(h_blocks))

        # --- equalities --------------------------------------------- #
        if self._has_costs:
            c_cost, A_trade, b_trade = _linear_turnover_blocks(
                N, T, initial_weights, proportional_costs
            )
            if A is not None:
                self._A = matrix(
                    np.vstack(
                        [
                            np.concatenate((A, np.zeros((A.shape[0], 1 + T + extra_len))), axis=1),
                            A_trade,
                        ]
                    )
                )
                self._b = matrix(np.hstack([b, b_trade]))
            else:
                self._A, self._b = matrix(A_trade), matrix(b_trade)
        elif A is not None:
            self._A = matrix(
                np.concatenate((A, np.zeros((A.shape[0], 1 + T))), axis=1)
            )

        self._finalise_expected_row(1 + T + extra_len)

    # ---------------------------------------------------------------- #
    # core solver
    # ---------------------------------------------------------------- #
    def _solve_target(self, return_target: float | None = None) -> np.ndarray:
        r"""
        Solve the CVaR *LP* for a given target return ``\tau``
        (or *min-CVaR* portfolio if ``\tau is None``).
        """
        G, h = self._G, self._h
        if return_target is not None:
            G = self._expected_row if G is None else matrix([G, self._expected_row])
            h = matrix([-return_target]) if h is None else matrix(
                np.append(np.asarray(h).ravel(), -return_target)
            )
        sol = solvers.lp(self._c, G, h, self._A, self._b, solver="glpk")
        self._assert_optimal(sol, "LP")
        return np.asarray(sol["x"]).ravel()[: self._I]

    efficient_portfolio = _solve_target

    def efficient_frontier(self, num_portfolios: int) -> np.ndarray:
        """
        Return the CVaR efficient frontier with ``num_portfolios`` vertices.
        """
        first = self._solve_target(None)
        max_ret = self._max_expected_return()
        return self._frontier(first, self._solve_target, num_portfolios, self._mean, max_ret)

# =================================================================== #
# 3.  ROBUST MEAN-VARIANCE (SOCP)
# =================================================================== #
class RobustOptimizer(_BaseOptimization):
    r"""
    RobustOptimizer
    ===============

    Single-period **mean-variance allocator** that immunises the portfolio
    against estimation error in the *expected-return vector* while leaving the
    covariance matrix untouched.  The model is the direct implementation of the
    ellipsoidal framework put forward in

    * **Goldfarb & Iyengar** - "Robust Portfolio Selection Problems",
      *Math. Oper. Res.* 28 (1), 1-38 (2003)  
    * **Meucci** - *Risk & Asset Allocation*, Ch. 9 (Springer, 2005) and the
      robust-Bayesian extension in SSRN 681553 (2011)

    ------------------------------------------------------------------------
    1  Ellipsoidal ambiguity set
    ------------------------------------------------------------------------
    We assume the unknown mean :math:`\mu` lies in

    .. math::
       \mathcal U(\hat\mu,S,q)
           := \bigl\{\mu\in\mathbb R^{N}\;|\;
               \lVert S^{-1/2}(\mu-\hat\mu)\rVert_2 \le q\bigr\},

    where  
      * :math:`\hat\mu` -- point estimate (MLE, posterior mean, ...)  
      * :math:`S\succ0` -- scatter matrix (posterior covariance, shrinkage, ...)  
      * :math:`q` -- radius, usually :math:`\sqrt{\chi^2_N(1-\alpha)}` for a
        :math:`100(1-\alpha)\%` credible set.

    ------------------------------------------------------------------------
    2  SOCP reformulation
    ------------------------------------------------------------------------
    Goldfarb-Iyengar (Thm 3.1) show

    .. math::
       \min_{\mu\in\mathcal U}w^{\top}\mu
         = \hat\mu^{\top}w-q\,\lVert S^{1/2}w\rVert_2,

    so the worst-case mean is a *linear* term minus a 2-norm penalty.  Introducing
    an epigraph variable :math:`t` gives the cone programme

    .. math::
       \min_{w,t}\;t+\lambda\lVert S^{1/2}w\rVert_2
       \;\;\text{s.t.}\;\;t\ge-\hat\mu^{\top}w,\;w\!\in\!C,

    which CVXOPT solves as a **second-order cone programme** (type `"q"`).

    ------------------------------------------------------------------------
    3. Two parameterisations
    ------------------------------------------------------------------------
    ``solve_lambda_variant(lam)``  
      Direct penalty :math:`\lambda=q`.  Higher \lambda => stronger shrinkage towards
      the global minimum-variance portfolio.

    ``solve_gamma_variant(gamma_mu, gamma_sigma_sq)``  
      Chance-constraint form (Ben-Tal & Nemirovski 2001).  For tolerance
      :math:`\gamma_\mu` and radius cap :math:`\gamma_{\sigma}^{2}` we enforce

      .. math::
         \Pr(\mu^{\top}w\le -t)\le\gamma_\mu,\quad t^2\le\gamma_{\sigma}^{2},

      implemented as a linear row ``t <= sqrt(gamma_sigma_sq)``.

    Both wrappers feed the same private routine :py:meth:`_solve_socp`.

    ------------------------------------------------------------------------
    4. Optional proportional turnover
    ------------------------------------------------------------------------
    Passing *both* ``initial_weights`` *and* ``proportional_costs`` activates the
    linear-cost mechanism of Lobo et al. (2007).  The decision vector becomes

    .. math:: (w,\;t,\;u^{+},u^{-})\in\mathbb R^{\,3N+1},

    with inventory balance  
    :math:`w=w_0+u^{+}-u^{-},\;u^{+},u^{-}\ge0`.

    ------------------------------------------------------------------------
    5. Limitations
    ------------------------------------------------------------------------
    * Only mean uncertainty is modelled; covariance risk would require an SDP.  
    * The ellipsoid is static; time-varying radii must be supplied upstream.  
    * Extremely ill-conditioned ``uncertainty_cov`` can trigger numerical
      warnings in CVXOPT.

    """

    def __init__(
        self,
        expected_return: npt.NDArray[np.floating],
        uncertainty_cov: npt.NDArray[np.floating],
        G: Optional[npt.ArrayLike] = None,
        h: Optional[npt.ArrayLike] = None,
        A: Optional[npt.ArrayLike] = None,
        b: Optional[npt.ArrayLike] = None,
        *,
        initial_weights: Optional[npt.NDArray[np.floating]] = None,
        proportional_costs: Optional[npt.NDArray[np.floating]] = None,
    ):
        super()._init_constraints(expected_return, G, h, A, b)
        self._s_sqrt = _cholesky_pd(np.asarray(uncertainty_cov, float))
        self._w0, self._costs = initial_weights, proportional_costs
        self._has_costs = initial_weights is not None and proportional_costs is not None
        if self._has_costs:
            _check_shapes(initial_weights=initial_weights, proportional_costs=proportional_costs)

    # ---------------------------------------------------------------- #
    # public wrappers
    # ---------------------------------------------------------------- #
    def solve_lambda_variant(self, lam: float) -> OptimizationResult:
        r"""
        Solve the *\lambda-variant*:

        .. math::
           \min_{w,t}\;t + \lambda\|S^{1/2}w\|_2
           \quad\text{s.t.}\;
           t \ge -\hat\mu^{\top}w,\;\ldots
        """
        if lam < 0:
            raise ValueError("\lambda must be non-negative.")
        return self._solve_socp(lam=lam)

    def solve_gamma_variant(self, gamma_mu: float, gamma_sigma_sq: float) -> OptimizationResult:
        r"""
        Solve the *chance-constraint* form (\gamma-variant).  Arguments map onto

        .. math::
           \Pr\bigl(\mu^{\top}w\le -t\bigr)\;\le\; \gamma_{\mu}, \qquad
           t\;\le\;\sqrt{\gamma_{\sigma}^{2}}.
        """
        if gamma_mu < 0 or gamma_sigma_sq < 0:
            raise ValueError("\gamma must be non-negative.")
        return self._solve_socp(gamma_mu=gamma_mu, gamma_sigma_sq=gamma_sigma_sq)

    def efficient_frontier(
        self, lambdas: Sequence[float]
    ) -> Tuple[list[float], list[float], npt.NDArray[np.floating]]:
        """
        Sweep a list of ``lambdas`` and return

        * nominal returns,
        * robust radii (``t*``),
        * and a weight matrix ``(N, len(lambdas))``.
        """
        res = [self.solve_lambda_variant(l) for l in lambdas]
        return (
            [r.nominal_return for r in res],
            [r.risk for r in res],
            np.column_stack([r.weights for r in res]),
        )

    # ---------------------------------------------------------------- #
    # core SOCP
    # ---------------------------------------------------------------- #
    def _solve_socp(self, **kw) -> OptimizationResult:
        r"""
        Internal driver - constructs and solves the *conic* form shared by
        both \lambda- and \gamma-variants.  Never call directly.
        """
        n = self._I
        extra = 2 * n if self._has_costs else 0
        n_vars = n + 1 + extra            #  w | t | (u^+,u^-)
        penalty = kw.get("lam", kw.get("gamma_mu"))

        c = np.hstack(
            (-self._mean, penalty, (self._costs, self._costs) if self._has_costs else [])
        )

        # --- SOC:  ||S^{1/2}w||_2 <= t  ------------------------------- #
        G_soc = np.zeros((n + 1, n_vars))
        G_soc[0, n] = -1
        G_soc[1:, :n] = -self._s_sqrt
        h_soc = np.zeros(n + 1)

        # --- linear inequalities ----------------------------------- #
        if self._has_costs:
            G_user = (
                np.concatenate((np.asarray(self._G), np.zeros((self._G.size[0], 1 + extra))), axis=1)
                if self._G is not None
                else np.empty((0, n_vars))
            )
            G_lin = np.vstack(
                [G_user, np.concatenate((np.zeros((extra, n + 1)), -np.eye(extra)), axis=1)]
            )
            h_lin = (
                np.hstack([np.asarray(self._h).ravel(), np.zeros(extra)])
                if self._h is not None
                else np.zeros(G_lin.shape[0])
            )
        else:
            G_lin = (
                np.concatenate((np.asarray(self._G), np.zeros((self._G.size[0], 1))), axis=1)
                if self._G is not None
                else np.empty((0, n_vars))
            )
            h_lin = np.asarray(self._h).ravel() if self._h is not None else np.zeros(G_lin.shape[0])

        # optional *gamma_sigma_sq* translates to an upper bound on *t*
        if "gamma_sigma_sq" in kw:
            G_lin = np.vstack([G_lin, np.eye(1, n_vars, n)])
            h_lin = np.hstack([h_lin, np.sqrt(kw["gamma_sigma_sq"])])

        # --- equalities -------------------------------------------- #
        if self._has_costs:
            A_trade = np.concatenate(
                (np.eye(n), np.zeros((n, 1)), -np.eye(n), np.eye(n)), axis=1
            )
            b_trade = self._w0
            if self._A is not None:
                A_mat = np.vstack(
                    [
                        np.concatenate(
                            (np.asarray(self._A), np.zeros((self._A.size[0], 1 + extra))), axis=1
                        ),
                        A_trade,
                    ]
                )
                b_vec = np.hstack([np.asarray(self._b).ravel(), b_trade])
            else:
                A_mat, b_vec = A_trade, b_trade
        else:
            A_mat, b_vec = (
                np.concatenate((np.asarray(self._A), np.zeros((self._A.size[0], 1))), axis=1),
                np.asarray(self._b).ravel(),
            ) if self._A is not None else (None, None)

        sol = solvers.conelp(
            matrix(c),
            matrix([matrix(G_lin), matrix(G_soc)]),
            matrix([matrix(h_lin), matrix(h_soc)]),
            dims={"l": int(G_lin.shape[0]), "q": [n + 1], "s": []},
            A=matrix(A_mat) if A_mat is not None else None,
            b=matrix(b_vec) if b_vec is not None else None,
        )
        self._assert_optimal(sol, "SOCP")
        x = np.asarray(sol["x"]).ravel()
        w, t = x[:n], x[n]
        return OptimizationResult(w, float(self._mean @ w), float(t))


# =================================================================== #
# 4.  RELAXED RISK PARITY (SOCP)
# =================================================================== #
class RelaxedRiskParity(_BaseOptimization):
    r"""
    Implement the relaxed risk parity model of Gambeta & Kwon (2020).

    The decision vector is ordered as

    .. code-block:: text

        [ x (n) | \zeta (n) | \psi | \gamma | \rho | q ],

    with ``q`` acting as the auxiliary variable that nests the average-risk
    constraint into standard SOC blocks. Long-only holdings, non-negative
    marginal risks, and the regulator variable are enforced explicitly.

    The formulation minimises :math:`\\psi - \\gamma` subject to:

    * marginal risk consistency :math:`\\zeta = \\Sigma x`;
    * budget constraint :math:`\\mathbf{1}^{\\top}x = 1`;
    * ARC floor :math:`x_i\\zeta_i \\ge \\gamma^2` (realised via rotated second-order cones);
    * regulated average risk :math:`\\|Lx\\|_2 \\le q`, :math:`\\|(q,\\sqrt{n}\\rho)\\|_2 \\le \\sqrt{n}\\psi`;
    * diagonal penalty :math:`\\sqrt{\\lambda}\\,\\|D x\\|_2 \\le \\rho` whenever :math:`\\lambda > 0`;
    * optional return target :math:`\\mu^{\\top}x \\ge R`.

    All conic blocks are expressed in a solver-ready format compatible with
    ``cvxopt.solvers.conelp``.
    """

    def __init__(
        self,
        mean: npt.ArrayLike,
        covariance: npt.ArrayLike,
        G: Optional[npt.ArrayLike] = None,
        h: Optional[npt.ArrayLike] = None,
        A: Optional[npt.ArrayLike] = None,
        b: Optional[npt.ArrayLike] = None,
    ):
        cov = np.asarray(covariance, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("`covariance` must be a square matrix.")
        cov = 0.5 * (cov + cov.T)

        super()._init_constraints(mean, G, h, A, b)
        if cov.shape[0] != self._I:
            raise ValueError("`mean` and `covariance` dimensions are inconsistent.")

        self._cov = cov
        self._chol = _cholesky_pd(cov)
        diag = np.clip(np.diag(cov), a_min=0.0, a_max=None)
        self._theta_sqrt = np.sqrt(diag, dtype=float)
        self._finalise_expected_row(self._I + 4)

    # ------------------------------------------------------------------ #
    # core SOCP
    # ------------------------------------------------------------------ #
    def solve(
        self,
        *,
        lambda_reg: float = 0.0,
        return_target: float | None = None,
        min_target: float | None = None,
    ) -> RelaxedRiskParityResult:
        r"""
        Solve the relaxed risk parity SOCP for a given ``lambda_reg`` and
        optional target return ``return_target``.
        """
        if lambda_reg < 0:
            raise ValueError("`lambda_reg` must be non-negative.")
        if return_target is not None and (not np.isfinite(return_target)):
            raise ValueError("`return_target` must be a finite scalar.")

        n = self._I
        x_slice = slice(0, n)
        zeta_slice = slice(n, 2 * n)
        psi_idx = 2 * n
        gamma_idx = 2 * n + 1
        rho_idx = 2 * n + 2
        q_idx = 2 * n + 3
        n_vars = 2 * n + 4

        G_user, h_user, A_user, b_user = self._extract_user_constraints()

        target_used = return_target
        max_return_value: float | None = None
        if return_target is not None:
            max_return_value = self._max_linear_return(G_user, h_user, A_user, b_user)
            tolerance = max(1e-8, 1e-3 * abs(max_return_value))
            if target_used >= max_return_value - tolerance:
                target_used = max_return_value - tolerance
            if min_target is not None and target_used is not None:
                target_used = max(target_used, min_target)

        c = np.zeros(n_vars, dtype=float)
        c[psi_idx] = 1.0
        c[gamma_idx] = -1.0

        # --- equalities ------------------------------------------------ #
        A_rows: list[np.ndarray] = []
        b_rows: list[np.ndarray] = []

        A_rows.append(
            np.hstack(
                (-self._cov, np.eye(n, dtype=float), np.zeros((n, 4), dtype=float))
            )
        )
        b_rows.append(np.zeros(n, dtype=float))

        add_budget_row = True
        if A_user is not None:
            ones = np.ones(n, dtype=float)
            for row, rhs in zip(A_user, b_user):
                if np.allclose(row, ones, atol=1e-8, rtol=1e-8):
                    add_budget_row = False
                    break

        if add_budget_row:
            budget_row = np.zeros((1, n_vars), dtype=float)
            budget_row[0, x_slice] = 1.0
            A_rows.append(budget_row)
            b_rows.append(np.array([1.0], dtype=float))

        if A_user is not None and A_user.size:
            pad = np.zeros((A_user.shape[0], n + 4), dtype=float)
            A_rows.append(np.hstack((A_user, pad)))
            b_rows.append(b_user)

        A_mat = np.vstack(A_rows)
        b_vec = np.hstack(b_rows)

        # --- linear inequalities -------------------------------------- #
        G_blocks: list[np.ndarray] = []
        h_blocks: list[np.ndarray] = []

        G_blocks.append(
            np.hstack(
                (-np.eye(n, dtype=float), np.zeros((n, n + 4), dtype=float))
            )
        )
        h_blocks.append(np.zeros(n, dtype=float))

        G_blocks.append(
            np.hstack(
                (np.zeros((n, n), dtype=float), -np.eye(n, dtype=float), np.zeros((n, 4), dtype=float))
            )
        )
        h_blocks.append(np.zeros(n, dtype=float))

        for idx in (psi_idx, gamma_idx, rho_idx):
            row = np.zeros((1, n_vars), dtype=float)
            row[0, idx] = -1.0
            G_blocks.append(row)
            h_blocks.append(np.zeros(1, dtype=float))

        if G_user is not None and G_user.size:
            pad = np.zeros((G_user.shape[0], n + 4), dtype=float)
            G_blocks.append(np.hstack((G_user, pad)))
            h_blocks.append(h_user)

        if target_used is not None:
            G_ret = np.zeros((1, n_vars), dtype=float)
            G_ret[0, x_slice] = -self._mean
            G_blocks.append(G_ret)
            h_blocks.append(np.array([-target_used], dtype=float))

        if G_blocks:
            G_lin = np.vstack(G_blocks)
            h_lin = np.hstack(h_blocks)
        else:
            G_lin = np.empty((0, n_vars), dtype=float)
            h_lin = np.empty(0, dtype=float)

        # --- SOC blocks ----------------------------------------------- #
        soc_dims: list[int] = []
        G_soc_all: list[np.ndarray] = []
        h_soc_all: list[np.ndarray] = []

        G1 = np.zeros((n + 1, n_vars), dtype=float)
        G1[0, q_idx] = -1.0
        G1[1:, x_slice] = -self._chol
        G_soc_all.append(G1)
        h_soc_all.append(np.zeros(n + 1, dtype=float))
        soc_dims.append(n + 1)

        sqrt_n = float(np.sqrt(n))
        G2 = np.zeros((3, n_vars), dtype=float)
        G2[0, psi_idx] = -sqrt_n
        G2[1, q_idx] = -1.0
        G2[2, rho_idx] = -sqrt_n
        G_soc_all.append(G2)
        h_soc_all.append(np.zeros(3, dtype=float))
        soc_dims.append(3)

        if lambda_reg > 0.0:
            G3 = np.zeros((n + 1, n_vars), dtype=float)
            G3[0, rho_idx] = -1.0
            coeffs = np.diag(np.sqrt(lambda_reg) * self._theta_sqrt, k=0)
            G3[1:, x_slice] = -coeffs
            G_soc_all.append(G3)
            h_soc_all.append(np.zeros(n + 1, dtype=float))
            soc_dims.append(n + 1)

        for i in range(n):
            block = np.zeros((3, n_vars), dtype=float)
            block[0, x_slice.start + i] = -1.0
            block[0, zeta_slice.start + i] = -1.0
            block[1, gamma_idx] = -2.0
            block[2, x_slice.start + i] = -1.0
            block[2, zeta_slice.start + i] = 1.0
            G_soc_all.append(block)
            h_soc_all.append(np.zeros(3, dtype=float))
            soc_dims.append(3)

        dims = {"l": int(G_lin.shape[0]), "q": soc_dims, "s": []}
        G_total = (
            np.vstack([G_lin, *G_soc_all])
            if G_lin.size
            else np.vstack(G_soc_all)
        )
        h_total = (
            np.hstack([h_lin, *h_soc_all])
            if h_lin.size
            else np.hstack(h_soc_all)
        )

        sol = solvers.conelp(
            matrix(c),
            matrix(G_total),
            matrix(h_total),
            dims=dims,
            A=matrix(A_mat),
            b=matrix(b_vec),
        )
        self._assert_optimal(sol, "SOCP")

        x_opt = np.asarray(sol["x"], dtype=float).ravel()
        weights = x_opt[x_slice]
        marginal_risk = x_opt[zeta_slice]
        psi = float(x_opt[psi_idx])
        gamma = float(x_opt[gamma_idx])
        rho = float(x_opt[rho_idx])

        return RelaxedRiskParityResult(
            weights=weights,
            marginal_risk=marginal_risk,
            psi=psi,
            gamma=gamma,
            rho=rho,
            objective=psi - gamma,
            target_return=target_used,
            max_return=max_return_value,
        )
    def _extract_user_constraints(
        self,
    ) -> tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Return user-supplied linear constraint matrices as NumPy arrays."""
        n = self._I
        G_user: Optional[np.ndarray] = None
        h_user: Optional[np.ndarray] = None
        if self._G is not None:
            G_user = np.asarray(self._G, dtype=float)
            if G_user.ndim != 2:
                raise ValueError("`G` must be a 2D array.")
            if G_user.shape[1] != n:
                raise ValueError("`G` must have `n` columns.")
            h_user = np.asarray(self._h, dtype=float).ravel()
            if h_user.size != G_user.shape[0]:
                raise ValueError("`h` dimension mismatch.")

        A_user: Optional[np.ndarray] = None
        b_user: Optional[np.ndarray] = None
        if self._A is not None:
            A_user = np.asarray(self._A, dtype=float)
            if A_user.ndim != 2:
                raise ValueError("`A` must be a 2D array.")
            if A_user.shape[1] != n:
                raise ValueError("`A` must have `n` columns.")
            b_user = np.asarray(self._b, dtype=float).ravel()
            if b_user.size != A_user.shape[0]:
                raise ValueError("`b` dimension mismatch.")

        return G_user, h_user, A_user, b_user

    def _max_linear_return(
        self,
        G_user: Optional[np.ndarray],
        h_user: Optional[np.ndarray],
        A_user: Optional[np.ndarray],
        b_user: Optional[np.ndarray],
    ) -> float:
        """Solve a linear program to compute the maximum attainable return."""
        n = self._I
        G_lp_rows: list[np.ndarray] = []
        h_lp_vals: list[np.ndarray] = []
        if G_user is not None and G_user.size:
            G_lp_rows.append(G_user)
            h_lp_vals.append(h_user)
        G_lp_rows.append(-np.eye(n, dtype=float))
        h_lp_vals.append(np.zeros(n, dtype=float))
        G_lp = np.vstack(G_lp_rows)
        h_lp = np.hstack(h_lp_vals)
        if A_user is None:
            A_lp = np.ones((1, n), dtype=float)
            b_lp = np.array([1.0], dtype=float)
        else:
            A_lp, b_lp = A_user, b_user

        sol_lp = solvers.lp(
            matrix(-self._mean),
            matrix(G_lp),
            matrix(h_lp),
            matrix(A_lp),
            matrix(b_lp),
            solver="glpk",
        )
        self._assert_optimal(sol_lp, "LP")
        return -sol_lp["primal objective"]

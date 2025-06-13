r"""
Portfolio-Optimisation Toolbox
==============================

This package groups three **single-period convex allocation models** and
exposes an identical, numerically stable API for each of them.  All back-end
calls rely exclusively on **CVXOPT 1.3+**; hence global optimality is
guaranteed provided the solver returns a status flag *“optimal”*.

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
* :math:`N`  — number of risky assets.
* :math:`T`  — number of Monte-Carlo or historical scenarios.
* :math:`R\in\mathbb R^{T\times N}` — scenario matrix of *excess* returns.
* :math:`p\in\Delta^{T}` — probability vector, :math:`\mathbf1^{\top}p=1`.
* :math:`\mu:=R^{\top}p`,   :math:`\Sigma:=(R-\mu^{\top})^{\top}\!\mathrm{diag}(p)(R-\mu^{\top})`.
* :math:`w\in\mathbb R^{N}` — portfolio after re-balancing,  
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

    The factor two stems from the *½* convention adopted internally by
    :pyfunc:`cvxopt.solvers.qp`.
    """
    _check_shapes(w0=w0, lambdas=lambdas)
    Λ = np.diag(lambdas)
    return P + 2 * Λ, q - 2 * Λ @ w0


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
    (σ⋆ for mean–variance, CVaR⋆ for mean-CVaR, or the robust radius *t⋆*).
    """
    weights: npt.NDArray[np.floating]
    nominal_return: float
    risk: float

# ------------------------------------------------------------------ #
# Base class
# ------------------------------------------------------------------ #
class _BaseOptimization(ABC):
    r"""
    Abstract helper that stores **data & affine constraints** common to all
    models.

    Implementations must

    1. call :pyfunc:`_init_constraints` early in ``__init__``;
    2. finish with :pyfunc:`_finalise_expected_row(extra_dims)`.

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
        """Pad ``-μ`` with *extra* zeros for later frontier sweeps."""
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
    Provides :pymeth:`_frontier` – a generic **linear interpolation** between
    the minimum-risk portfolio and the *return-maximiser* obtained from
    :pyfunc:`_BaseOptimization._max_expected_return`.
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
# 1.  MEAN–VARIANCE
# =================================================================== #
class MeanVariance(_FrontierMixin, _BaseOptimization):
    r"""
    **Classic mean–variance programme** à la Markowitz (1952).

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

    * ``τ`` is supplied on the fly via :py:meth:`efficient_portfolio`.
    * ``Λ`` (quadratic impact) is optional; if omitted the model degenerates to
      the textbook QP with *no* trading costs.

    Notes
    -----
    The QP is **strictly convex** whenever ``Σ`` ≻ 0 or at least one
    positive λᵢ is present, hence the solution is unique.

    Parameters
    ----------
    mean, covariance :
        Mean vector :math:`\mu` and covariance matrix :math:`\Sigma`.
    G, h, A, b :
        Optional affine constraints as defined in the module docstring.
    initial_weights, market_impact_costs :
        ``w0`` and diag-elements of ``Λ`` – must be passed *together*.
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
# 2.  MEAN–CVaR
# =================================================================== #
class MeanCVaR(_FrontierMixin, _BaseOptimization):
    r"""
    **Rockafellar–Uryasev CVaR optimisation** with optional proportional costs.

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
        # 1) ξ ≥ -R w - c
        G_blocks += [
            np.concatenate(
                (-R, -np.ones((T, 1)), -np.eye(T), np.zeros((T, extra_len))), axis=1
            ),
            # 2) ξ ≥ 0
            np.concatenate(
                (np.zeros((T, N + 1)), -np.eye(T), np.zeros((T, extra_len))), axis=1
            ),
        ]
        h_blocks += [np.zeros(T), np.zeros(T)]

        # 3) turnover cost cone u⁺,u⁻ ≥ 0
        if self._has_costs:
            G_blocks.append(
                np.concatenate((np.zeros((2 * N, base_len)), -np.eye(2 * N)), axis=1)
            )
            h_blocks.append(np.zeros(2 * N))

        # 4) user-supplied G w ≤ h
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
        Solve the CVaR *LP* for a given target return ``τ``
        (or *min-CVaR* portfolio if ``τ is None``).
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
# 3.  ROBUST MEAN–VARIANCE (SOCP)
# =================================================================== #
class RobustOptimizer(_BaseOptimization):
    r"""
    RobustOptimizer
    ===============

    Single-period **mean–variance allocator** that immunises the portfolio
    against estimation error in the *expected-return vector* while leaving the
    covariance matrix untouched.  The model is the direct implementation of the
    ellipsoidal framework put forward in

    * **Goldfarb & Iyengar** – “Robust Portfolio Selection Problems”,
      *Math. Oper. Res.* 28 (1), 1-38 (2003)  
    * **Meucci** – *Risk & Asset Allocation*, Ch. 9 (Springer, 2005) and the
      robust-Bayesian extension in SSRN 681553 (2011)

    ------------------------------------------------------------------------
    1  Ellipsoidal ambiguity set
    ------------------------------------------------------------------------
    We assume the unknown mean :math:`\mu` lies in

    .. math::
       \mathcal U(\hat\mu,S,q)
           := \bigl\{\mu\in\mathbb R^{N}\;|\;
               \lVert S^{-1/2}(\mu-\hat\mu)\rVert_2 \le q\bigr\},

    where  
      * :math:`\hat\mu` — point estimate (MLE, posterior mean, …)  
      * :math:`S\succ0` — scatter matrix (posterior covariance, shrinkage, …)  
      * :math:`q` — radius, usually :math:`\sqrt{\chi^2_N(1-\alpha)}` for a
        :math:`100(1-\alpha)\%` credible set.

    ------------------------------------------------------------------------
    2  SOCP reformulation
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
    3  Two parameterisations
    ------------------------------------------------------------------------
    ``solve_lambda_variant(lam)``  
      Direct penalty :math:`\lambda=q`.  Higher λ ⇒ stronger shrinkage towards
      the global minimum-variance portfolio.

    ``solve_gamma_variant(gamma_mu, gamma_sigma_sq)``  
      Chance-constraint form (Ben-Tal & Nemirovski 2001).  For tolerance
      :math:`\gamma_\mu` and radius cap :math:`\gamma_{\sigma}^{2}` we enforce

      .. math::
         \Pr(\mu^{\top}w\le -t)\le\gamma_\mu,\quad t^2\le\gamma_{\sigma}^{2},

      implemented as a linear row ``t ≤ √gamma_sigma_sq``.

    Both wrappers feed the same private routine :py:meth:`_solve_socp`.

    ------------------------------------------------------------------------
    4  Optional proportional turnover
    ------------------------------------------------------------------------
    Passing *both* ``initial_weights`` *and* ``proportional_costs`` activates the
    linear-cost mechanism of Lobo et al. (2007).  The decision vector becomes

    .. math:: (w,\;t,\;u^{+},u^{-})\in\mathbb R^{\,3N+1},

    with inventory balance  
    :math:`w=w_0+u^{+}-u^{-},\;u^{+},u^{-}\ge0`.

    ------------------------------------------------------------------------
    5  Limitations
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
        Solve the *λ-variant*:

        .. math::
           \min_{w,t}\;t + λ\|S^{1/2}w\|_2
           \quad\text{s.t.}\;
           t \ge -\hat\mu^{\top}w,\;\ldots
        """
        if lam < 0:
            raise ValueError("λ must be non-negative.")
        return self._solve_socp(lam=lam)

    def solve_gamma_variant(self, gamma_mu: float, gamma_sigma_sq: float) -> OptimizationResult:
        r"""
        Solve the *chance-constraint* form (γ-variant).  Arguments map onto

        .. math::
           \Pr\bigl(\mu^{\top}w\le -t\bigr)\;\le\; γ_{\mu}, \qquad
           t\;\le\;\sqrt{γ_{\sigma}^{2}}.
        """
        if gamma_mu < 0 or gamma_sigma_sq < 0:
            raise ValueError("γ must be non-negative.")
        return self._solve_socp(gamma_mu=gamma_mu, gamma_sigma_sq=gamma_sigma_sq)

    def efficient_frontier(
        self, lambdas: Sequence[float]
    ) -> Tuple[list[float], list[float], npt.NDArray[np.floating]]:
        """
        Sweep a list of ``lambdas`` and return

        * nominal returns,
        * robust radii (``t⋆``),
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
        Internal driver – constructs and solves the *conic* form shared by
        both λ- and γ-variants.  Never call directly.
        """
        n = self._I
        extra = 2 * n if self._has_costs else 0
        n_vars = n + 1 + extra            #  w | t | (u⁺,u⁻)
        penalty = kw.get("lam", kw.get("gamma_mu"))

        c = np.hstack(
            (-self._mean, penalty, (self._costs, self._costs) if self._has_costs else [])
        )

        # --- SOC:  ‖S^{1/2}w‖_2 ≤ t  ------------------------------- #
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

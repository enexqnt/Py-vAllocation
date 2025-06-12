"""
This module provides classes for solving various portfolio optimization problems,
including Mean-Variance, Mean-CVaR, and Robust Optimization.

It leverages `cvxopt` for solving quadratic programming (QP) and conic
programming (SOCP) problems, and integrates with Bayesian methods for robust
estimation.

Classes:

* `OptimizationResult`: A dataclass to hold the results of an optimization.
* `Optimization`: A base class providing common optimization utilities.
* `MeanVariance`: Implements classical Mean-Variance portfolio optimization.
* `MeanCVaR`: Implements Mean-Conditional Value-at-Risk portfolio optimization.
* `RobustOptimizer`: Implements robust portfolio optimization based on
    uncertainty sets for mean and covariance.
"""

from __future__ import annotations

import logging
import numbers
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from cvxopt import matrix, solvers

from .bayesian import _cholesky_pd

solvers.options.update({"glpk": {"msg_lev": "GLP_MSG_OFF"}, "show_progress": False})

logger = logging.getLogger(__name__)

@dataclass(slots=True, frozen=True)
class OptimizationResult:
    """
    A dataclass to store the results of a portfolio optimization.

    :ivar weights: The optimal portfolio weights.
    :vartype weights: npt.NDArray[np.floating]
    :ivar nominal_return: The expected return of the portfolio based on the nominal (point estimate) mean.
    :vartype nominal_return: float
    :ivar risk: The calculated risk of the portfolio (e.g., variance, CVaR, or uncertainty budget).
    :vartype risk: float
    """
    weights: npt.NDArray[np.floating]
    nominal_return: float
    risk: float

class Optimization:
    """
    Base class for portfolio optimization problems.

    Provides common attributes and a utility method for calculating the maximum
    expected return under given constraints.

    :cvar _I: The number of assets in the portfolio.
    :cvar _mean: The expected return vector of assets.
    :cvar _G: Matrix for inequality constraints, :math:`G w \\le h`.
    :cvar _h: Vector for inequality constraints, :math:`G w \\le h`.
    :cvar _A: Matrix for equality constraints, :math:`A w = b`.
    :cvar _b: Vector for equality constraints, :math:`A w = b`.
    :cvar _expected_return_row: Row vector representing the negative of the expected returns, used for objective function.
    """
    _I: int
    _mean: np.ndarray
    _G: matrix
    _h: matrix
    _A: matrix
    _b: matrix
    _expected_return_row: matrix

    def _calculate_max_expected_return(self) -> float:
        """
        Calculates the maximum possible expected return given the current constraints.

        This is solved as a linear programming problem where the objective is
        to maximize the portfolio's expected return subject to the defined
        linear equality and inequality constraints. This value typically serves as
        the upper bound for constructing an efficient frontier.

        The problem is formulated as:

        .. math::

            \\max_{w} \\quad & w^T \\mu \\\\
            \\text{subject to} \\quad & G w \\le h \\\\
                                   & A w = b

        :return: The maximum expected return achievable.
        :rtype: float
        :raises ValueError: If the LP solver fails to find an optimal solution,
                            indicating infeasible or unbounded constraints.
        """
        c = self._expected_return_row.T
        sol = solvers.lp(c, self._G, self._h, self._A, self._b, solver="glpk")
        if sol["status"] != "optimal":
            raise ValueError(
                "Could not solve for maximum expected return; "
                "constraints may be infeasible or unbounded."
            )
        return -sol["primal objective"]

class MeanVariance(Optimization):
    """
    Implements classical Mean-Variance portfolio optimization.

    This class solves the quadratic programming problem to find the portfolio
    of assets that minimizes risk (variance) for a given level of expected return,
    forming the efficient frontier as described by Markowitz (1952).

    The optimization problem is formulated as a Quadratic Program (QP):

    .. math::

        \\min_{w} \\quad & w^T \\Sigma w \\\\
        \\text{subject to} \\quad & w^T \\mu \\ge R_{\\text{target}} \\\\
                               & G w \\le h \\\\
                               & A w = b

    where :math:`w` is the vector of portfolio weights, :math:`\\Sigma` is the
    covariance matrix of asset returns, :math:`\\mu` is the vector of expected
    returns, and :math:`R_{\\text{target}}` is the desired minimum expected return.
    Transaction costs can be incorporated as a quadratic penalty in the objective function.

    The implementation uses `cvxopt.solvers.qp`.

    .. seealso:: :cite:t:`markowitz1952portfolio`


    **Transaction Cost Modeling**

    The `market_impact_costs` parameter introduces a quadratic penalty for portfolio turnover,
    modeling the market impact of trading. The cost term is defined as:

    .. math::

        C(w, w_0) = (w - w_0)^T \\Lambda (w - w_0)

    where :math:`w` is the new portfolio weights vector, :math:`w_0` is the initial
    weights vector, and :math:`\\Lambda` is a diagonal matrix with the `market_impact_costs`
    on its diagonal, i.e., :math:`\\Lambda = \\text{diag}(\\lambda_1, \\dots, \\lambda_N)`.

    This cost is incorporated into the standard QP objective function. The solver minimizes
    :math:`\\frac{1}{2} w^T P w + q^T w`. The original objective is :math:`w^T \\Sigma w`.
    The new objective becomes:

    .. math::

        \\min_{w} \\quad w^T \\Sigma w + (w - w_0)^T \\Lambda (w - w_0)
        = w^T (\\Sigma + \\Lambda) w - 2 w^T \\Lambda w_0 + w_0^T \\Lambda w_0

    Ignoring the constant term :math:`w_0^T \\Lambda w_0`, we match this to the solver's formulation:
    - The quadratic term matrix becomes :math:`P = 2(\\Sigma + \\Lambda)`.
    - The linear term vector becomes :math:`q = -2 \\Lambda w_0`.

    These updates are applied internally to the `_P` and `_q` attributes.
    """
    def __init__(
        self,
        mean: np.ndarray,
        covariance_matrix: np.ndarray,
        G: Optional[np.ndarray] = None,
        h: Optional[np.ndarray] = None,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        initial_weights: Optional[npt.NDArray[np.floating]] = None,
        market_impact_costs: Optional[npt.NDArray[np.floating]] = None,
    ):
        """
        Initializes the MeanVariance optimizer.

        :param mean: Expected return vector of assets (:math:`N`).
        :type mean: np.ndarray
        :param covariance_matrix: Covariance matrix of asset returns (:math:`N \\times N`).
        :type covariance_matrix: np.ndarray
        :param G: Matrix for linear inequality constraints (:math:`M \\times N`). Defaults to None.
        :type G: Optional[np.ndarray]
        :param h: Vector for linear inequality constraints (:math:`M`). Defaults to None.
        :type h: Optional[np.ndarray]
        :param A: Matrix for linear equality constraints (:math:`P \\times N`). Defaults to None.
        :type A: Optional[np.ndarray]
        :param b: Vector for linear equality constraints (:math:`P`). Defaults to None.
        :type b: Optional[np.ndarray]
        :param initial_weights: Current portfolio weights (:math:`w_0`), shape (:math:`N`). Required if `market_impact_costs` are provided. Defaults to None.
        :type initial_weights: Optional[npt.NDArray[np.floating]]
        :param market_impact_costs: Per-asset market impact coefficients (:math:`\\lambda_i`), shape (:math:`N`). Defaults to None.
        :type market_impact_costs: Optional[npt.NDArray[np.floating]]
        :raises ValueError: If dimensions of transaction cost parameters do not match.
        """
        self._I = len(mean)
        self._mean = mean
        self._cov = covariance_matrix

        P = 2 * self._cov
        q = np.zeros(self._I)

        if initial_weights is not None and market_impact_costs is not None:
            if initial_weights.shape != (self._I,) or market_impact_costs.shape != (
                self._I,
            ):
                raise ValueError("Dimension mismatch in transaction cost parameters.")
            Lambda = np.diag(market_impact_costs)
            P += 2 * Lambda
            q -= 2 * (Lambda @ initial_weights)

        self._P = matrix(P)
        self._q = matrix(q)

        self._G = matrix(G) if G is not None else None
        self._h = matrix(h) if h is not None else None
        self._A = matrix(A) if A is not None else None
        self._b = matrix(b) if b is not None else None

        self._expected_return_row = -matrix(self._mean).T

    def efficient_portfolio(self, return_target: Optional[float] = None) -> np.ndarray:
        """
        Solves for the efficient portfolio weights.

        :param return_target: The target expected return for the portfolio.
                              If None, the global minimum variance portfolio is returned. Defaults to None.
        :type return_target: Optional[float]
        :return: A 1D NumPy array of optimal portfolio weights.
        :rtype: np.ndarray
        :raises RuntimeError: If the QP solver fails to find an optimal solution.
        """
        G, h = self._G, self._h
        if return_target is not None:
            # Add the return target constraint: -mu^T * w <= -return_target
            if G is None:
                G = self._expected_return_row
                h = matrix([-return_target])
            else:
                G = matrix([self._G, self._expected_return_row])
                h = matrix(np.hstack([np.array(self._h).flatten(), -return_target]))

        sol = solvers.qp(self._P, self._q, G, h, self._A, self._b)
        if sol["status"] != "optimal":
            raise RuntimeError(
                f"QP solver failed to find an optimal solution. Status: {sol['status']}"
            )
        return np.array(sol["x"]).flatten()

    def efficient_frontier(self, num_portfolios: int) -> np.ndarray:
        """
        Computes a series of efficient portfolios to construct the efficient frontier.

        The frontier spans from the minimum variance portfolio up to the maximum
        expected return portfolio achievable under the given constraints.

        :param num_portfolios: The number of portfolios to compute along the frontier.
        :type num_portfolios: int
        :return: A 2D NumPy array of shape (:math:`N`, `num_portfolios`), where :math:`N` is
                 the number of assets. Each column represents the weights of an
                 efficient portfolio.
        :rtype: np.ndarray
        :raises RuntimeError: If the solver fails for any target return along the frontier.
        """
        w_min_vol = self.efficient_portfolio()
        min_ret = self._mean @ w_min_vol
        max_ret = self._calculate_max_expected_return()

        frontier = np.full((self._I, num_portfolios), np.nan)
        frontier[:, 0] = w_min_vol

        if num_portfolios > 1:
            if np.isclose(min_ret, max_ret):
                logger.warning(
                    "Min and max returns are too close; frontier is a single point."
                )
                return frontier[:, :1]

            targets = np.linspace(min_ret, max_ret, num_portfolios)
            for i, target in enumerate(targets):
                if i == 0:
                    continue
                try:
                    frontier[:, i] = self.efficient_portfolio(return_target=target)
                except RuntimeError as e:
                    logger.warning(
                        f"Could not solve for return target {target:.4f} ({e}). "
                        "Truncating frontier."
                    )
                    frontier = frontier[:, :i]
                    break
        return frontier

class MeanCVaR(Optimization):
    """
    Implements Mean-Conditional Value-at-Risk (CVaR) portfolio optimization.

    This class finds the optimal portfolio by minimizing the Conditional Value-at-Risk
    (CVaR), a coherent risk measure that quantifies the expected loss in the tail of
    the return distribution. CVaR is also known as Mean Excess Loss, Mean Shortfall,
    or Tail VaR. It is defined as the conditional expectation of losses
    exceeding the Value-at-Risk (VaR).

    The key insight, following Rockafellar and Uryasev (2000), is that minimizing CVaR
    is equivalent to minimizing a simpler auxiliary function, which can be formulated as
    a Linear Program (LP) when using return scenarios.

    The LP formulation is as follows:

    .. math::

        \\min_{w, \\zeta, u} \\quad & \\zeta + \\frac{1}{T \\alpha} \\sum_{k=1}^T u_k \\\\
        \\text{subject to} \\quad & -R_k^T w - \\zeta \\le u_k, \\quad \\forall k=1, \\dots, T \\\\
                               & u_k \\ge 0, \\quad \\forall k=1, \\dots, T \\\\
                               & w^T \\mu \\ge R_{\\text{target}} \\\\
                               & G w \\le h \\\\
                               & A w = b

    where:
    - :math:`w` are the portfolio weights.
    - :math:`\\zeta` is the Value-at-Risk (VaR).
    - :math:`u_k` are auxiliary variables representing the excess loss for each scenario :math:`k`.
    - :math:`\\alpha` is the tail probability (e.g., 0.05 for 5% CVaR).
    - :math:`R_k` is the vector of asset returns in scenario :math:`k`.
    - :math:`T` is the number of scenarios.

    This implementation uses `cvxopt.solvers.lp`.

    .. seealso:: :cite:t:`rockafellar2000optimization`

    **Transaction Cost Modeling**

    The `proportional_costs` are modeled as a linear penalty on the absolute value of trades.
    The total cost is :math:`C(w, w_0) = \\sum_{i=1}^N c_i |w_i - w_{0,i}|`.

    To maintain a linear formulation, the absolute value is linearized by introducing
    auxiliary variables for buys (:math:`w_{\\text{buy}}`) and sells (:math:`w_{\\text{sell}}`).
    The optimization variables are augmented to include these, and the following constraints are added:
    1.  Decomposition of trades: :math:`w - w_0 = w_{\\text{buy}} - w_{\\text{sell}}`
    2.  Non-negativity: :math:`w_{\\text{buy}} \\ge 0`, :math:`w_{\\text{sell}} \\ge 0`

    The cost term :math:`c^T |w - w_0|` is replaced by :math:`c^T (w_{\\text{buy}} + w_{\\text{sell}})`
    in the objective function. This ensures that the LP solver correctly penalizes turnover
    while finding the optimal risk-return trade-off. The implementation stacks the optimization
    variables as :math:`[w, \\zeta, u, w_{\\text{buy}}, w_{\\text{sell}}]`.
    """
    def __init__(
        self,
        R: np.ndarray,
        p: np.ndarray,
        alpha: float,
        G: Optional[np.ndarray] = None,
        h: Optional[np.ndarray] = None,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        initial_weights: Optional[npt.NDArray[np.floating]] = None,
        proportional_costs: Optional[npt.NDArray[np.floating]] = None,
    ):
        """
        Initializes the MeanCVaR optimizer.

        :param R: Scenarios of asset returns (:math:`T \\times N`), where :math:`T` is the number
                  of scenarios and :math:`N` is the number of assets.
        :type R: np.ndarray
        :param p: Probabilities of each scenario (:math:`T`). Must be non-negative
                  and sum to one.
        :type p: np.ndarray
        :param alpha: The confidence level (tail probability) for CVaR calculation (e.g., 0.05 for 5% CVaR).
                      Must be between 0 and 1 (exclusive).
        :type alpha: float
        :param G: Matrix for linear inequality constraints (:math:`M \\times N`). Defaults to None.
        :type G: Optional[np.ndarray]
        :param h: Vector for linear inequality constraints (:math:`M`). Defaults to None.
        :type h: Optional[np.ndarray]
        :param A: Matrix for linear equality constraints (:math:`P \\times N`). Defaults to None.
        :type A: Optional[np.ndarray]
        :param b: Vector for linear equality constraints (:math:`P`). Defaults to None.
        :type b: Optional[np.ndarray]
        :param initial_weights: Current portfolio weights (:math:`w_0`), shape (:math:`N`). Required if `proportional_costs` are provided. Defaults to None.
        :type initial_weights: Optional[npt.NDArray[np.floating]]
        :param proportional_costs: Per-asset proportional cost coefficients (:math:`c_i`), shape (:math:`N`). Required if `initial_weights` are provided. Defaults to None.
        :type proportional_costs: Optional[npt.NDArray[np.floating]]
        :raises ValueError: If dimensions of transaction cost parameters do not match.
        """
        T, N = R.shape
        self._I = N
        self._mean = p @ R
        self.has_costs = initial_weights is not None and proportional_costs is not None

        # The optimization variables are stacked as [w, zeta, u_1, ..., u_T, (optional: trades)]
        if not self.has_costs:
            # Objective: min(zeta + (p/alpha)' * u)
            c = np.hstack([np.zeros(N), 1.0, p / alpha])

            # Constraints for u: -R*w - zeta <= u  =>  -R*w - zeta - u <= 0
            G_cvar_base = np.hstack([-R, -np.ones((T, 1)), -np.eye(T)])
            h_cvar_base = np.zeros(T)
            
            # Constraints for u: u >= 0
            G_cvar_nonneg = np.hstack([np.zeros((T, N + 1)), -np.eye(T)])
            h_cvar_nonneg = np.zeros(T)

            G_lp = np.vstack([G_cvar_base, G_cvar_nonneg])
            h_lp = np.hstack([h_cvar_base, h_cvar_nonneg])

            # Add user-defined constraints
            if G is not None:
                G_user = np.hstack([G, np.zeros((G.shape[0], 1 + T))])
                G_lp = np.vstack([G_lp, G_user])
                h_lp = np.hstack([h_lp, h])

            A_lp = None
            if A is not None:
                A_lp = np.hstack([A, np.zeros((A.shape[0], 1 + T))])

            self._c = matrix(c)
            self._G = matrix(G_lp)
            self._h = matrix(h_lp)
            self._A = matrix(A_lp) if A_lp is not None else None
            self._b = matrix(b) if b is not None else None
            self._expected_return_row = -matrix(np.hstack([self._mean, np.zeros(1 + T)])).T
        else: # With transaction costs
            # Variables: [w, zeta, u, w_buy, w_sell]
            c = np.hstack([np.zeros(N), 1.0, p / alpha, proportional_costs, proportional_costs])
            
            G_cvar_base = np.hstack([-R, -np.ones((T, 1)), -np.eye(T), np.zeros((T, 2 * N))])
            h_cvar_base = np.zeros(T)
            
            G_cvar_nonneg = np.hstack([np.zeros((T, N + 1)), -np.eye(T), np.zeros((T, 2 * N))])
            h_cvar_nonneg = np.zeros(T)
            
            G_cost_nonneg = np.hstack([np.zeros((2*N, N+1+T)), -np.eye(2*N)])
            h_cost_nonneg = np.zeros(2*N)
            
            G_lp = np.vstack([G_cvar_base, G_cvar_nonneg, G_cost_nonneg])
            h_lp = np.hstack([h_cvar_base, h_cvar_nonneg, h_cost_nonneg])

            if G is not None:
                G_user = np.hstack([G, np.zeros((G.shape[0], 1 + T + 2*N))])
                G_lp = np.vstack([G_lp, G_user])
                h_lp = np.hstack([h_lp, h])
            
            # Equality constraint: w - w_initial = w_buy - w_sell => w - w_buy + w_sell = w_initial
            A_trade = np.hstack([np.eye(N), np.zeros((N, 1+T)), -np.eye(N), np.eye(N)])
            b_trade = initial_weights
            
            A_lp, b_lp = A_trade, b_trade
            if A is not None:
                A_user = np.hstack([A, np.zeros((A.shape[0], 1 + T + 2*N))])
                A_lp = np.vstack([A_user, A_trade])
                b_lp = np.hstack([b, b_trade])

            self._c = matrix(c)
            self._G = matrix(G_lp)
            self._h = matrix(h_lp)
            self._A = matrix(A_lp)
            self._b = matrix(b_lp)
            self._expected_return_row = -matrix(np.hstack([self._mean, np.zeros(1 + T + 2*N)])).T


    def efficient_portfolio(self, return_target: Optional[float] = None) -> np.ndarray:
        """
        Solves for the efficient portfolio weights for Mean-CVaR optimization.

        :param return_target: The target expected return for the portfolio.
                              If None, the minimum CVaR portfolio is returned. Defaults to None.
        :type return_target: Optional[float]
        :return: A 1D NumPy array of optimal portfolio weights.
        :rtype: np.ndarray
        :raises RuntimeError: If the LP solver fails to find an optimal solution.
        """
        G, h = self._G, self._h
        if return_target is not None:
            # Add the return target constraint: -mu^T * w <= -return_target
            if G is None:
                G = self._expected_return_row
                h = matrix([-return_target])
            else:
                G = matrix([self._G, self._expected_return_row])
                h = matrix(np.hstack([np.array(self._h).flatten(), -return_target]))
        
        sol = solvers.lp(self._c, G, h, self._A, self._b, solver="glpk")
        if sol["status"] != "optimal":
            raise RuntimeError(
                f"LP solver failed to find an optimal solution. Status: {sol['status']}"
            )
        return np.array(sol["x"]).flatten()[: self._I]

    def efficient_frontier(self, num_portfolios: int) -> np.ndarray:
        """
        Computes a series of efficient portfolios to construct the Mean-CVaR efficient frontier.

        The frontier spans from the minimum CVaR portfolio up to the maximum
        expected return portfolio achievable under the given constraints.

        :param num_portfolios: The number of portfolios to compute along the frontier.
        :type num_portfolios: int
        :return: A 2D NumPy array of shape (:math:`N`, `num_portfolios`), where :math:`N` is
                 the number of assets. Each column represents the weights of an
                 efficient portfolio.
        :rtype: np.ndarray
        :raises RuntimeError: If the solver fails for any target return along the frontier.
        :raises ValueError: If the minimum and maximum returns are too close, indicating
                            a single-point frontier.
        """
        w_min_cvar = self.efficient_portfolio()
        min_ret = self._mean @ w_min_cvar
        max_ret = self._calculate_max_expected_return()

        frontier = np.full((self._I, num_portfolios), np.nan)
        frontier[:, 0] = w_min_cvar

        if num_portfolios > 1:
            if np.isclose(min_ret, max_ret):
                logger.warning(
                    "Min and max returns are too close; frontier is a single point."
                )
                return frontier[:, :1]

            targets = np.linspace(min_ret, max_ret, num_portfolios)
            for i, target in enumerate(targets):
                if i == 0:
                    continue
                try:
                    frontier[:, i] = self.efficient_portfolio(return_target=target)
                except (RuntimeError, ValueError) as e:
                    logger.warning(
                        f"Could not solve for return target {target:.4f} ({e}). "
                        "Truncating frontier."
                    )
                    frontier = frontier[:, :i]
                    break
        return frontier

class RobustOptimizer:
    """
    Implements robust portfolio optimization based on uncertainty sets.

    This class addresses *estimation risk*, the sub-optimality that arises from
    using estimated, rather than true, market parameters. The methodology,
    based on Meucci (2005), finds a portfolio that is optimal in the worst-case
    scenario within a given *uncertainty set* for the market parameters.

    The Bayesian framework provides a natural way to define these uncertainty sets
    as *credibility sets*, specifically the location-dispersion ellipsoid of the
    posterior distribution of the parameters.

    The uncertainty set for the mean return :math:`\\mu` is an ellipsoid:

    .. math::

        \\hat{\\Theta}_{\\mu} \\equiv \\{\\mu : (\\mu - \\hat{\\mu}_{ce})' S_{\\mu}^{-1} (\\mu - \\hat{\\mu}_{ce}) \\le q_{\\mu}^2\\}

    where :math:`\\hat{\\mu}_{ce}` is the posterior mean and :math:`S_{\\mu}` is the posterior
    scatter matrix. The max-min problem :math:`\\max_{w} \\{ \\min_{\\mu \\in \\hat{\\Theta}_{\\mu}} \\{ w' \\mu \\} \\}`
    simplifies to a tractable form. This class solves the resulting problem, which is
    formulated as a Second-Order Cone Program (SOCP) and solved with `cvxopt.solvers.conelp`.

    Two variants are offered:

    Lambda (λ) variant
        Maximizes a utility function that balances nominal return against estimation
        risk, controlled by a risk-aversion parameter :math:`\\lambda`.
        The problem is:

        .. math::

            \\max_{w} \\quad & w' \\mu - \\lambda \\sqrt{w' \\Sigma' w} \\\\
            \\text{subject to} \\quad & G w \\le h, \\quad A w = b

    Gamma (γ) variant
        Explicitly constrains the size of the uncertainty, finding the portfolio
        with the minimum required return that satisfies the robustness constraints.
        This provides direct control over the confidence level of the uncertainty
        ellipsoids.

    .. seealso:: :cite:t:`meucci2005robust`
    
    **Transaction Cost Modeling**

    Proportional transaction costs are modeled identically to the `MeanCVaR` class.
    The cost :math:`C(w, w_0) = c^T |w - w_0|` is linearized by introducing variables
    :math:`w_{\\text{buy}}` and :math:`w_{\\text{sell}}`, and adding the term
    :math:`c^T (w_{\\text{buy}} + w_{\\text{sell}})` to the objective function.

    The SOCP formulation is augmented with the corresponding variables and constraints.
    The optimization variables become :math:`[w, t, w_{\\text{buy}}, w_{\\text{sell}}]`, where :math:`t`
    is the auxiliary variable for the conic constraint. The objective function and
    equality constraints are modified to include the buy/sell variables and their costs.
    """
    def __init__(
        self,
        expected_return: npt.NDArray[np.floating],
        uncertainty_covariance: npt.NDArray[np.floating],
        G: Optional[npt.NDArray[np.floating]] = None,
        h: Optional[npt.NDArray[np.floating]] = None,
        A: Optional[npt.NDArray[np.floating]] = None,
        b: Optional[npt.NDArray[np.floating]] = None,
        initial_weights: Optional[npt.NDArray[np.floating]] = None,
        proportional_costs: Optional[npt.NDArray[np.floating]] = None,
    ):
        """
        Initializes the RobustOptimizer.

        :param expected_return: The nominal expected return vector of assets (:math:`N`).
                                This is typically the posterior mean (:math:`\\mu_1`) from a Bayesian update.
        :type expected_return: npt.NDArray[np.floating]
        :param uncertainty_covariance: The uncertainty covariance matrix (:math:`N \\times N`), denoted as :math:`\\Sigma_1`
                                       in Meucci (2005). This matrix defines the shape of the uncertainty
                                       ellipsoid.
        :type uncertainty_covariance: npt.NDArray[np.floating]
        :param G: Matrix for linear inequality constraints (:math:`M \\times N`). Defaults to None.
        :type G: Optional[npt.NDArray[np.floating]]
        :param h: Vector for linear inequality constraints (:math:`M`). Defaults to None.
        :type h: Optional[npt.NDArray[np.floating]]
        :param A: Matrix for linear equality constraints (:math:`P \\times N`). Defaults to None.
        :type A: Optional[npt.NDArray[np.floating]]
        :param b: Vector for linear equality constraints (:math:`P`). Defaults to None.
        :type b: Optional[npt.NDArray[np.floating]]
        :param initial_weights: Current portfolio weights (:math:`w_0`), shape (:math:`N`). Required if `proportional_costs` are provided. Defaults to None.
        :type initial_weights: Optional[npt.NDArray[np.floating]]
        :param proportional_costs: Per-asset proportional cost coefficients (:math:`c_i`), shape (:math:`N`). Required if `initial_weights` are provided. Defaults to None.
        :type proportional_costs: Optional[npt.NDArray[np.floating]]
        :raises ValueError: If dimensions of transaction cost parameters do not match.
        """
        self.mu = np.asarray(expected_return, dtype=float)
        self.sigma_prime = np.asarray(uncertainty_covariance, dtype=float)
        self.N = self.mu.size

        self.s_prime_sqrt = _cholesky_pd(self.sigma_prime)

        self.G, self.h, self.A, self.b = G, h, A, b
        self.initial_weights = initial_weights
        self.proportional_costs = proportional_costs
        self.has_costs = initial_weights is not None and proportional_costs is not None
        if self.has_costs and (
            initial_weights.shape != (self.N,) or proportional_costs.shape != (self.N,)
        ):
            raise ValueError("Dimension mismatch in transaction cost parameters.")

    def solve_lambda_variant(self, lam: float) -> OptimizationResult:
        """
        Solves the robust optimization problem using the lambda (:math:`\\lambda`) variant.

        This variant maximizes a utility function that balances expected return
        against the uncertainty in that return, controlled by :math:`\\lambda`. The parameter
        :math:`\\lambda` can be interpreted as an investor's aversion to both market
        and estimation risk.

        :param lam: The lambda (:math:`\\lambda`) parameter, a non-negative real number
                    controlling the trade-off between expected return and robustness.
                    Higher values imply more emphasis on robustness.
        :type lam: float
        :return: An object containing the optimal weights, nominal return, and the risk
                 (uncertainty budget :math:`t = \\sqrt{w' \\Sigma' w}`).
        :rtype: OptimizationResult
        :raises ValueError: If `lam` is negative.
        :raises RuntimeError: If the SOCP solver fails to find an optimal solution.
        """
        if not isinstance(lam, numbers.Real) or lam < 0:
            raise ValueError("Lambda (λ) must be a non-negative real number.")
        return self._solve_socp(lam=lam)

    def solve_gamma_variant(
        self, gamma_mu: float, gamma_sigma_sq: float
    ) -> OptimizationResult:
        """
        Solves the robust optimization problem using the gamma (:math:`\\gamma`) variant.

        This variant minimizes the uncertainty budget :math:`t` subject to an explicit
        upper bound on its value, derived from the uncertainty in the covariance
        matrix estimate. The parameter :math:`\\gamma_\\mu` is used as a penalty in the
        objective function.

        :param gamma_mu: The gamma mu (:math:`\\gamma_\\mu`) parameter, a non-negative real
                         number defining the aversion to uncertainty in the mean.
        :type gamma_mu: float
        :param gamma_sigma_sq: The gamma sigma squared (:math:`\\gamma_{\\Sigma}^{(i)}` in Meucci (2005))
                               parameter, an upper bound on the portfolio variance under
                               uncertainty. The risk budget is capped at :math:`\\sqrt{\\gamma_{\\sigma}^2}`.
        :type gamma_sigma_sq: float
        :return: An object containing the optimal weights, nominal return, and the risk
                 (uncertainty budget :math:`t = \\sqrt{w' \\Sigma' w}`).
        :rtype: OptimizationResult
        :raises ValueError: If `gamma_mu` or `gamma_sigma_sq` are negative.
        :raises RuntimeError: If the SOCP solver fails to find an optimal solution.
        """
        if not isinstance(gamma_mu, numbers.Real) or gamma_mu < 0:
            raise ValueError("Gamma mu (γ_μ) must be a non-negative real number.")
        if not isinstance(gamma_sigma_sq, numbers.Real) or gamma_sigma_sq < 0:
            raise ValueError(
                "Gamma sigma squared (γ_σ) must be a non-negative real number."
            )
        return self._solve_socp(gamma_mu=gamma_mu, gamma_sigma_sq=gamma_sigma_sq)

    def efficient_frontier(
        self, lambdas: Sequence[float]
    ) -> Tuple[list[float], list[float], npt.NDArray[np.floating]]:
        """
        Computes a series of robust efficient portfolios for different lambda values.

        This method generates the robust efficient frontier by solving the
        lambda variant of the robust optimization problem for a range of
        lambda values. As :math:`\\lambda` increases, the allocation shrinks
        towards the global minimum variance portfolio to reduce estimation risk.

        :param lambdas: A sequence of non-negative lambda values
                        for which to compute efficient portfolios.
        :type lambdas: Sequence[float]
        :return: A tuple containing:

                 - **returns**: A list of nominal expected returns for each portfolio on the frontier.
                 - **risks**: A list of risk measures (:math:`t = \\sqrt{w' \\Sigma' w}`) for each portfolio on the frontier.
                 - **weights**: A 2D NumPy array of shape (:math:`N`, `len(lambdas)`), where
                   each column represents the weights of an efficient portfolio.
        :rtype: Tuple[list[float], list[float], npt.NDArray[np.floating]]
        """
        results = [self.solve_lambda_variant(l) for l in lambdas]
        returns = [res.nominal_return for res in results]
        risks = [res.risk for res in results]
        weights = np.column_stack([res.weights for res in results])
        return returns, risks, weights

    def _solve_socp(self, **kwargs) -> OptimizationResult:
        """
        Internal method to solve the Second-Order Cone Program (SOCP).

        This method constructs and solves the SOCP by reformulating the robust
        optimization problem. An auxiliary variable :math:`t` is introduced, where
        :math:`t \\ge \\sqrt{w' \\Sigma' w}`, which is equivalent to the conic constraint
        :math:`||\\Sigma'^{1/2} w||_2 \\le t`. The objective becomes maximizing
        :math:`w'\\mu - \\lambda t` (for the lambda variant).

        :param kwargs: Keyword arguments for the optimization variant:
                       - `lam` (float): For the lambda variant, the penalty parameter.
                       - `gamma_mu` (float): For the gamma variant, the mean uncertainty radius.
                       - `gamma_sigma_sq` (float): For the gamma variant, the covariance uncertainty radius.
        :return: An object containing the optimal weights, nominal return, and the risk.
        :rtype: OptimizationResult
        :raises RuntimeError: If the SOCP solver fails to find an optimal solution.
        """
        penalty = kwargs.get("lam", kwargs.get("gamma_mu"))
        
        # Optimization variables are [w, t, (optional trades)]
        if not self.has_costs:
            num_vars = self.N + 1
            # Objective: min(-mu' * w + penalty * t)
            c_obj = np.hstack([-self.mu, penalty])
            
            # SOCP constraint: ||s_prime_sqrt * w||_2 <= t
            # [ -t ]
            # [ -s_prime_sqrt * w ] is in the second-order cone
            G_soc = np.zeros((self.N + 1, num_vars))
            G_soc[0, self.N] = -1.0
            G_soc[1:, : self.N] = -self.s_prime_sqrt
            h_soc = np.zeros(self.N + 1)
            
            num_lin_ineq = self.G.shape[0] if self.G is not None else 0
            G_ineq_ext = np.hstack([self.G, np.zeros((num_lin_ineq, 1))]) if self.G is not None else np.zeros((0, num_vars))
            h_ineq_ext = self.h if self.h is not None else np.zeros(0)
            
            # Gamma variant adds a cap on the risk budget: t <= sqrt(gamma_sigma_sq)
            if "gamma_sigma_sq" in kwargs:
                cap_row = np.zeros((1, num_vars))
                cap_row[0, self.N] = 1.0
                G_ineq_ext = np.vstack([G_ineq_ext, cap_row])
                h_ineq_ext = np.hstack([h_ineq_ext, np.sqrt(kwargs["gamma_sigma_sq"])])
                num_lin_ineq += 1
            
            A_eq = matrix(np.hstack([self.A, np.zeros((self.A.shape[0], 1))])) if self.A is not None else None
            b_eq = matrix(self.b) if self.b is not None else None
        else:
            # Variables: [w, t, w_buy, w_sell]
            num_vars = self.N + 1 + 2*self.N
            c_obj = np.hstack([-self.mu, penalty, self.proportional_costs, self.proportional_costs])

            G_soc = np.zeros((self.N + 1, num_vars))
            G_soc[0, self.N] = -1.0
            G_soc[1:, : self.N] = -self.s_prime_sqrt
            h_soc = np.zeros(self.N + 1)
            
            num_lin_ineq = self.G.shape[0] if self.G is not None else 0
            G_ineq_ext = np.hstack([self.G, np.zeros((num_lin_ineq, 1 + 2*self.N))]) if self.G is not None else np.zeros((0, num_vars))
            h_ineq_ext = self.h if self.h is not None else np.zeros(0)
            
            # Non-negativity for trade variables
            G_cost_nonneg = np.hstack([np.zeros((2*self.N, self.N+1)), -np.eye(2*self.N)])
            G_ineq_ext = np.vstack([G_ineq_ext, G_cost_nonneg])
            h_ineq_ext = np.hstack([h_ineq_ext, np.zeros(2*self.N)])
            num_lin_ineq += 2*self.N

            if "gamma_sigma_sq" in kwargs:
                cap_row = np.zeros((1, num_vars))
                cap_row[0, self.N] = 1.0
                G_ineq_ext = np.vstack([G_ineq_ext, cap_row])
                h_ineq_ext = np.hstack([h_ineq_ext, np.sqrt(kwargs["gamma_sigma_sq"])])
                num_lin_ineq += 1

            A_trade = np.hstack([np.eye(self.N), np.zeros((self.N, 1)), -np.eye(self.N), np.eye(self.N)])
            b_trade = self.initial_weights

            A_eq, b_eq = matrix(A_trade), matrix(b_trade)
            if self.A is not None:
                A_user = np.hstack([self.A, np.zeros((self.A.shape[0], 1 + 2*self.N))])
                A_eq = matrix(np.vstack([A_user, A_trade]))
                b_eq = matrix(np.hstack([self.b, b_trade]))

        G_cone = matrix([matrix(G_ineq_ext), matrix(G_soc)])
        h_cone = matrix([matrix(h_ineq_ext), matrix(h_soc)])
        dims = {"l": num_lin_ineq, "q": [self.N + 1], "s": []}

        sol = solvers.conelp(matrix(c_obj), G_cone, h_cone, dims=dims, A=A_eq, b=b_eq)

        if sol["status"] != "optimal":
            raise RuntimeError(
                f"SOCP solver failed to find an optimal solution. Status: {sol['status']}"
            )

        x_opt = np.array(sol["x"]).flatten()
        w_opt = x_opt[: self.N]
        t_opt = x_opt[self.N]

        return OptimizationResult(
            weights=w_opt, nominal_return=self.mu @ w_opt, risk=t_opt
        )
"""
This module provides classes for solving various portfolio optimization problems,
including Mean-Variance, Mean-CVaR, and Robust Optimization.

It leverages `cvxopt` for solving quadratic programming (QP) and conic
programming (SOCP) problems, and integrates with Bayesian methods for robust
estimation.

Classes:
-   `OptimizationResult`: A dataclass to hold the results of an optimization.
-   `Optimization`: A base class providing common optimization utilities.
-   `MeanVariance`: Implements classical Mean-Variance portfolio optimization.
-   `MeanCVaR`: Implements Mean-Conditional Value-at-Risk portfolio optimization.
-   `RobustOptimizer`: Implements robust portfolio optimization based on
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

    Attributes:
        weights (npt.NDArray[np.floating]): The optimal portfolio weights as a NumPy array.
        nominal_return (float): The nominal expected return of the optimized portfolio.
        risk (float): The calculated risk measure of the optimized portfolio
                      (e.g., standard deviation for Mean-Variance, CVaR for Mean-CVaR,
                      or the uncertainty budget for Robust Optimization).
    """
    weights: npt.NDArray[np.floating]
    nominal_return: float
    risk: float


class Optimization:
    """
    Base class for portfolio optimization problems.

    Provides common attributes and a utility method for calculating the maximum
    expected return under given constraints.

    Attributes:
        _I (int): The number of assets in the portfolio.
        _mean (np.ndarray): The expected return vector of assets.
        _G (cvxopt.matrix): Matrix for inequality constraints (G * x <= h).
        _h (cvxopt.matrix): Vector for inequality constraints (G * x <= h).
        _A (cvxopt.matrix): Matrix for equality constraints (A * x = b).
        _b (cvxopt.matrix): Vector for equality constraints (A * x = b).
        _expected_return_row (cvxopt.matrix): Row vector representing the
            negative of the expected returns, used for objective function.
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
        linear equality and inequality constraints.

        Returns:
            float: The maximum expected return achievable.

        Raises:
            ValueError: If the LP solver fails to find an optimal solution,
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

    This class solves the quadratic programming problem to find portfolio
    weights that minimize risk for a given level of expected return, or
    maximize return for a given level of risk. It supports linear equality
    and inequality constraints, as well as transaction costs.

    The optimization problem is formulated as:

    Minimize: :math:`w^T \\Sigma w - w^T \\mu` (or similar, depending on formulation)
    Subject to:
        - Linear inequality constraints: :math:`G w \\le h`
        - Linear equality constraints: :math:`A w = b`
        - Optional: Target return constraint
        - Optional: Transaction costs

    Uses `cvxopt.solvers.qp` for solving the quadratic program.
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

        Args:
            mean (np.ndarray): Expected return vector of assets (N,).
            covariance_matrix (np.ndarray): Covariance matrix of asset returns (N, N).
            G (Optional[np.ndarray]): Matrix for linear inequality constraints (M, N).
                Defaults to None.
            h (Optional[np.ndarray]): Vector for linear inequality constraints (M,).
                Defaults to None.
            A (Optional[np.ndarray]): Matrix for linear equality constraints (P, N).
                Defaults to None.
            b (Optional[np.ndarray]): Vector for linear equality constraints (P,).
                Defaults to None.
            initial_weights (Optional[npt.NDArray[np.floating]]): Current portfolio weights (N,).
                Required if `market_impact_costs` are provided. Defaults to None.
            market_impact_costs (Optional[npt.NDArray[np.floating]]): Proportional transaction
                costs for each asset (N,). Applied as a quadratic cost.
                Required if `initial_weights` are provided. Defaults to None.

        Raises:
            ValueError: If dimensions of transaction cost parameters do not match.
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

        Args:
            return_target (Optional[float]): The target expected return for the portfolio.
                If None, the minimum variance portfolio is returned. Defaults to None.

        Returns:
            np.ndarray: A 1D NumPy array of optimal portfolio weights.

        Raises:
            RuntimeError: If the QP solver fails to find an optimal solution.
        """
        G, h = self._G, self._h
        if return_target is not None:
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

        Args:
            num_portfolios (int): The number of portfolios to compute along the frontier.

        Returns:
            np.ndarray: A 2D NumPy array of shape (N, num_portfolios), where N is
            the number of assets. Each column represents the weights of an
            efficient portfolio.

        Raises:
            RuntimeError: If the solver fails for any target return along the frontier.
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

    This class solves a linear programming problem to find portfolio weights
    that minimize CVaR for a given expected return, or maximize expected return
    for a given CVaR. It supports linear equality and inequality constraints,
    and proportional transaction costs.

    The optimization problem is formulated as:

    Minimize: :math:`CVaR_{\\alpha}(R_p)`
    Subject to:
        - Expected return constraint: :math:`E[R_p] \\ge R_{target}`
        - Linear inequality constraints: :math:`G w \\le h`
        - Linear equality constraints: :math:`A w = b`
        - Optional: Transaction costs

    Uses `cvxopt.solvers.lp` for solving the linear program.
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

        Args:
            R (np.ndarray): Scenarios of asset returns (T, N), where T is the number
                of scenarios and N is the number of assets.
            p (np.ndarray): Probabilities of each scenario (T,). Must be non-negative
                and sum to one.
            alpha (float): The confidence level for CVaR calculation (e.g., 0.05 for 5% CVaR).
                Must be between 0 and 1 (exclusive).
            G (Optional[np.ndarray]): Matrix for linear inequality constraints (M, N).
                Defaults to None.
            h (Optional[np.ndarray]): Vector for linear inequality constraints (M,).
                Defaults to None.
            A (Optional[np.ndarray]): Matrix for linear equality constraints (P, N).
                Defaults to None.
            b (Optional[np.ndarray]): Vector for linear equality constraints (P,).
                Defaults to None.
            initial_weights (Optional[npt.NDArray[np.floating]]): Current portfolio weights (N,).
                Required if `proportional_costs` are provided. Defaults to None.
            proportional_costs (Optional[npt.NDArray[np.floating]]): Proportional transaction
                costs for each asset (N,). Required if `initial_weights` are provided.
                Defaults to None.

        Raises:
            ValueError: If dimensions of transaction cost parameters do not match.
        """
        T, N = R.shape
        self._I = N
        self._mean = p @ R
        self.has_costs = initial_weights is not None and proportional_costs is not None

        if not self.has_costs:
            c = np.hstack([np.zeros(N), 1.0, p / alpha])
            G_cvar_base = np.hstack([-R, -np.ones((T, 1)), -np.eye(T)])
            h_cvar_base = np.zeros(T)
            G_cvar_nonneg = np.hstack([np.zeros((T, N + 1)), -np.eye(T)])
            h_cvar_nonneg = np.zeros(T)
            G_lp = np.vstack([G_cvar_base, G_cvar_nonneg])
            h_lp = np.hstack([h_cvar_base, h_cvar_nonneg])

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
        else:
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

        Args:
            return_target (Optional[float]): The target expected return for the portfolio.
                If None, the minimum CVaR portfolio is returned. Defaults to None.

        Returns:
            np.ndarray: A 1D NumPy array of optimal portfolio weights.

        Raises:
            RuntimeError: If the LP solver fails to find an optimal solution.
        """
        G, h = self._G, self._h
        if return_target is not None:
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

        Args:
            num_portfolios (int): The number of portfolios to compute along the frontier.

        Returns:
            np.ndarray: A 2D NumPy array of shape (N, num_portfolios), where N is
            the number of assets. Each column represents the weights of an
            efficient portfolio.

        Raises:
            RuntimeError: If the solver fails for any target return along the frontier.
            ValueError: If the minimum and maximum returns are too close, indicating
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
    Implements robust portfolio optimization based on uncertainty sets for
    expected return and covariance matrix.

    This class solves a Second-Order Cone Program (SOCP) to find portfolio
    weights that are robust to estimation errors in the input parameters.
    It supports two main variants:
    -   **Lambda (λ) variant**: Controls the trade-off between nominal return
        and robustness, similar to a risk-aversion parameter.
    -   **Gamma (γ) variant**: Explicitly defines the size of the uncertainty
        ellipsoids for the mean and covariance, allowing for direct control
        over the level of robustness.

    The optimization problem is formulated as:

    Maximize: :math:`w^T \\mu - \\lambda \\sqrt{w^T \\Sigma' w}` (Lambda variant)
    Minimize: :math:`t`
    Subject to:
        - :math:`w^T \\mu - t \\ge R_{target}` (Gamma variant)
        - :math:`\\|\\Sigma'^{1/2} w\\|_2 \\le t`
        - Linear inequality constraints: :math:`G w \\le h`
        - Linear equality constraints: :math:`A w = b`
        - Optional: Transaction costs

    Uses `cvxopt.solvers.conelp` for solving the second-order cone program.
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

        Args:
            expected_return (npt.NDArray[np.floating]): The nominal expected return
                vector of assets (N,). This is typically the posterior mean from
                a Bayesian update.
            uncertainty_covariance (npt.NDArray[np.floating]): The uncertainty
                covariance matrix (N, N). This is typically related to the
                posterior covariance of the mean estimate (e.g., :math:`S_{\\mu}`
                from :class:`~pyvallocation.bayesian.NIWPosterior`).
            G (Optional[npt.NDArray[np.floating]]): Matrix for linear inequality
                constraints (M, N). Defaults to None.
            h (Optional[npt.NDArray[np.floating]]): Vector for linear inequality
                constraints (M,). Defaults to None.
            A (Optional[npt.NDArray[np.floating]]): Matrix for linear equality
                constraints (P, N). Defaults to None.
            b (Optional[npt.NDArray[np.floating]]): Vector for linear equality
                constraints (P,). Defaults to None.
            initial_weights (Optional[npt.NDArray[np.floating]]): Current portfolio weights (N,).
                Required if `proportional_costs` are provided. Defaults to None.
            proportional_costs (Optional[npt.NDArray[np.floating]]): Proportional transaction
                costs for each asset (N,). Required if `initial_weights` are provided.
                Defaults to None.

        Raises:
            ValueError: If dimensions of transaction cost parameters do not match.
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
        Solves the robust optimization problem using the lambda (λ) variant.

        This variant maximizes a utility function that balances expected return
        against the uncertainty in the return estimate, controlled by `lam`.

        Args:
            lam (float): The lambda (λ) parameter, a non-negative real number
                controlling the trade-off between expected return and robustness.
                Higher values imply more emphasis on robustness.

        Returns:
            OptimizationResult: An object containing the optimal weights,
            nominal return, and the risk (uncertainty budget `t`).

        Raises:
            ValueError: If `lam` is negative.
            RuntimeError: If the SOCP solver fails to find an optimal solution.
        """
        if not isinstance(lam, numbers.Real) or lam < 0:
            raise ValueError("Lambda (λ) must be a non-negative real number.")
        return self._solve_socp(lam=lam)

    def solve_gamma_variant(
        self, gamma_mu: float, gamma_sigma_sq: float
    ) -> OptimizationResult:
        """
        Solves the robust optimization problem using the gamma (γ) variant.

        This variant minimizes the uncertainty budget `t` subject to a minimum
        expected return and explicit bounds on the uncertainty of mean and
        covariance estimates.

        Args:
            gamma_mu (float): The gamma mu (γ_μ) parameter, a non-negative real
                number defining the radius of the uncertainty ellipsoid for the mean.
            gamma_sigma_sq (float): The gamma sigma squared (γ_σ) parameter, a
                non-negative real number defining the radius of the uncertainty
                ellipsoid for the covariance.

        Returns:
            OptimizationResult: An object containing the optimal weights,
            nominal return, and the risk (uncertainty budget `t`).

        Raises:
            ValueError: If `gamma_mu` or `gamma_sigma_sq` are negative.
            RuntimeError: If the SOCP solver fails to find an optimal solution.
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
        lambda values.

        Args:
            lambdas (Sequence[float]): A sequence of non-negative lambda values
                for which to compute efficient portfolios.

        Returns:
            Tuple[list[float], list[float], npt.NDArray[np.floating]]: A tuple containing:
                -   **returns** (list[float]): A list of nominal expected returns for
                    each portfolio on the frontier.
                -   **risks** (list[float]): A list of risk measures (uncertainty budget `t`)
                    for each portfolio on the frontier.
                -   **weights** (npt.NDArray[np.floating]): A 2D NumPy array of shape
                    (N, len(lambdas)), where each column represents the weights of
                    an efficient portfolio.
        """
        results = [self.solve_lambda_variant(l) for l in lambdas]
        returns = [res.nominal_return for res in results]
        risks = [res.risk for res in results]
        weights = np.column_stack([res.weights for res in results])
        return returns, risks, weights

    def _solve_socp(self, **kwargs) -> OptimizationResult:
        """
        Internal method to solve the Second-Order Cone Program (SOCP).

        This method constructs and solves the SOCP based on the provided
        parameters, handling both lambda and gamma variants, and incorporating
        transaction costs if specified.

        Args:
            **kwargs: Keyword arguments for the optimization variant:
                -   `lam` (float): For the lambda variant, the penalty parameter.
                -   `gamma_mu` (float): For the gamma variant, the mean uncertainty radius.
                -   `gamma_sigma_sq` (float): For the gamma variant, the covariance uncertainty radius.

        Returns:
            OptimizationResult: An object containing the optimal weights,
            nominal return, and the risk (uncertainty budget `t`).

        Raises:
            RuntimeError: If the SOCP solver fails to find an optimal solution.
        """
        penalty = kwargs.get("lam", kwargs.get("gamma_mu"))
        
        if not self.has_costs:
            num_vars = self.N + 1
            c_obj = np.hstack([-self.mu, penalty])

            G_soc = np.zeros((self.N + 1, num_vars))
            G_soc[0, self.N] = -1.0
            G_soc[1:, : self.N] = -self.s_prime_sqrt
            h_soc = np.zeros(self.N + 1)

            num_lin_ineq = self.G.shape[0] if self.G is not None else 0
            G_ineq_ext = np.hstack([self.G, np.zeros((num_lin_ineq, 1))]) if self.G is not None else np.zeros((0, num_vars))
            h_ineq_ext = self.h if self.h is not None else np.zeros(0)

            if "gamma_sigma_sq" in kwargs:
                cap_row = np.zeros((1, num_vars))
                cap_row[0, self.N] = 1.0
                G_ineq_ext = np.vstack([G_ineq_ext, cap_row])
                h_ineq_ext = np.hstack([h_ineq_ext, np.sqrt(kwargs["gamma_sigma_sq"])])
                num_lin_ineq += 1

            A_eq = matrix(np.hstack([self.A, np.zeros((self.A.shape[0], 1))])) if self.A is not None else None
            b_eq = matrix(self.b) if self.b is not None else None
        else:
            num_vars = self.N + 1 + 2*self.N
            c_obj = np.hstack([-self.mu, penalty, self.proportional_costs, self.proportional_costs])

            G_soc = np.zeros((self.N + 1, num_vars))
            G_soc[0, self.N] = -1.0
            G_soc[1:, : self.N] = -self.s_prime_sqrt
            h_soc = np.zeros(self.N + 1)
            
            num_lin_ineq = self.G.shape[0] if self.G is not None else 0
            G_ineq_ext = np.hstack([self.G, np.zeros((num_lin_ineq, 1 + 2*self.N))]) if self.G is not None else np.zeros((0, num_vars))
            h_ineq_ext = self.h if self.h is not None else np.zeros(0)
            
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

            A_lp, b_lp = matrix(A_trade), matrix(b_trade)
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

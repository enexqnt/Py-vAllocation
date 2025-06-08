from __future__ import annotations

import logging
import numbers
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from cvxopt import matrix, solvers
from .bayesian import _cholesky_pd
# --- Configure CVXOPT Solver Options ---
solvers.options["glpk"] = {"msg_lev": "GLP_MSG_OFF"}
solvers.options["show_progress"] = False

# --- Module-level logger ---
logger = logging.getLogger(__name__)

@dataclass(slots=True, frozen=True)
class OptimizationResult:
    """
    An immutable container for the results of a single portfolio optimization.

    Attributes:
        weights: The optimal portfolio weights (N-dimensional array).
        nominal_return: The expected return of the portfolio (μᵀw).
        risk: The value of the risk measure for the optimal portfolio.
    """
    weights: npt.NDArray[np.floating]
    nominal_return: float
    risk: float


# --- Base and Concrete Optimizer Classes ---

class Optimization:
    """Base class for portfolio optimizers."""
    _I: int
    _mean: np.ndarray
    _G: matrix
    _h: matrix
    _A: matrix
    _b: matrix
    _expected_return_row: matrix
    
    def _calculate_max_expected_return(self) -> float:
        """Solves an LP to find the maximum possible return under the constraints."""
        c = self._expected_return_row.T
        sol = solvers.lp(c, self._G, self._h, self._A, self._b, solver="glpk")
        if sol["status"] != "optimal":
            raise ValueError("Could not solve for maximum expected return; constraints may be infeasible or unbounded.")
        return -sol["primal objective"]


class MeanVariance(Optimization):
    """Classical mean–variance optimization via Quadratic Programming[cite: 7]."""

    def __init__(
        self,
        mean: np.ndarray,
        covariance_matrix: np.ndarray,
        G: Optional[np.ndarray] = None,
        h: Optional[np.ndarray] = None,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
    ):
        self._I = len(mean)
        self._mean = mean
        self._cov = covariance_matrix

        # Objective function for QP: min (1/2)x'Px + q'x
        # We use P=2*cov so the objective becomes min w' * cov * w
        self._P = matrix(2 * self._cov)
        self._q = matrix(np.zeros(self._I))

        # Constraints
        self._G = matrix(G) if G is not None else None
        self._h = matrix(h) if h is not None else None
        self._A = matrix(A) if A is not None else None
        self._b = matrix(b) if b is not None else None
        
        # Row for return constraint: -μᵀw <= -target
        self._expected_return_row = -matrix(self._mean).T

    def efficient_portfolio(self, return_target: Optional[float] = None) -> np.ndarray:
        """Solves for a single portfolio on the efficient frontier."""
        G, h = self._G, self._h
        if return_target is not None:
            if G is None:
                G = self._expected_return_row
                h = matrix([-return_target])
            else:
                G = matrix([self._G, self._expected_return_row])
                h = matrix([self._h, -return_target])

        sol = solvers.qp(self._P, self._q, G, h, self._A, self._b)
        if sol['status'] != 'optimal':
            raise RuntimeError(f"QP solver failed to find an optimal solution. Status: {sol['status']}")
        return np.array(sol['x']).flatten()

    def efficient_frontier(self, num_portfolios: int) -> np.ndarray:
        """Constructs the efficient frontier by solving for multiple return targets."""
        w_min_vol = self.efficient_portfolio()
        min_ret = self._mean @ w_min_vol
        max_ret = self._calculate_max_expected_return()
        
        frontier = np.full((self._I, num_portfolios), np.nan)
        frontier[:, 0] = w_min_vol

        if num_portfolios > 1:
            if np.isclose(min_ret, max_ret):
                logger.warning("Min and max returns are too close; frontier is a single point.")
                return frontier[:, :1]
            
            targets = np.linspace(min_ret, max_ret, num_portfolios)
            for i, target in enumerate(targets):
                if i == 0: continue # Already have min-vol
                try:
                    frontier[:, i] = self.efficient_portfolio(return_target=target)
                except RuntimeError as e:
                    logger.warning(f"Could not solve for return target {target:.4f} ({e}). Truncating frontier.")
                    frontier = frontier[:, :i]
                    break
        return frontier


class MeanCVaR(Optimization):
    """
    Mean-CVaR optimizer using a Linear Programming formulation.

    CVaR is the expected loss in the worst `alpha` fraction of cases[cite: 6].
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
    ):
        """
        Args:
            R: (T, N) array of T scenarios for N assets.
            p: (T,) array of probabilities for each scenario.
            alpha: The tail probability (e.g., 0.05 for 95% CVaR).
        """
        T, N = R.shape
        self._I = N
        self._mean = p @ R
        
        # LP variable vector: [w (N), VaR (z), u (T)] where u are positive slack vars
        # Total variables: N + 1 + T
        
        # Objective: min z + (1/alpha) * sum(p_i * u_i)
        c = np.hstack([np.zeros(N), 1.0, p / alpha])
        
        # Constraints:
        # 1. Losses - VaR - u <= 0, where Loss = -R@w. So: -R@w - z - u <= 0.
        # 2. u >= 0
        G1 = np.hstack([-R, -np.ones((T, 1)), -np.eye(T)])
        h1 = np.zeros(T)
        
        G2 = np.hstack([np.zeros((T, N + 1)), -np.eye(T)])
        h2 = np.zeros(T)

        G_cvar = np.vstack([G1, G2])
        h_cvar = np.hstack([h1, h2])

        # Combine with user-provided constraints
        G_lp, h_lp = G_cvar, h_cvar
        if G is not None:
            G_user = np.hstack([G, np.zeros((G.shape[0], 1 + T))])
            G_lp = np.vstack([G_cvar, G_user])
            h_lp = np.hstack([h_cvar, h])
        
        A_lp = None
        if A is not None:
            A_lp = np.hstack([A, np.zeros((A.shape[0], 1 + T))])

        self._c = matrix(c)
        self._G = matrix(G_lp)
        self._h = matrix(h_lp)
        self._A = matrix(A_lp) if A_lp is not None else None
        self._b = matrix(b) if b is not None else None

        self._expected_return_row = -matrix(np.hstack([self._mean, np.zeros(1 + T)])).T


    def efficient_portfolio(self, return_target: Optional[float] = None) -> np.ndarray:
        """Solves the Mean-CVaR LP for a single portfolio."""
        G, h = self._G, self._h
        if return_target is not None:
            G = matrix([self._G, self._expected_return_row])
            h = matrix([self._h, -return_target])

        sol = solvers.lp(self._c, G, h, self._A, self._b, solver='glpk')
        if sol['status'] != 'optimal':
            raise RuntimeError(f"LP solver failed to find an optimal solution. Status: {sol['status']}")
        return np.array(sol['x']).flatten()[:self._I]

    def efficient_frontier(self, num_portfolios: int) -> np.ndarray:
        """Constructs the Mean-CVaR efficient frontier."""
        w_min_cvar = self.efficient_portfolio()
        min_ret = self._mean @ w_min_cvar
        max_ret = self._calculate_max_expected_return()

        frontier = np.full((self._I, num_portfolios), np.nan)
        frontier[:, 0] = w_min_cvar

        if num_portfolios > 1:
            if np.isclose(min_ret, max_ret):
                logger.warning("Min and max returns are too close; frontier is a single point.")
                return frontier[:, :1]
                
            targets = np.linspace(min_ret, max_ret, num_portfolios)
            for i, target in enumerate(targets):
                if i == 0: continue
                try:
                    frontier[:, i] = self.efficient_portfolio(return_target=target)
                except (RuntimeError, ValueError) as e:
                    logger.warning(f"Could not solve for return target {target:.4f} ({e}). Truncating frontier.")
                    frontier = frontier[:, :i]
                    break
        return frontier


class RobustOptimizer:
    """
    Robust portfolio optimization using Second-Order Cone Programming
    following Meucci's robust allocation framework[cite: 1].

    This optimizer is designed to handle parameter uncertainty. It assumes the
    user provides the posterior mean (as `expected_return`) and the posterior
    scale matrix (as `uncertainty_covariance`) from a Bayesian estimation
    process.
    """
    def __init__(
        self,
        expected_return: npt.NDArray[np.floating],
        uncertainty_covariance: npt.NDArray[np.floating],
        G: Optional[npt.NDArray[np.floating]] = None,
        h: Optional[npt.NDArray[np.floating]] = None,
        A: Optional[npt.NDArray[np.floating]] = None,
        b: Optional[npt.NDArray[np.floating]] = None,
    ):
        self.mu = np.asarray(expected_return, dtype=float)
        self.sigma_prime = np.asarray(uncertainty_covariance, dtype=float)
        self.N = self.mu.size
        
        # Pre-calculate the Cholesky factor of the uncertainty matrix
        self.s_prime_sqrt = _cholesky_pd(self.sigma_prime)
        
        self.G, self.h, self.A, self.b = G, h, A, b

    def solve_lambda_variant(self, lam: float) -> OptimizationResult:
        """
        Solves `max μᵀw − λ·‖Σ′¹/²w‖₂` (Eq. 22 in Meucci's framework[cite: 1]).
        """
        if not isinstance(lam, numbers.Real) or lam < 0:
            raise ValueError("Lambda (λ) must be a non-negative real number.")
        return self._solve_socp(lam=lam)

    def solve_gamma_variant(self, gamma_mu: float, gamma_sigma_sq: float) -> OptimizationResult:
        """
        Solves `max μᵀw − γμ·t` subject to
        `‖Σ′¹/²w‖₂ ≤ t ≤ √γ_σ` (Eq. 19 in Meucci's framework[cite: 1]).
        """
        if not isinstance(gamma_mu, numbers.Real) or gamma_mu < 0:
            raise ValueError("Gamma mu (γ_μ) must be a non-negative real number.")
        if not isinstance(gamma_sigma_sq, numbers.Real) or gamma_sigma_sq < 0:
            raise ValueError("Gamma sigma squared (γ_σ) must be a non-negative real number.")
        return self._solve_socp(gamma_mu=gamma_mu, gamma_sigma_sq=gamma_sigma_sq)

    def efficient_frontier(self, lambdas: Sequence[float]) -> Tuple[list[float], list[float], npt.NDArray[np.floating]]:
        """Computes the λ-variant efficient frontier."""
        results = [self.solve_lambda_variant(l) for l in lambdas]
        returns = [res.nominal_return for res in results]
        risks = [res.risk for res in results]
        weights = np.column_stack([res.weights for res in results])
        return returns, risks, weights

    def _solve_socp(self, **kwargs) -> OptimizationResult:
        """
        Internal SOCP solver using CVXOPT.

        Variables are `x = [w (N), t (1)]`. The core constraint is `t >= ||S'¹/²w||`.
        """
        # Objective function `c'x`: `min -μᵀw + penalty·t`
        penalty = kwargs.get('lam', kwargs.get('gamma_mu'))
        c_obj = np.hstack([-self.mu, penalty])

        # Conic constraint block for `t >= ||S'¹/²w||`
        # Formulated as h - Gx in the second-order cone
        G_soc = np.zeros((self.N + 1, self.N + 1))
        G_soc[0, self.N] = -1.0
        G_soc[1:, :self.N] = -self.s_prime_sqrt
        h_soc = np.zeros(self.N + 1)
        
        # Combine with linear inequality constraints
        num_lin_ineq = self.G.shape[0] if self.G is not None else 0
        G_ineq_ext = np.hstack([self.G, np.zeros((num_lin_ineq, 1))]) if self.G is not None else np.zeros((0, self.N + 1))
        h_ineq_ext = self.h if self.h is not None else np.zeros(0)
        
        # Add γ-variant constraint: `t <= sqrt(γ_σ)`
        if 'gamma_sigma_sq' in kwargs:
            cap_row = np.zeros((1, self.N + 1))
            cap_row[0, self.N] = 1.0
            G_ineq_ext = np.vstack([G_ineq_ext, cap_row])
            h_ineq_ext = np.hstack([h_ineq_ext, np.sqrt(kwargs['gamma_sigma_sq'])])
            num_lin_ineq += 1

        G_cone = matrix([matrix(G_ineq_ext), matrix(G_soc)])
        h_cone = matrix([matrix(h_ineq_ext), matrix(h_soc)])
        
        # Equality constraints
        A_eq = matrix(np.hstack([self.A, np.zeros((self.A.shape[0], 1))])) if self.A is not None else None
        b_eq = matrix(self.b) if self.b is not None else None

        dims = {'l': num_lin_ineq, 'q': [self.N + 1], 's': []}
        
        sol = solvers.conelp(matrix(c_obj), G_cone, h_cone, dims=dims, A=A_eq, b=b_eq)
        
        if sol['status'] != 'optimal':
            raise RuntimeError(f"SOCP solver failed to find an optimal solution. Status: {sol['status']}")
            
        x_opt = np.array(sol['x']).flatten()
        w_opt = x_opt[:self.N]
        t_opt = x_opt[self.N]
        
        return OptimizationResult(weights=w_opt, nominal_return=self.mu @ w_opt, risk=t_opt)

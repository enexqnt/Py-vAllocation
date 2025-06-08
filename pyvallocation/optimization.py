from __future__ import annotations

import logging
import numbers
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from cvxopt import matrix, solvers

from .bayesian import _cholesky_pd

solvers.options["glpk"] = {"msg_lev": "GLP_MSG_OFF"}
solvers.options["show_progress"] = False

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class OptimizationResult:
    weights: npt.NDArray[np.floating]
    nominal_return: float
    risk: float


class Optimization:
    _I: int
    _mean: np.ndarray
    _G: matrix
    _h: matrix
    _A: matrix
    _b: matrix
    _expected_return_row: matrix

    def _calculate_max_expected_return(self) -> float:
        c = self._expected_return_row.T
        sol = solvers.lp(c, self._G, self._h, self._A, self._b, solver="glpk")
        if sol["status"] != "optimal":
            raise ValueError(
                "Could not solve for maximum expected return; "
                "constraints may be infeasible or unbounded."
            )
        return -sol["primal objective"]


class MeanVariance(Optimization):
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
        if not isinstance(lam, numbers.Real) or lam < 0:
            raise ValueError("Lambda (λ) must be a non-negative real number.")
        return self._solve_socp(lam=lam)

    def solve_gamma_variant(
        self, gamma_mu: float, gamma_sigma_sq: float
    ) -> OptimizationResult:
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
        results = [self.solve_lambda_variant(l) for l in lambdas]
        returns = [res.nominal_return for res in results]
        risks = [res.risk for res in results]
        weights = np.column_stack([res.weights for res in results])
        return returns, risks, weights

    def _solve_socp(self, **kwargs) -> OptimizationResult:
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
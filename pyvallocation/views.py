# entropy_pooling and _dual_objective functions are adapted from fortitudo-tech https://github.com/fortitudo-tech/fortitudo.tech

from collections.abc import Sequence
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from scipy.optimize import Bounds, minimize

import pandas as pd


def _entropy_pooling_dual_objective(
    lagrange_multipliers: np.ndarray,
    log_p_col: np.ndarray,
    lhs: np.ndarray,
    rhs_squeezed: np.ndarray,
) -> Tuple[float, np.ndarray]:
    r"""Calculates the objective function and gradient for the dual formulation of the entropy-pooling problem.

    This function is a core component of the entropy-pooling optimization,
    designed to be called by a numerical solver like `scipy.optimize.minimize`.
    It evaluates the dual of the relative entropy minimization problem, which is
    an unconstrained and convex optimization problem, making it significantly
    more efficient to solve than the primal problem.

    The primal problem seeks a posterior probability vector :math:`\mathbf{q}` that minimizes the
    Kullback-Leibler (KL) divergence from a prior vector :math:`\mathbf{p}`:

    .. math::

       \min_{\mathbf{q}} \sum_{s=1}^S q_s (\log q_s - \log p_s)
       \quad \text{s.t.} \quad \mathbf{A} \mathbf{q} = \mathbf{b},
       \quad \mathbf{G} \mathbf{q} \le \mathbf{h}.

    The Lagrangian for this problem leads to a closed-form solution for
    :math:`\mathbf{q}` in terms of the Lagrange multipliers
    :math:`\boldsymbol{\lambda}`. Substituting this back into the Lagrangian yields the
    dual objective function :math:`\mathcal{G}(\boldsymbol{\lambda})`, which is to be maximized.
    This function computes the *negative* of the dual objective (for minimization)
    and its gradient.

    The posterior probability vector, conditional on the Lagrange multipliers, is:

    .. math::

       \mathbf{q}(\boldsymbol{\lambda}) = \exp(\log \mathbf{p} - 1 - \mathbf{M}^\top \boldsymbol{\lambda})

    where :math:`\mathbf{M}` is the vertically stacked matrix of constraints (`lhs`).
    The dual objective is then:

    .. math::

       \mathcal{G}(\boldsymbol{\lambda}) = -\mathbf{1}^\top \mathbf{q}(\boldsymbol{\lambda}) - \boldsymbol{\lambda}^\top \mathbf{c}

    and its gradient is:

    .. math::

        \nabla \mathcal{G}(\boldsymbol{\lambda}) = \mathbf{M} \mathbf{q}(\boldsymbol{\lambda}) - \mathbf{c}

    where :math:`\mathbf{c}` is the stacked vector of constraint targets (`rhs_squeezed`).
    For numerical stability, the objective and gradient are scaled by a factor of 1000.

    Parameters
    ----------
    lagrange_multipliers : (K,) ndarray
        The vector of Lagrange multipliers, :math:`\boldsymbol{\lambda}`, where :math:`K` is the total number of constraints.
    log_p_col : (S, 1) ndarray
        The natural logarithm of the prior probabilities, :math:`\log \mathbf{p}`.
    lhs : (K, S) ndarray
        The stacked matrix of constraint coefficients, :math:`\mathbf{M}`.
    rhs_squeezed : (K,) ndarray
        The stacked vector of constraint targets, :math:`\mathbf{c}`.

    Returns
    -------
    value : float
        The scaled value of the dual objective function to be minimized.
    gradient : (K,) ndarray
        The scaled gradient of the dual objective function.
    """
    lagrange_multipliers_col = lagrange_multipliers[:, np.newaxis]

    x = np.exp(log_p_col - 1.0 - lhs.T @ lagrange_multipliers_col)

    rhs_vec = np.atleast_1d(rhs_squeezed)
    # The objective function to be minimized is the negative of the dual Lagrangian
    objective_value = -(-np.sum(x) - lagrange_multipliers @ rhs_vec)
    # The gradient is computed for the minimization objective
    gradient_vector = rhs_vec - (lhs @ x).squeeze()

    return 1000.0 * objective_value, 1000.0 * gradient_vector


def entropy_pooling(
    p: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    G: Optional[np.ndarray] = None,
    h: Optional[np.ndarray] = None,
    method: Optional[str] = None,
) -> np.ndarray:
    r"""Computes posterior probabilities using the entropy-pooling algorithm.

    This function implements the Entropy Pooling (EP) methodology developed by
    Meucci (2008) to find a posterior probability distribution, :math:`\mathbf{q}`, that deviates
    minimally from a prior distribution, :math:`\mathbf{p}`, while satisfying a set
    of moment and stress-test constraints. The deviation is quantified by the
    Kullback-Leibler (KL) divergence, or relative entropy.

    The primal optimization problem is formulated as:

    .. math::

       \mathbf{q}^* = \arg\min_{\mathbf{q}} \left\{ \sum_{s=1}^S q_s (\log q_s - \log p_s) \right\}

    subject to a set of linear equality and inequality constraints:

    .. math::
       \begin{aligned}
       \mathbf{A} \mathbf{q} &= \mathbf{b} \quad \text{(equality constraints)} \\
       \mathbf{G} \mathbf{q} &\le \mathbf{h} \quad \text{(inequality constraints)} \\
       \mathbf{1}^\top \mathbf{q} &= 1 \\
       \mathbf{q} &> \mathbf{0}
       \end{aligned}

    Directly solving this high-dimensional constrained problem is computationally
    demanding. Instead, this function solves the corresponding dual problem,
    which is an unconstrained convex optimization over the much lower-dimensional
    space of Lagrange multipliers. The optimal posterior probabilities :math:`\mathbf{q}^*` are then
    analytically recovered from the optimal Lagrange multipliers.

    Parameters
    ----------
    p : (S,) or (S, 1) ndarray
        The prior probability vector :math:`\mathbf{p}`, where `S` is the number of scenarios.
        Must be a valid distribution (non-negative, sums to 1).
    A : (K_eq, S) ndarray
        The matrix defining the equality constraints :math:`\mathbf{A}\mathbf{q} = \mathbf{b}`.
    b : (K_eq,) ndarray
        The vector of target values for the equality constraints.
    G : (K_ineq, S) ndarray, optional
        The matrix defining the inequality constraints :math:`\mathbf{G}\mathbf{q} \le \mathbf{h}`.
    h : (K_ineq,) ndarray, optional
        The vector of upper bounds for the inequality constraints.
    method : {'TNC', 'L-BFGS-B'}, optional
        The optimization algorithm for `scipy.optimize.minimize`. Both methods
        support the necessary box constraints on the Lagrange multipliers
        for inequality views. If None, defaults to 'TNC'.

    Returns
    -------
    q : (S, 1) ndarray
        The posterior probability vector, :math:`\mathbf{q}^*`, as a column vector.

    Raises
    ------
    ValueError
        If an unsupported `method` is specified.

    Notes
    -----
    The constraint that probabilities must sum to one is automatically
    incorporated. Positivity of probabilities is an intrinsic property of the
    solution to the dual problem and does not need to be explicitly enforced.
    This implementation is adapted from the `fortitudo.tech` open-source package
    and adheres to the framework in Meucci (2008), "Fully Flexible Views: Theory and Practice".
    """
    opt_method = method or "TNC"
    if opt_method not in ("TNC", "L-BFGS-B"):
        raise ValueError(f"Method {opt_method} not supported. Choose 'TNC' or 'L-BFGS-B'.")

    p_col = p.reshape(-1, 1)
    # Ensure probabilities sum to 1 to avoid downstream errors.
    p_col /= p_col.sum()

    b_col = b.reshape(-1, 1)

    # The constraint that probabilities sum to 1 is always enforced.
    A_ext = np.vstack([A, np.ones(A.shape[1])])
    b_ext = np.vstack([b_col, [[1.0]]])
    num_equalities = b_ext.shape[0]

    if G is None or h is None:
        current_lhs = A_ext
        current_rhs_stacked = b_ext
        # Lagrange multipliers for equalities are unbounded.
        bounds_lower = [-np.inf] * num_equalities
        bounds_upper = [np.inf] * num_equalities
    else:
        h_col = h.reshape(-1, 1)
        num_inequalities = h_col.shape[0]
        current_lhs = np.vstack((A_ext, G))
        current_rhs_stacked = np.vstack((b_ext, h_col))
        # Lagrange multipliers for inequalities (Gq <= h) must be non-negative.
        bounds_lower = [-np.inf] * num_equalities + [0.0] * num_inequalities
        bounds_upper = [np.inf] * (num_equalities + num_inequalities)

    log_p_col = np.log(p_col)

    initial_lagrange_multipliers = np.zeros(current_lhs.shape[0])
    optimizer_bounds = Bounds(bounds_lower, bounds_upper)

    solution = minimize(
        _entropy_pooling_dual_objective,
        x0=initial_lagrange_multipliers,
        args=(log_p_col, current_lhs, current_rhs_stacked.squeeze()),
        method=opt_method,
        jac=True,
        bounds=optimizer_bounds,
        options={"maxiter": 1000, "maxfun": 10000},
    )

    optimal_lagrange_multipliers_col = solution.x[:, np.newaxis]

    q_posterior = np.exp(log_p_col - 1.0 - current_lhs.T @ optimal_lagrange_multipliers_col)

    return q_posterior


class FlexibleViewsProcessor:
    r"""An engine for incorporating fully flexible views into a prior distribution using entropy pooling.

    This class implements the *Fully Flexible Views* framework, which adjusts a
    prior probability distribution over a set of market scenarios to reflect
    investor views on various moments of the distribution. The adjustment is
    performed by minimizing the relative entropy (Kullback-Leibler divergence)
    between the prior and posterior distributions, subject to constraints
    representing the views.

    This approach is highly versatile, supporting views on means, volatilities,
    correlations, and skewness, expressed as either equalities or inequalities.

    Core Methodologies
    ------------------
    The processor supports two distinct methodologies for applying views:

    1.  **Simultaneous Entropy Pooling (default):** All views are compiled into a
        single set of linear constraints and applied in one optimization step.
        This is the original method proposed by Meucci (2008). While direct, it
        can introduce strong, unintended implicit views, as higher-order moment
        constraints (e.g., on variance) are linearized by fixing lower-order
        moments (e.g., the mean) at their *prior* values.

    2.  **Sequential Entropy Pooling (SeqEP):** When `sequential=True`, views
        are processed in a logical sequence: means -> volatilities -> skews ->
        correlations. The posterior distribution from one step becomes the
        prior for the next. This iterative refinement allows constraints on
        higher-order moments to be linearized using updated, more consistent
        values for lower-order moments, often leading to more intuitive and
        numerically superior results.

    Parameters
    ----------
    prior_returns : (S, N) ndarray or DataFrame, optional
        A matrix of `S` scenarios for `N` assets. If provided, this is used as
        the basis for the prior distribution. `prior_mean` and `prior_cov` are ignored.
    prior_probabilities : (S,) array_like, optional
        Prior probabilities for each scenario. If not provided, a uniform
        distribution (`1/S`) is assumed.
    prior_mean : (N,) array_like, optional
        The prior mean vector. Required alongside `prior_cov` if `prior_returns`
        is not supplied. Scenarios will be synthesized.
    prior_cov : (N, N) array_like, optional
        The prior covariance matrix. Required alongside `prior_mean` if
        `prior_returns` is not supplied.
    distribution_fn : callable, optional
        A function to generate synthetic scenarios, e.g., from a specific
        copula or non-normal distribution. Signature must be
        `f(mean, cov, n_scenarios, rng)`. Defaults to multivariate normal.
    num_scenarios : int, default 10000
        Number of scenarios to synthesize if `prior_returns` is not given.
    random_state : int or numpy.random.Generator, optional
        Seed for the random number generator to ensure reproducibility of
        synthesized scenarios.
    mean_views, vol_views, corr_views, skew_views : mapping or array_like, optional
        Dictionaries specifying views.
        Keys are asset labels (or `('AssetA', 'AssetB')` for correlations).
        Values can be a scalar `x` for an equality view (`== x`) or a
        tuple `('op', x)` for an inequality, e.g., `('<=', 0.20)`.
    sequential : bool, default False
        If `True`, applies Sequential Entropy Pooling (SeqEP). If `False`,
        applies all views simultaneously.

    Attributes
    ----------
    posterior_probabilities : (S, 1) ndarray
        The final posterior probability vector :math:`\mathbf{q}^*`.
    posterior_returns : ndarray or Series
        The posterior mean vector.
    posterior_cov : ndarray or DataFrame
        The posterior covariance matrix.

    Examples
    --------
    >>> # --- Example with historical returns and sequential views ---
    >>> returns_df = pd.DataFrame(np.random.randn(500, 2), columns=['US Equity', 'US Bonds'])
    >>> fvp = FlexibleViewsProcessor(
    ...     prior_returns=returns_df,
    ...     mean_views={'US Equity': ('>=', 0.05)},
    ...     vol_views={'US Equity': ('<=', 0.20)},
    ...     sequential=True
    ... )
    >>> post_mean, post_cov = fvp.get_posterior()
    >>> print("Sequential Posterior Mean:\n", post_mean)

    >>> # --- Example with synthesized scenarios and correlation view ---
    >>> prior_mu = np.array([0.04, 0.01])
    >>> prior_sigma = np.array([[0.02, 0.005], [0.005, 0.002]])
    >>> fvp_synth = FlexibleViewsProcessor(
    ...     prior_mean=prior_mu,
    ...     prior_cov=prior_sigma,
    ...     corr_views={('0', '1'): -0.5}
    ... )
    >>> _, post_cov_synth = fvp_synth.get_posterior()
    >>> print("\nSimultaneous Posterior Correlation:", post_cov_synth[0, 1] / np.sqrt(post_cov_synth[0, 0] * post_cov_synth[1, 1]))
    """

    def __init__(
        self,
        prior_returns: Optional[Union[np.ndarray, "pd.DataFrame"]] = None,
        prior_probabilities: Optional[Union[np.ndarray, "pd.Series"]] = None,
        *,
        prior_mean: Optional[Union[np.ndarray, "pd.Series"]] = None,
        prior_cov: Optional[Union[np.ndarray, "pd.DataFrame"]] = None,
        distribution_fn: Optional[
            Callable[[np.ndarray, np.ndarray, int, Any], np.ndarray]
        ] = None,
        num_scenarios: int = 10000,
        random_state: Any = None,
        mean_views: Any = None,
        vol_views: Any = None,
        corr_views: Any = None,
        skew_views: Any = None,
        sequential: bool = False,
    ):
        if prior_returns is not None:
            if isinstance(prior_returns, pd.DataFrame):
                self.R = prior_returns.values
                self.assets = list(prior_returns.columns)
                self._use_pandas = True
            else:
                self.R = np.atleast_2d(np.asarray(prior_returns, float))
                self.assets = [str(i) for i in range(self.R.shape[1])]
                self._use_pandas = False

            S, N = self.R.shape

            if prior_probabilities is None:
                self.p0 = np.full((S, 1), 1.0 / S)
            else:
                p_array = np.asarray(prior_probabilities, float).ravel()
                if p_array.size != S:
                    raise ValueError(
                        "`prior_probabilities` must match the number of scenarios."
                    )
                self.p0 = p_array.reshape(-1, 1) / p_array.sum()

        else:
            if prior_mean is None or prior_cov is None:
                raise ValueError(
                    "Provide either `prior_returns` or both `prior_mean` and `prior_cov`."
                )

            if isinstance(prior_mean, pd.Series):
                mu = prior_mean.values.astype(float)
                self.assets = list(prior_mean.index)
                self._use_pandas = True
            else:
                mu = np.asarray(prior_mean, float).ravel()
                self.assets = [str(i) for i in range(mu.size)]
                self._use_pandas = False

            if isinstance(prior_cov, pd.DataFrame):
                cov = prior_cov.values.astype(float)
                if not self._use_pandas:
                    self.assets = list(prior_cov.index)
                    self._use_pandas = True
            else:
                cov = np.asarray(prior_cov, float)

            N = mu.size

            # Add small jitter to diagonal for numerical stability if needed
            cov = cov + np.eye(N) * 1e-9

            rng = np.random.default_rng(random_state)

            if distribution_fn is None:
                self.R = rng.multivariate_normal(mu, cov, size=num_scenarios)
            else:
                try:
                    self.R = distribution_fn(mu, cov, num_scenarios, rng)
                except TypeError:
                    self.R = distribution_fn(mu, cov, num_scenarios)

            self.R = np.atleast_2d(np.asarray(self.R, float))
            if self.R.shape != (num_scenarios, N):
                raise ValueError(
                    f"`distribution_fn` must return shape ({num_scenarios}, {N}), got {self.R.shape}."
                )

            S = num_scenarios
            self.p0 = np.full((S, 1), 1.0 / S)

        self.mu0 = (self.R.T @ self.p0).flatten()
        self.cov0 = np.cov(self.R.T, aweights=self.p0.flatten(), bias=True)
        self.var0 = np.diag(self.cov0)

        def _vec_to_dict(vec_like, name):
            if vec_like is None:
                return {}
            if isinstance(vec_like, dict):
                return vec_like
            vec = np.asarray(vec_like, float).ravel()
            if vec.size != len(self.assets):
                raise ValueError(f"`{name}` must have length {len(self.assets)} matching the number of assets.")
            return {a: vec[i] for i, a in enumerate(self.assets)}

        self.mean_views = _vec_to_dict(mean_views, "mean_views")
        self.vol_views = _vec_to_dict(vol_views, "vol_views")
        self.skew_views = _vec_to_dict(skew_views, "skew_views")
        self.corr_views = corr_views or {}
        self.sequential = bool(sequential)

        self.posterior_probabilities = self._compute_posterior_probabilities()

        q = self.posterior_probabilities
        mu_post = (self.R.T @ q).flatten()
        cov_post = np.cov(self.R.T, aweights=q.flatten(), bias=True)

        if self._use_pandas:
            self.posterior_returns = pd.Series(mu_post, index=self.assets)
            self.posterior_cov = pd.DataFrame(
                cov_post, index=self.assets, columns=self.assets
            )
        else:
            self.posterior_returns = mu_post
            self.posterior_cov = cov_post

    @staticmethod
    def _parse_view(v: Any) -> Tuple[str, float]:
        r"""Parses a view value into a standardized (operator, target) tuple.

        This method provides a flexible interface for specifying views, accepting
        either a direct scalar value (implying equality) or an explicit tuple
        containing an operator and a target value.

        Supported formats:
        - A scalar `x` is interpreted as `('==', x)`.
        - A tuple `(op, x)` where `op` is one of `==, >=, <=, >, <`.

        Parameters
        ----------
        v : Any
            The raw view value.

        Returns
        -------
        Tuple[str, float]
            A standardized tuple of (operator, target_value).
        """
        if (
            isinstance(v, (list, tuple))
            and len(v) == 2
            and v[0] in ("==", ">=", "<=", ">", "<")
        ):
            return v[0], float(v[1])
        return "==", float(v)

    def _asset_idx(self, key: Any) -> int:
        """Resolves an asset label to its integer index.

        This utility function maps a user-provided asset identifier (e.g., a
        string from a DataFrame column) to its corresponding zero-based integer
        index for use with internal NumPy arrays.

        Parameters
        ----------
        key : Any
            The asset label to look up.

        Returns
        -------
        int
            The zero-based index of the asset.

        Raises
        ------
        ValueError
            If the asset label is not found.
        """
        try:
            return self.assets.index(key)
        except ValueError:
            if isinstance(key, str) and key.isdigit():
                k_int = int(key)
                try:
                    return self.assets.index(k_int)
                except ValueError:
                    pass
        raise ValueError(f"Asset label '{key}' not recognised.")

    def _build_constraints(
        self,
        view_dict: Dict,
        moment_type: str,
        mu: np.ndarray,
        var: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[float], List[np.ndarray], List[float]]:
        """Constructs linear constraint matrices from a dictionary of views.

        This method is the engine for translating high-level, human-readable
        views into the low-level linear algebra constructs (`A, b, G, h`)
        required by the `entropy_pooling` solver. It handles the linearization
        of inherently non-linear views (e.g., variance, correlation) by fixing
        lower-order moments at their current best estimates.

        Parameters
        ----------
        view_dict : Dict
            The dictionary of views for a specific moment (e.g., `self.vol_views`).
        moment_type : str
            The type of moment being constrained ('mean', 'vol', 'skew', 'corr').
        mu : (N,) ndarray
            The current estimate of the mean vector, used to linearize constraints
            on higher-order moments.
        var : (N,) ndarray
            The current estimate of the variance vector, used for linearization.

        Returns
        -------
        A_eq : list
            List of row vectors for equality constraints.
        b_eq : list
            List of target values for equality constraints.
        G_ineq : list
            List of row vectors for inequality constraints.
        h_ineq : list
            List of target values for inequality constraints.

        Raises
        ------
        ValueError
            If an unsupported `moment_type` is provided.

        Notes
        -----
        Linearization strategy:
        - **Variance:** :math:`\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2`.
          The constraint on :math:`\text{Var}(X)` becomes a linear constraint on
          :math:`\mathbb{E}[X^2]` by fixing :math:`\mathbb{E}[X]` to its current value from `mu`.
        - **Correlation:** :math:`\text{Corr}(X, Y) = (\mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]) / (\sigma_X \sigma_Y)`.
          This becomes a linear constraint on :math:`\mathbb{E}[XY]` by fixing all
          other terms using `mu` and `var`.
        - **Skewness:** A similar linearization is applied using the definition of skewness.
        """
        A_eq, b_eq, G_ineq, h_ineq = [], [], [], []
        R = self.R

        def add(op, row, raw):
            if op == "==":
                A_eq.append(row)
                b_eq.append(raw)
            elif op in ("<=", "<"):
                G_ineq.append(row)
                h_ineq.append(raw)
            else: #'>=', '>'
                G_ineq.append(-row)
                h_ineq.append(-raw)

        if moment_type == "mean":
            for key, vw in view_dict.items():
                op, tgt = self._parse_view(vw)
                if isinstance(key, tuple) and len(key) == 2:
                    a1, a2 = key
                    i, j = self._asset_idx(a1), self._asset_idx(a2)
                    row = R[:, i] - R[:, j]
                    add(op, row, tgt)
                else:
                    idx = self._asset_idx(key)
                    add(op, R[:, idx], tgt)

        elif moment_type == "vol":
            for asset, vw in view_dict.items():
                op, tgt = self._parse_view(vw)
                idx = self._asset_idx(asset)
                raw = tgt**2 + mu[idx] ** 2
                add(op, R[:, idx] ** 2, raw)

        elif moment_type == "skew":
            for asset, vw in view_dict.items():
                op, tgt = self._parse_view(vw)
                idx = self._asset_idx(asset)
                s = np.sqrt(var[idx])
                raw = tgt * s**3 + 3 * mu[idx] * var[idx] + mu[idx] ** 3
                add(op, R[:, idx] ** 3, raw)

        elif moment_type == "corr":
            for (a1, a2), vw in view_dict.items():
                op, tgt = self._parse_view(vw)
                i = self._asset_idx(a1)
                j = self._asset_idx(a2)
                s_i, s_j = np.sqrt(var[i]), np.sqrt(var[j])
                raw = tgt * s_i * s_j + mu[i] * mu[j]
                add(op, R[:, i] * R[:, j], raw)
        else:
            raise ValueError(f"Unknown moment type '{moment_type}'.")

        return A_eq, b_eq, G_ineq, h_ineq

    def _compute_posterior_probabilities(self) -> np.ndarray:
        """Orchestrates the computation of posterior probabilities based on the selected methodology.

        This internal method directs the flow of the entropy pooling process.
        It either bundles all views for a single, simultaneous optimization or
        iteratively applies blocks of views for the sequential approach.

        Returns
        -------
        np.ndarray
            The computed (S, 1) posterior probability vector `q`.
        """
        R, p0 = self.R, self.p0
        mu_cur, var_cur = self.mu0.copy(), self.var0.copy()

        def do_ep(prior_probs, A_eq_list, b_eq_list, G_ineq_list, h_ineq_list):
            S = R.shape[0]

            A = np.vstack(A_eq_list) if A_eq_list else np.zeros((0, S))
            b = np.array(b_eq_list, float).ravel() if b_eq_list else np.zeros(0)

            if G_ineq_list:
                G = np.vstack(G_ineq_list)
                h = np.array(h_ineq_list, float).ravel()
            else:
                G, h = None, None

            return entropy_pooling(prior_probs, A, b, G, h)

        if not any((self.mean_views, self.vol_views, self.skew_views, self.corr_views)):
            return p0

        view_blocks = [
            ("mean", self.mean_views),
            ("vol", self.vol_views),
            ("skew", self.skew_views),
            ("corr", self.corr_views),
        ]

        if self.sequential:
            q_last = p0.copy()
            # In sequential mode, iterate through view blocks
            for mtype, vd in view_blocks:
                if vd:
                    Aeq, beq, G, h = self._build_constraints(vd, mtype, mu_cur, var_cur)
                    if not Aeq and not G:
                        continue
                    q_last = do_ep(q_last, Aeq, beq, G, h)

                    # Update moments for the next iteration
                    mu_cur = (R.T @ q_last).flatten()
                    var_cur = ((R - mu_cur) ** 2).T @ q_last
                    var_cur = var_cur.flatten()
            return q_last
        else:
            # In simultaneous mode, aggregate all constraints
            A_all, b_all, G_all, h_all = [], [], [], []
            for mtype, vd in view_blocks:
                if vd:
                    Aeq, beq, G, h = self._build_constraints(vd, mtype, mu_cur, var_cur)
                    A_all.extend(Aeq)
                    b_all.extend(beq)
                    G_all.extend(G)
                    h_all.extend(h)

            return do_ep(p0, A_all, b_all, G_all, h_all)

    def get_posterior_probabilities(self) -> np.ndarray:
        """Returns the computed posterior probability vector.

        This vector represents the updated probabilities for each market scenario
        after incorporating all specified views through entropy pooling.

        Returns
        -------
        np.ndarray
            The (S, 1) posterior probability vector `q`.
        """
        return self.posterior_probabilities

    def get_posterior(
        self,
    ) -> Tuple[Union[np.ndarray, pd.Series], Union[np.ndarray, pd.DataFrame]]:
        """Returns the posterior mean and covariance matrix.

        These are the primary outputs of the Flexible Views framework, representing
        the first two moments of the asset return distribution implied by the
        posterior probabilities. They can be directly used in portfolio
        optimization.

        Returns
        -------
        posterior_mean : ndarray or pandas.Series
            The posterior mean vector of asset returns.
        posterior_cov : ndarray or pandas.DataFrame
            The posterior covariance matrix of asset returns.
        """
        return self.posterior_returns, self.posterior_cov


class BlackLittermanProcessor:
    r"""Implements the Bayesian Black-Litterman model for incorporating mean views.

    This processor updates a prior distribution of asset returns,
    :math:`\mathcal{N}(\boldsymbol{\pi}, \boldsymbol{\Sigma})`, with user-specified
    views on expected returns. The model provides a robust framework for blending
    market equilibrium (the prior) with subjective forecasts (the views) in a
    mathematically consistent manner.

    The model is based on Bayesian mixed estimation. Given a set of `K` linear
    views on the `N` asset means, expressed as:

    .. math::

       \mathbf{P}\boldsymbol{\mu} = \mathbf{Q} + \boldsymbol{\varepsilon},
       \quad \text{where} \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Omega})

    - :math:`\mathbf{P}` is the :math:`K \times N` "pick" matrix, defining the assets in each view.
    - :math:`\mathbf{Q}` is the :math:`K \times 1` vector of expected returns for the views.
    - :math:`\boldsymbol{\Omega}` is the :math:`K \times K` covariance matrix of the view errors,
      representing the confidence in each view.

    The resulting posterior mean :math:`\boldsymbol{\mu}^\star` and covariance
    :math:`\boldsymbol{\Sigma}^\star` are given by the Black-Litterman formulas:

    .. math::
       :label: bl_posterior_main

       \begin{aligned}
       \boldsymbol{\mu}^{\star}
       &= \bigl((\tau\boldsymbol{\Sigma})^{-1} + \mathbf{P}^\top\boldsymbol{\Omega}^{-1}\mathbf{P}\bigr)^{-1}
          \bigl((\tau\boldsymbol{\Sigma})^{-1}\boldsymbol{\pi} + \mathbf{P}^\top\boldsymbol{\Omega}^{-1}\mathbf{Q}\bigr) \\
       \boldsymbol{\Sigma}^{\star}
       &= \boldsymbol{\Sigma} + \bigl((\tau\boldsymbol{\Sigma})^{-1} + \mathbf{P}^\top\boldsymbol{\Omega}^{-1}\mathbf{P}\bigr)^{-1}
       \end{aligned}

    where :math:`\tau` is a scalar controlling the weight on the prior covariance. This
    class supports multiple methods for specifying the prior :math:`\boldsymbol{\pi}` and
    the view confidence matrix :math:`\boldsymbol{\Omega}`.

    Parameters
    ----------
    prior_cov : (N, N) array_like
        The prior covariance matrix, :math:`\boldsymbol{\Sigma}`.
    prior_mean : (N,) array_like, optional
        A user-specified prior mean vector, :math:`\boldsymbol{\pi}`. (Mutually exclusive with
        `market_weights` and `pi`).
    market_weights : (N,) array_like, optional
        Market capitalization weights for deriving the CAPM equilibrium prior returns via
        reverse optimization: :math:`\boldsymbol{\pi} = \delta\boldsymbol{\Sigma}\mathbf{w}`.
    risk_aversion : float, default 1.0
        The risk-aversion coefficient, :math:`\delta > 0`, for reverse optimization.
    tau : float, default 0.05
        A scalar, :math:`\tau`, that scales the prior covariance matrix, reflecting
        uncertainty in the prior mean. Typically a small value.
    idzorek_use_tau : bool, default True
        If `True` when using `omega='idzorek'`, the view variance is scaled by
        :math:`\tau`, aligning with He & Litterman. If `False`, it is not.
    pi : (N,) array_like, optional
        Alias for `prior_mean`.
    mean_views : mapping or array_like, optional
        A dictionary or array specifying equality mean views.
        - Absolute view: `{'AssetA': 0.05}`
        - Relative view: `{('AssetA', 'AssetB'): 0.01}` for :math:`\mu_A - \mu_B = 0.01`
    view_confidences : float or sequence or dict, optional
        Confidence levels :math:`c_k \in (0, 1]` for each view, used when
        `omega='idzorek'`.
    omega : {"idzorek"} or array_like, optional
        Method for specifying the view covariance matrix :math:`\boldsymbol{\Omega}`.
        - `"idzorek"`: Derives :math:`\boldsymbol{\Omega}` from `view_confidences`.
        - `array_like`: A user-provided vector (for a diagonal :math:`\boldsymbol{\Omega}`) or a full matrix.
        - If `None`, a default diagonal matrix is constructed based on the variance of the view portfolios.
    verbose : bool, default False
        If `True`, prints intermediate diagnostic information.

    Attributes
    ----------
    posterior_mean : ndarray or pandas.Series
        The posterior mean vector, :math:`\boldsymbol{\mu}^{\star}`.
    posterior_cov : ndarray or pandas.DataFrame
        The posterior covariance matrix, :math:`\boldsymbol{\Sigma}^{\star}`.

    """
    def get_posterior(
        self,
    ) -> Tuple[Union[np.ndarray, "pd.Series"], Union[np.ndarray, "pd.DataFrame"]]:
        """Returns the posterior mean and covariance matrix.

        This is the primary accessor method for retrieving the results of the
        Black-Litterman update. The output format (NumPy or Pandas) matches the
        input format of `prior_cov`.

        Returns
        -------
        posterior_mean : ndarray or pandas.Series
            The posterior mean vector, :math:`\boldsymbol{\mu}^{\star}`.
        posterior_cov : ndarray or pandas.DataFrame
            The posterior covariance matrix, :math:`\boldsymbol{\Sigma}^{\star}`.
        """
        return self._posterior_mean, self._posterior_cov

    def __init__(
        self,
        *,
        prior_cov: Union[np.ndarray, "pd.DataFrame"],
        prior_mean: Optional[Union[np.ndarray, "pd.Series"]] = None,
        market_weights: Optional[Union[np.ndarray, "pd.Series"]] = None,
        risk_aversion: float = 1.0,
        tau: float = 0.05,
        idzorek_use_tau: bool = True,
        pi: Optional[Union[np.ndarray, "pd.Series"]] = None,
        mean_views: Any = None,
        view_confidences: Any = None,
        omega: Any = None,
        verbose: bool = False,
    ) -> None:

        self._is_pandas: bool = isinstance(prior_cov, pd.DataFrame)
        self._assets: List[Union[str, int]] = (
            list(prior_cov.index)
            if self._is_pandas
            else list(range(np.asarray(prior_cov).shape[0]))
        )
        self._sigma: np.ndarray = np.asarray(prior_cov, dtype=float)
        n_assets: int = self._sigma.shape[0]

        if self._sigma.shape != (n_assets, n_assets):
            raise ValueError("prior_cov must be a square (N, N) matrix.")
        if not np.allclose(self._sigma, self._sigma.T, atol=1e-8):
            warnings.warn("prior_cov is not symmetric; symmetrising for calculation.")
            self._sigma = 0.5 * (self._sigma + self._sigma.T)

        if risk_aversion <= 0.0:
            raise ValueError("risk_aversion must be positive.")
        self._tau: float = float(tau)

        # Determine the prior mean vector from one of the three sources
        if sum(x is not None for x in [pi, market_weights, prior_mean]) != 1:
            raise ValueError("Specify exactly one of: pi, market_weights, or prior_mean.")

        if pi is not None:
            self._pi = np.asarray(pi, dtype=float).reshape(-1, 1)
            src = "user-supplied 'pi'"
        elif market_weights is not None:
            weights = np.asarray(market_weights, dtype=float).ravel()
            if weights.size != n_assets:
                raise ValueError("market_weights length must match the number of assets.")
            weights /= weights.sum()
            self._pi = risk_aversion * self._sigma @ weights.reshape(-1, 1)
            src = "reverse-optimized from 'market_weights'"
        else: # prior_mean is not None
            self._pi = np.asarray(prior_mean, dtype=float).reshape(-1, 1)
            src = "user-supplied 'prior_mean'"

        if verbose:
            print(f"[BL] Prior mean source: {src}.")

        def _vec_to_dict(vec_like):
            if vec_like is None:
                return {}
            if isinstance(vec_like, dict):
                return vec_like
            vec = np.asarray(vec_like, float).ravel()
            if vec.size != n_assets:
                raise ValueError(f"`mean_views` array must have length {n_assets}.")
            return {self._assets[i]: vec[i] for i in range(n_assets)}

        mv_dict = _vec_to_dict(mean_views)
        self._p, self._q, view_keys = self._build_views(mv_dict)
        self._k: int = self._p.shape[0]
        if verbose and self._k > 0:
            print(f"[BL] Built P matrix ({self._p.shape}) and Q vector ({self._q.shape}) for {self._k} views.")

        self._conf: Optional[np.ndarray] = self._parse_conf(view_confidences, view_keys)
        self._idzorek_use_tau = bool(idzorek_use_tau)
        self._omega: np.ndarray = self._build_omega(omega, verbose)

        self._posterior_mean, self._posterior_cov = self._compute_posterior(verbose)
        if self._is_pandas:
            self._posterior_mean = pd.Series(self._posterior_mean.flatten(), index=self._assets)
            self._posterior_cov = pd.DataFrame(
                self._posterior_cov, index=self._assets, columns=self._assets
            )

    def _asset_index(self, label: Union[str, int]) -> int:
        """Resolves an asset label to its integer index."""
        try:
            return self._assets.index(label)
        except ValueError:
            if isinstance(label, str) and label.isdigit():
                idx = int(label)
                if idx < len(self._assets):
                    return idx
        raise ValueError(f"Unknown asset label '{label}'.")

    def _build_views(
        self, mean_views: Dict[Any, Any]
    ) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
        """Constructs the P and Q matrices from the mean_views dictionary."""
        rows: List[np.ndarray] = []
        targets: List[float] = []
        keys: List[Any] = []
        n = len(self._assets)

        for key, value in mean_views.items():
            if isinstance(value, Sequence) and not isinstance(value, str):
                if len(value) != 1 and not (len(value) == 2 and value[0] == '=='):
                    raise ValueError(
                        "Only equality views are supported. Use a scalar value or ('==', value)."
                    )
                target = float(value[-1])
            else:
                target = float(value)

            row = np.zeros(n)
            if isinstance(key, tuple):  # Relative view
                asset_i, asset_j = key
                i_idx, j_idx = self._asset_index(asset_i), self._asset_index(asset_j)
                row[i_idx], row[j_idx] = 1.0, -1.0
            else:  # Absolute view
                idx = self._asset_index(key)
                row[idx] = 1.0

            rows.append(row)
            targets.append(target)
            keys.append(key)

        p_mat = np.vstack(rows) if rows else np.zeros((0, n))
        q_vec = np.array(targets, dtype=float).reshape(-1, 1) if targets else np.zeros((0, 1))
        return p_mat, q_vec, keys

    @staticmethod
    def _parse_conf(conf: Any, keys: List[Any]) -> Optional[np.ndarray]:
        """Parses the view_confidences input into a consistent numpy array."""
        if conf is None:
            return None
        k = len(keys)
        if isinstance(conf, (int, float)):
            return np.full(k, float(conf))
        if isinstance(conf, dict):
            return np.array([float(conf.get(key, 1.0)) for key in keys])
        arr = np.asarray(conf, dtype=float).ravel()
        if arr.size != k:
            raise ValueError(f"view_confidences length mismatch: expected {k}, got {arr.size}.")
        return arr

    def _build_omega(self, omega: Any, verbose: bool) -> np.ndarray:
        """Constructs the view covariance matrix Omega."""
        if self._k == 0:
            return np.zeros((0, 0))

        tau_sigma = self._tau * self._sigma

        if isinstance(omega, str) and omega.lower() == "idzorek":
            if self._conf is None:
                raise ValueError("Idzorek method requires 'view_confidences' to be set.")
            diag = []
            base_sigma = tau_sigma if self._idzorek_use_tau else self._sigma
            for i, conf_level in enumerate(self._conf):
                p_i = self._p[i: i + 1]  # (1, N)
                view_var = (p_i @ base_sigma @ p_i.T).item()
                c = np.clip(conf_level, 1e-6, 1.0 - 1e-6)
                # Idzorek's alpha is 1/c - 1, but this form is more stable
                factor = (1.0 - c) / c
                diag.append(factor * view_var)
            omega_mat = np.diag(diag)
            if verbose:
                suffix = "τΣ" if self._idzorek_use_tau else "Σ"
                print(f"[BL] Ω constructed via Idzorek method (base variance from {suffix}).")
            return omega_mat

        if omega is None:
            # Default method from He & Litterman: diagonal matrix of view variances
            omega_mat = np.diag(np.diag(self._p @ tau_sigma @ self._p.T))
            if verbose:
                print("[BL] Ω constructed as diag(P(τΣ)Pᵀ).")
            return omega_mat

        omega_arr = np.asarray(omega, dtype=float)
        if omega_arr.ndim == 1 and omega_arr.size == self._k:
            omega_mat = np.diag(omega_arr)
        elif omega_arr.shape == (self._k, self._k):
            omega_mat = omega_arr
        else:
            raise ValueError(
                "omega must be 'idzorek', a length-K vector (diagonal), or a KxK matrix."
            )
        if verbose:
            print("[BL] Using user-supplied Ω matrix.")
        return omega_mat

    def _compute_posterior(self, verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the posterior mean and covariance using the Black-Litterman formulas."""
        if self._k == 0:
            if verbose:
                print("[BL] No views provided. Posterior is identical to prior.")
            return self._pi.flatten(), self._sigma

        tau_sigma = self._tau * self._sigma
        P = self._p
        Omega = self._omega

        # Invert (P * tau_sigma * P' + Omega)
        # Using solve for better numerical stability than direct inversion
        middle_term_inv = np.linalg.inv(P @ tau_sigma @ P.T + Omega)

        # Posterior mean
        mean_adjustment = tau_sigma @ P.T @ middle_term_inv @ (self._q - P @ self._pi)
        posterior_mean = self._pi + mean_adjustment

        # Posterior covariance
        cov_adjustment = tau_sigma @ P.T @ middle_term_inv @ P @ tau_sigma
        posterior_cov = self._sigma + tau_sigma - cov_adjustment
        posterior_cov = 0.5 * (posterior_cov + posterior_cov.T)  # Enforce symmetry

        if verbose:
            print("[BL] Posterior mean and covariance computed.")
        return posterior_mean, posterior_cov
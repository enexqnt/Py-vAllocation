# entropy_pooling and _dual_objective functions are adapted from fortitudo-tech https://github.com/fortitudo-tech/fortitudo.tech

from collections.abc import Sequence
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import Bounds, minimize

import pandas as pd

def _entropy_pooling_dual_objective(
    lagrange_multipliers: np.ndarray,
    log_p_col: np.ndarray,
    lhs: np.ndarray,
    rhs_squeezed: np.ndarray,
) -> Tuple[float, np.ndarray]:
    r"""Return dual objective value and gradient for entropy pooling.

    This function computes the dual objective function and its gradient, which are
    necessary for the optimization process in Entropy Pooling (EP). The core idea
    of EP is to find a new probability distribution that incorporates user views
    while minimizing the Kullback-Leibler (KL) divergence (relative entropy) from
    a prior distribution. This constrained optimization problem
    can be efficiently solved by minimizing its dual formulation.

    Let :math:`p^{(0)} \in (0,1)^{S}` be the prior probabilities, where :math:`S`
    is the number of scenarios. Let the constraints from the views be represented
    by a matrix :math:`A \in \mathbb{R}^{K \times S}` and a vector
    :math:`b \in \mathbb{R}^{K}`, where :math:`K` is the total number of
    constraints (equality and inequality). For a vector of Lagrange multipliers
    :math:`\lambda \in \mathbb{R}^{K}` (``lagrange_multipliers``), the intermediate
    variable :math:`x(\lambda)` is defined as:

    .. math::

       x(\lambda) \;:=\; \exp\bigl(\log p^{(0)} - 1 - A^\top \lambda\bigr),

    where :math:`\log p^{(0)}` corresponds to ``log_p_col`` and :math:`A`
    corresponds to ``lhs``. This formulation for :math:`x(\lambda)` arises from the
    first-order optimality conditions of the Lagrangian in the primal entropy
    minimization problem.

    The dual objective function, denoted :math:`\varphi(\lambda)`, which is
    strictly convex, is given by:

    .. math::

       \varphi(\lambda) \;=\; \mathbf 1^\top x(\lambda) + \lambda^\top b.

    Here, :math:`\mathbf{1}` is a vector of ones, and :math:`b` corresponds to
    ``rhs_squeezed``. The term :math:`\mathbf 1^\top x(\lambda)` represents the sum
    of the elements of :math:`x(\lambda)`, and :math:`\lambda^\top b` is the dot
    product of the Lagrange multipliers and the constraint targets.

    The gradient of the dual objective function, :math:`\nabla \varphi(\lambda)`, is
    derived from the dual formulation and used by the optimizer for efficient
    minimization. It is given by:

    .. math::

       \nabla \varphi(\lambda) = b - A\,x(\lambda).

    A scaling factor of ``1e3`` is applied to both the objective value and the
    gradient. This is a common numerical practice to improve the stability of the
    optimization algorithm (e.g., preventing very small numbers from causing
    precision issues), and it does not affect the location of the minimizer
    for the dual problem.

    Parameters
    ----------
    lagrange_multipliers : (K,) ndarray
        The current vector of Lagrange multipliers, :math:`\lambda`, at which the
        objective and gradient are to be evaluated.
    log_p_col : (S, 1) ndarray
        The natural logarithm of the prior probabilities :math:`p^{(0)}`, provided
        as a column vector.
    lhs : (K, S) ndarray
        The left-hand side matrix :math:`A` representing the coefficients of the
        linear constraints. This matrix combines both equality and inequality
        constraints.
    rhs_squeezed : (K,) ndarray
        The right-hand side vector :math:`b` representing the target values for the
        linear constraints. This vector combines targets for both equality and
        inequality constraints.

    Returns
    -------
    value : float
        The scaled value of the dual objective function, i.e., ``1e3 * varphi(lambda)``.
    gradient : (K,) ndarray
        The scaled gradient of the dual objective function, i.e., ``1e3 * nabla varphi(lambda)``.

    Notes
    -----
    This function is intended for internal use by the `scipy.optimize.minimize`
    solver within the `entropy_pooling` function. It is based on the dual problem
    formulation for entropy minimization as described in.
    """
    lagrange_multipliers_col = lagrange_multipliers[:, np.newaxis]

    x = np.exp(log_p_col - 1.0 - lhs.T @ lagrange_multipliers_col)

    rhs_vec = np.atleast_1d(rhs_squeezed)
    objective_value = np.sum(x) + lagrange_multipliers @ rhs_vec
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
    r"""Return posterior probabilities via the entropy‑pooling algorithm.

    This function serves as a wrapper around :func:`scipy.optimize.minimize` to
    solve the dual optimization problem of *Entropy Pooling* (EP), a method
    developed by Attilio Meucci. EP aims to find a new probability
    distribution that is as "close" as possible to a given prior distribution,
    while satisfying a set of linear constraints (views). The "closeness"
    is measured by the Kullback-Leibler (KL) divergence, also known as relative entropy.

    The problem is formulated as minimizing the relative entropy
    :math:`D_{\mathrm{KL}}(q\,\|\,p^{(0)})` subject to linear equality constraints
    :math:`Eq = b` and inequality constraints :math:`Gq \le h`.
    This function solves the dual of this problem using numerical optimization.

    Equality constraints are represented by the matrix ``A`` and vector ``b``.
    Inequality constraints are represented by ``G`` and ``h``. For inequality
    constraints :math:`Gq \le h`, the corresponding Lagrange multipliers are
    constrained to be non-negative.

    The optimization is performed using quasi-Newton methods from `scipy.optimize.minimize`.
    Only ``'TNC'`` (Truncated Newton) and ``'L-BFGS-B'``
    (Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm with box
    constraints) are supported, as they allow for box bounds on the Lagrange
    multipliers.

    Parameters
    ----------
    p : (S,) or (S, 1) ndarray
        The prior probability vector, :math:`p^{(0)} \in (0,1)^S`, where :math:`S`
        is the number of scenarios. This vector is typically uniform
        but can be any valid probability distribution (i.e., elements sum to 1 and are positive).
    A : ndarray
        The matrix :math:`E` for the equality constraints, :math:`Eq = b`. Its
        shape is :math:`(K_{eq}, S)`, where :math:`K_{eq}` is the number of
        equality constraints.
    b : ndarray
        The target vector :math:`b` for the equality constraints. Its shape is
        :math:`(K_{eq},)`.
    G : ndarray, optional
        The matrix :math:`G` for the inequality constraints, :math:`Gq \le h`.
        Its shape is :math:`(K_{ineq}, S)`, where :math:`K_{ineq}` is the number
        of inequality constraints. Defaults to *None* if no inequality constraints
        are present.
    h : ndarray, optional
        The target vector :math:`h` for the inequality constraints. Its shape is
        :math:`(K_{ineq},)`. Defaults to *None* if no inequality constraints are
        present.
    method : {'TNC', 'L-BFGS-B'}, optional
        The optimization method to be used by `scipy.optimize.minimize`.
        Supported options are `'TNC'` and `'L-BFGS-B'`. If *None*, the function
        defaults to `'TNC'`, which is generally faster for the current SciPy
        build.

    Returns
    -------
    q : (S, 1) ndarray
        The posterior probability vector, :math:`q`, in column-vector form,
        satisfying the given constraints while minimizing relative entropy to `p`.

    Raises
    ------
    ValueError
        If an unsupported `method` is specified.

    Notes
    -----
    This function is an adaptation from the `fortitudo.tech` open-source package
    (https://github.com/fortitudo-tech/fortitudo.tech). The core methodology is
    described in Meucci (2008). The dual problem is minimized, and the
    primal solution (posterior probabilities) is recovered from the optimal Lagrange
    multipliers.
    """
    opt_method = method or "TNC"
    if opt_method not in ("TNC", "L-BFGS-B"):
        raise ValueError(
            f"Method {opt_method} not supported. Choose 'TNC' or 'L-BFGS-B'."
        )

    p_col = p.reshape(-1, 1)
    b_col = b.reshape(-1, 1)

    num_equalities = b_col.shape[0]

    if G is None or h is None:
        current_lhs = A
        current_rhs_stacked = b_col
        bounds_lower = [-np.inf] * num_equalities
        bounds_upper = [np.inf] * num_equalities
    else:
        h_col = h.reshape(-1, 1)
        num_inequalities = h_col.shape[0]
        current_lhs = np.vstack((A, G))
        current_rhs_stacked = np.vstack((b_col, h_col))
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

    q_posterior = np.exp(
        log_p_col - 1.0 - current_lhs.T @ optimal_lagrange_multipliers_col
    )

    return q_posterior

class FlexibleViewsProcessor:
    r"""Entropy‑pooling engine with fully flexible moment views.

    The `FlexibleViewsProcessor` class provides a robust framework for adjusting a
    discrete **prior** distribution of multivariate asset returns to incorporate
    user-specified *views*. This is achieved by minimizing the relative entropy
    (Kullback-Leibler divergence) between the prior and the new, "posterior"
    distribution. The class implements the *Fully Flexible Views*
    methodology proposed by Attilio Meucci, supporting views on
    various statistical moments, including **means**, **variances**, **skewnesses**,
    and **correlations**. It supports both *simultaneous* (“one‑shot”) and
    *iterated* (block‑wise) entropy pooling.

    Mathematical background
    -----------------------
    Let :math:`R \in \mathbb{R}^{S\times N}` be a scenario matrix representing :math:`S`
    scenarios for :math:`N` asset returns, with associated prior probabilities
    :math:`p^{(0)} \in (0,1)^S`. The entropy pooling problem is formally stated as:

    .. math::

       \min_{q \in \Delta_S} &\; D_{\mathrm{KL}}(q\,\|\,p^{(0)}) \\
       \text{s.t.} &\; Eq = b, \\
                   &\; Gq \le h,

    where :math:`\Delta_S` denotes the probability simplex (i.e., :math:`q_s > 0` for
    all :math:`s` and :math:`\sum_{s=1}^S q_s = 1`). The matrices :math:`E` and :math:`G`,
    along with vectors :math:`b` and :math:`h`, encode the user's views as linear
    constraints on the posterior probabilities :math:`q`. The minimization of this
    problem is efficiently performed by minimizing its dual formulation, handled by
    the internal helper function :py:func:`_entropy_pooling_dual_objective`.

    For practitioners
    ~~~~~~~~~~~~~~~~~
    * **Plug‑and‑play Scenario Handling:** Users can either provide historical
        returns directly or specify prior mean and covariance, allowing the
        processor to synthesize scenarios. The output includes posterior moments
        (mean, covariance) and the adjusted probabilities.
    * **Sequential Updating (Iterated EP):** By setting `sequential=True`, the
        class applies view blocks (e.g., mean views, then volatility views) in a
        predefined order (*mean → vol → skew → corr*). This iterated approach,
        also known as Sequential Entropy Pooling (SeqEP), can lead to
        significantly better solutions and ensure logical consistency, especially
        when views on higher-order moments (like variance or skewness) implicitly
        depend on lower-order moments (like the mean).
        The original EP approach might introduce strong implicit views by fixing
        parameters to their prior values.
    * **Flexible Inequality Views:** Views can be specified not only as equalities
        but also as inequalities (e.g., '>=', '<=', '>', '<'). For instance,
        `vol_views={'Equity US': ('<=', 0.20)}` sets an upper bound on the
        volatility. Equality is the default if no operator is specified.

    Parameters
    ----------
    prior_returns : (S, N) ndarray or DataFrame, optional
        A matrix or DataFrame of historical or simulated prior scenarios. `S` is the
        number of scenarios, and `N` is the number of assets/risk factors. If
        provided, `prior_mean` and `prior_cov` are ignored.
    prior_probabilities : (S,) array_like, optional
        A vector of prior scenario weights. If not provided, a uniform distribution
        (i.e., `1/S` for each scenario) is assumed.
    prior_mean : (N,) array_like, optional
        The mean vector used for synthesizing scenarios when `prior_returns` is not supplied.
    prior_cov : (N, N) array_like, optional
        The covariance matrix used for synthesizing scenarios when `prior_returns` is not supplied.
    distribution_fn : callable, optional
        A custom function for generating synthetic scenarios when `prior_returns`
        is not supplied. The function signature should be
        `f(mu: np.ndarray, cov: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray`
        returning an `(n, N)` array. If omitted,
        :py:func:`numpy.random.Generator.multivariate_normal` is used.
    num_scenarios : int, default ``10000``
        The number of synthetic scenarios to generate if `prior_returns` is not
        provided. Defaults to 10,000.
    random_state : int or numpy.random.Generator, optional
        A seed or a `numpy.random.Generator` instance to ensure reproducibility
        when generating synthetic scenarios via `distribution_fn` or
        `numpy.random.multivariate_normal`.
    mean_views, vol_views, corr_views, skew_views : mapping or array_like, optional
        View payloads. Keys are asset labels (or pairs for correlations);
        values are either a scalar *x* (equality) or a 2‑tuple such as
        ``('>=', x)``.
    sequential : bool, default ``False``
        If ``True``, views are applied sequentially (iterated EP). The views are
        processed in the order *mean → vol → skew → corr*. If ``False``,
        all views are processed simultaneously in a single EP optimization.

    Attributes
    ----------
    posterior_probabilities : (S, 1) ndarray
        The optimal probability vector :math:`q`.
    posterior_returns : ndarray or Series
        The posterior mean.
    posterior_cov : ndarray or DataFrame
        The posterior covariance.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.stats import multivariate_normal
    >>>
    >>> # Example with historical returns
    >>> np.random.seed(42)
    >>> returns_data = np.random.randn(100, 3) # S=100 scenarios, N=3 assets
    >>> return_df = pd.DataFrame(returns_data, columns=['Asset A', 'Asset B', 'Asset C'])
    >>>
    >>> # Initialize with historical returns and views
    >>> fp_hist = FlexibleViewsProcessor(
    ...     prior_returns=return_df,
    ...     mean_views={'Asset A': 0.05, 'Asset B': ('>=', 0.01)},
    ...     vol_views={'Asset A': ('<=', 0.15)},
    ...     sequential=True, # Apply views sequentially
    ... )
    >>> q_hist = fp_hist.get_posterior_probabilities()
    >>> mu_post_hist, cov_post_hist = fp_hist.get_posterior()
    >>> print("Posterior Mean (Historical):", mu_post_hist)
    >>> print("Posterior Covariance (Historical):\n", cov_post_hist)
    >>>
    >>> # Example with synthesized scenarios
    >>> prior_mu = np.array([0.01, 0.02])
    >>> prior_sigma = np.array([[0.01, 0.005], [0.005, 0.015]])
    >>>
    >>> # Initialize with mean and covariance, synthesizing 5000 scenarios
    >>> fp_synth = FlexibleViewsProcessor(
    ...     prior_mean=prior_mu,
    ...     prior_cov=prior_sigma,
    ...     num_scenarios=5000,
    ...     random_state=123,
    ...     corr_views={('0', '1'): 0.8}, # View on correlation between asset 0 and 1
    ... )
    >>> q_synth = fp_synth.get_posterior_probabilities()
    >>> mu_post_synth, cov_post_synth = fp_synth.get_posterior()
    >>> print("\nPosterior Mean (Synthesized):", mu_post_synth)
    >>> print("Posterior Covariance (Synthesized):\n", cov_post_synth)
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
                self.p0 = p_array.reshape(-1, 1)

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

            cov = cov + np.eye(N) * 1e-6

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
                    "`distribution_fn` must return shape "
                    f"({num_scenarios}, {N}), got {self.R.shape}."
                )

            S = num_scenarios
            self.p0 = np.full((S, 1), 1.0 / S)

        self.mu0 = (self.R.T @ self.p0).flatten()
        self.cov0 = np.cov(self.R.T, aweights=self.p0.flatten())
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
        r"""
        Converts a raw view value into a standardized (operator, target) tuple.

        This static method provides flexibility in how views are specified. A view
        can be a simple scalar (implying an equality constraint) or a tuple
        containing an operator string and a scalar target.

        Accepted syntaxes:
        -----------------
        * ``x`` (e.g., `0.03`)   → `('==', x)`: Implies an equality view.
        * ``('>=', x)`` (e.g., `('>=', 0.05)`) → `('>=', x)`: Implies a greater-than-or-equal-to inequality view.
        * Similar for `'<=', '>', '<'`.

        For **relative mean views** (e.g., `mean_views={('Asset A', 'Asset B'): ('>=', 0.0)}`),
        the target `x` is interpreted as the desired difference between the means
        (e.g., :math:`\mu_1 - \mu_2 \ge x`).

        Parameters
        ----------
        v : Any
            The raw view value, which can be a scalar or a tuple.

        Returns
        -------
        Tuple[str, float]
            A tuple containing the operator string ('==', '>=', '<=', '>', '<')
            and the numerical target value.
        """
        if (
            isinstance(v, (list, tuple))
            and len(v) == 2
            and v[0] in ("==", ">=", "<=", ">", "<")
        ):
            return v[0], float(v[1])
        return "==", float(v)

    def _asset_idx(self, key) -> int:
        """
        Returns the integer index (position) of an asset given its label or numeric string.

        This internal helper method handles the mapping from user-friendly asset
        labels (strings or integers from pandas) to the zero-based integer indices
        used in NumPy arrays.

        Parameters
        ----------
        key : Any
            The asset label. This can be the exact label used during initialization
            (e.g., column name for a DataFrame, or an integer if `range` was used),
            or a numeric string that can be cast to an integer (e.g., "0", "1").

        Returns
        -------
        int
            The zero-based integer index of the asset within the scenario matrix.

        Raises
        ------
        ValueError
            If the asset label `key` is not recognized or found in the list of assets.
        """
        if key in self.assets:
            return self.assets.index(key)
        if isinstance(key, str) and key.isdigit():
            k_int = int(key)
            if k_int in self.assets:
                return self.assets.index(k_int)
        raise ValueError(f"Asset label '{key}' not recognised.")

    def _build_constraints(
        self,
        view_dict: Dict,
        moment_type: str,
        mu: np.ndarray,
        var: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[float], List[np.ndarray], List[float]]:
        """
        Translates a dictionary of views for a specific moment type into lists of
        equality and inequality constraints suitable for the `entropy_pooling` function.

        This method is crucial for converting high-level user views (e.g., "mean of
        Asset A is 5%") into the low-level linear constraints on probabilities
        (:math:`Eq=b`, :math:`Gq \le h`) required by the entropy pooling solver.
        The conversion leverages properties of expected values and higher moments
        (e.g., :math:`\mathbb{E}[R^2] = \text{Var}[R] + (\mathbb{E}[R])^2`).

        Parameters
        ----------
        view_dict : Dict
            A dictionary containing views for a specific moment type (e.g.,
            `self.mean_views`, `self.vol_views`). Keys are asset labels (or tuples
            for combined assets), and values are the view specifications.
        moment_type : str
            A string indicating the type of moment view to process.
            Accepted values are "mean", "vol" (volatility), "skew" (skewness),
            and "corr" (correlation).
        mu : (N,) ndarray
            The current mean vector of the asset returns. In sequential EP, this is
            the posterior mean from previous steps; in simultaneous EP, it's the prior mean.
            Used to linearize constraints for higher-order moments.
        var : (N,) ndarray
            The current variance vector of the asset returns. In sequential EP, this is
            the posterior variance from previous steps; in simultaneous EP, it's the prior variance.
            Used to linearize constraints for higher-order moments.

        Returns
        -------
        Tuple[List[np.ndarray], List[float], List[np.ndarray], List[float]]
            A tuple containing four lists:
            * `A_eq`: List of NumPy arrays, where each array is a row for the
                equality constraint matrix :math:`E`.
            * `b_eq`: List of floats, where each float is a target value for
                the equality constraints :math:`b`.
            * `G_ineq`: List of NumPy arrays, where each array is a row for the
                inequality constraint matrix :math:`G`.
            * `h_ineq`: List of floats, where each float is a target value for
                the inequality constraints :math:`h`.

        Raises
        ------
        ValueError
            If an unknown `moment_type` is provided.

        Notes
        -----
        For higher-order moments (volatility, skewness, correlation), the views
        are inherently non-linear in probabilities. To convert them
        into linear constraints, this method fixes lower-order moments (e.g.,
        mean and variance for skewness views) to their *current* values (`mu`, `var`).
        In the original EP, these would be fixed to prior values. In
        Sequential EP, these values are updated iteratively.

        * **Mean views**: `E[R_i] = target` directly translates to `sum(p_j * R_j_i) = target`.
            Relative mean views `E[R_i - R_j] = target` translate to `sum(p_j * (R_j_i - R_j_j)) = target`.
        * **Volatility views**: `StdDev[R_i] = target` implies `Var[R_i] = target^2`.
            Since `Var[R_i] = E[R_i^2] - (E[R_i])^2`, the constraint becomes
            `E[R_i^2] = target^2 + (E[R_i])^2`. This is linear in probabilities if `E[R_i]`
            is fixed (using the `mu` parameter).
        * **Skewness views**: `Skew[R_i] = target` implies a constraint on `E[R_i^3]`.
            `E[R_i^3] = Skew[R_i] * StdDev[R_i]^3 + 3*E[R_i]*Var[R_i] + E[R_i]^3`.
            This is linear in probabilities if `E[R_i]` and `Var[R_i]` are fixed.
        * **Correlation views**: `Corr[R_i, R_j] = target` implies a constraint on `E[R_i R_j]`.
            `E[R_i R_j] = Corr[R_i, R_j] * StdDev[R_i] * StdDev[R_j] + E[R_i] * E[R_j]`.
            This is linear in probabilities if `E[R_i]`, `E[R_j]`, `StdDev[R_i]`, `StdDev[R_j]` are fixed.
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
            else:
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
        """
        The core Entropy Pooling (EP) logic. This method orchestrates the application
        of views, supporting both simultaneous ("one-shot") and sequential
        ("iterated") processing. It does not handle confidence blending, assuming
        full confidence in the provided views for this step.

        The general EP problem aims to find a new probability vector `q` that
        minimizes relative entropy to the prior `p0` subject to linear constraints
        derived from the views.

        Returns
        -------
        np.ndarray
            The (S x 1) posterior probability vector `q`.
        """
        R, p0 = self.R, self.p0
        mu_cur, var_cur = self.mu0.copy(), self.var0.copy()

        def do_ep(prior_probs, A_eq_list, b_eq_list, G_ineq_list, h_ineq_list):
            S = R.shape[0]
            A_eq_list.append(np.ones(S))
            b_eq_list.append(1.0)

            A = np.vstack(A_eq_list) if A_eq_list else np.zeros((0, S))
            b = np.array(b_eq_list, float).reshape(-1, 1) if b_eq_list else np.zeros((0, 1))

            if G_ineq_list:
                G = np.vstack(G_ineq_list)
                h = np.array(h_ineq_list, float).reshape(-1, 1)
            else:
                G, h = None, None

            return entropy_pooling(prior_probs, A, b, G, h)

        if not any((self.mean_views, self.vol_views, self.skew_views, self.corr_views)):
            return p0

        if self.sequential:
            q_last = p0
            view_blocks = [
                ("mean", self.mean_views),
                ("vol", self.vol_views),
                ("skew", self.skew_views),
                ("corr", self.corr_views),
            ]

            for mtype, vd in view_blocks:
                if vd:
                    Aeq, beq, G, h = self._build_constraints(vd, mtype, mu_cur, var_cur)
                    q_last = do_ep(q_last, Aeq, beq, G, h)

                    mu_cur = (R.T @ q_last).flatten()
                    var_cur = ((R - mu_cur) ** 2).T @ q_last
                    var_cur = var_cur.flatten()

            return q_last

        else:
            A_all, b_all, G_all, h_all = [], [], [], []
            view_blocks = [
                ("mean", self.mean_views),
                ("vol", self.vol_views),
                ("skew", self.skew_views),
                ("corr", self.corr_views),
            ]
            for mtype, vd in view_blocks:
                if vd:
                    Aeq, beq, G, h = self._build_constraints(vd, mtype, mu_cur, var_cur)
                    A_all.extend(Aeq)
                    b_all.extend(beq)
                    G_all.extend(G)
                    h_all.extend(h)

            return do_ep(p0, A_all, b_all, G_all, h_all)

    def get_posterior_probabilities(self) -> np.ndarray:
        """Return the (S × 1) posterior probability vector.

        This method provides access to the final probability distribution `q`
        computed by the entropy pooling process, which incorporates all specified
        views while remaining as close as possible to the original prior
        distribution.

        Returns
        -------
        np.ndarray
            The optimal posterior probability vector, `q`, in column-vector form.
        """
        return self.posterior_probabilities

    def get_posterior(
        self,
    ) -> Tuple[Union[np.ndarray, pd.Series], Union[np.ndarray, pd.DataFrame]]:
        """Return *(posterior mean, posterior covariance)*.

        This method provides the key outputs of the entropy pooling process:
        the mean vector and covariance matrix of asset returns under the new,
        posterior probability distribution. These moments reflect the impact
        of the incorporated views.

        Returns
        -------
        Tuple[Union[np.ndarray, pd.Series], Union[np.ndarray, pd.DataFrame]]
            A tuple containing:
            * The posterior mean vector (as `np.ndarray` or `pd.Series`).
            * The posterior covariance matrix (as `np.ndarray` or `pd.DataFrame`).
            The type of return object (NumPy or Pandas) depends on the input
            type provided during the initialization of the `FlexibleViewsProcessor`.
        """
        return self.posterior_returns, self.posterior_cov
    
class BlackLittermanProcessor:
    r"""
    Bayesian Black–Litterman (BL) updater for *equality* **mean views**.

    The processor combines a *prior* distribution of excess returns
    :math:`\mathcal N(\boldsymbol\pi,\;\boldsymbol\Sigma)` with
    user-supplied views

    .. math::

       \mathbf P\,\boldsymbol\mu \;=\; \mathbf Q\;+\;\boldsymbol\varepsilon,
       \qquad
       \boldsymbol\varepsilon \sim \mathcal N\!\bigl(\mathbf 0,\,
       \boldsymbol\Omega\bigr),

    where

    * ``P`` (``self._p``) is the :math:`K\times N` **pick matrix** selecting
      linear combinations of the *N* asset means that are subject to views,
    * ``Q`` (``self._q``) is the :math:`K\times1` **view target** vector,
    * :math:`\boldsymbol\Omega` encodes view confidence as in
      He & Litterman :cite:p:`he2002intuition`,
      or via the Idzorek diagonal construction
      :cite:p:`idzorek2005step`.

    With the **shrinkage** scalar :math:`\tau>0` the posterior moments follow
    immediately from Bayesian mixed estimation
    :cite:p:`black1992global`

    .. math::
       :label: bl_posterior

       \begin{aligned}
       \boldsymbol\mu^{\star}
       &= \boldsymbol\pi
          + \tau\boldsymbol\Sigma\,\mathbf P^\top
          \bigl(\mathbf P\,\tau\boldsymbol\Sigma\,\mathbf P^\top
                +\boldsymbol\Omega\bigr)^{-1}
          \bigl(\mathbf Q - \mathbf P\,\boldsymbol\pi\bigr),\\
       \boldsymbol\Sigma^{\star}
       &= \boldsymbol\Sigma + \tau\boldsymbol\Sigma
          - \tau\boldsymbol\Sigma\,\mathbf P^\top
          \bigl(\mathbf P\,\tau\boldsymbol\Sigma\,\mathbf P^\top
                +\boldsymbol\Omega\bigr)^{-1}
          \mathbf P\,\tau\boldsymbol\Sigma.
       \end{aligned}

    Only **equality** mean views (absolute or relative) are implemented; the
    entropy-pooling framework in :class:`~FlexibleViewsProcessor` should be
    used for inequalities or higher-order moment constraints.

    Prior specification
    -------------------
    Exactly *one* of the following mutually exclusive inputs must be supplied
    to initialise :math:`\boldsymbol\pi`:

    1. ``pi`` – direct numeric vector.
    2. ``market_weights`` – reverse-optimised
       :math:`\boldsymbol\pi=\delta\boldsymbol\Sigma\mathbf w`
       (CAPM equilibrium) with
       **risk-aversion** :math:`\delta>0` :cite:p:`black1992global`.
    3. ``prior_mean`` – treat the sample mean as :math:`\boldsymbol\pi`.

    Parameters
    ----------
    prior_cov : (N, N) array_like
        Prior covariance :math:`\boldsymbol\Sigma`.
    prior_mean : (N,) array_like, optional
        Prior mean vector (exclusive with ``pi`` / ``market_weights``).
    market_weights : (N,) array_like, optional
        Market-cap weights used for CAPM reverse optimisation.
    risk_aversion : float, default 1.0
        Risk-aversion coefficient :math:`\delta\;(>0)`.
    tau : float, default 0.05
        Shrinkage scalar :math:`\tau` controlling the weight on the prior
        covariance; typical values 0.01–0.10
        :cite:p:`he2002intuition`.
    idzorek_use_tau : bool, default True
        If *True* the Idzorek confidence rule scales by
        :math:`\tau\boldsymbol\Sigma` (original He–Litterman convention);
        if *False* it uses :math:`\boldsymbol\Sigma` instead.
    pi : (N,) array_like, optional
        Direct prior mean (exclusive with the other two options above).
    mean_views : mapping or array_like, optional
        Equality mean views:

        * ``{'Asset': 0.02}`` (absolute)
        * ``{('A','B'): 0.00}`` (relative :math:`\mu_A-\mu_B = 0`)
        * length-*N* array (per-asset absolute views).

    view_confidences : float | sequence | dict, optional
        Idzorek confidences :math:`c_k\in(0,1]` per view.
    omega : {"idzorek"} | array_like, optional
        View covariance :math:`\boldsymbol\Omega`.
        * ``"idzorek"`` – derive from confidences.
        * vector length *K* – treated as diagonal.
        * full :math:`K\times K` matrix – used verbatim.
    verbose : bool, default False
        Print intermediate diagnostics.

    Attributes
    ----------
    posterior_mean : ndarray or pandas.Series
        :math:`\boldsymbol\mu^{\star}` from Eq. :eq:`bl_posterior`.
    posterior_cov : ndarray or pandas.DataFrame
        :math:`\boldsymbol\Sigma^{\star}` from Eq. :eq:`bl_posterior`.

    Methods
    -------
    get_posterior() → (posterior_mean, posterior_cov)
        Return the posterior moments in the same *NumPy / Pandas* flavour as
        the inputs.

    Examples
    --------
    >>> bl = BlackLittermanProcessor(
    ...         prior_cov=cov,
    ...         market_weights=cap_weights,
    ...         risk_aversion=2.5,
    ...         mean_views={('EM', 'DM'): 0.03},
    ...         view_confidences={'EM,DM': 0.60},
    ...         omega='idzorek')
    >>> mu_bl, sigma_bl = bl.get_posterior()

    """
    def get_posterior(
        self,
    ) -> Tuple[Union[np.ndarray, "pd.Series"], Union[np.ndarray, "pd.DataFrame"]]:
        """:no-index:"""
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

        # ---------- Σ (prior covariance) --------------------------------
        self._is_pandas: bool = isinstance(prior_cov, pd.DataFrame)
        self._assets: List[Union[str, int]] = (
            list(prior_cov.index)
            if self._is_pandas
            else list(range(np.asarray(prior_cov).shape[0]))
        )
        self._sigma: np.ndarray = np.asarray(prior_cov, dtype=float)
        n_assets: int = self._sigma.shape[0]

        if self._sigma.shape != (n_assets, n_assets):
            raise ValueError("prior_cov must be square (N, N).")
        if not np.allclose(self._sigma, self._sigma.T, atol=1e-8):
            warnings.warn("prior_cov not symmetric; symmetrising.")
            self._sigma = 0.5 * (self._sigma + self._sigma.T)

        if risk_aversion <= 0.0:
            raise ValueError("risk_aversion must be positive.")
        self._tau: float = float(tau)

        if pi is not None:
            self._pi = np.asarray(pi, dtype=float).reshape(-1, 1)
            src = "user π"
        elif market_weights is not None:
            weights = np.asarray(market_weights, dtype=float).ravel()
            if weights.size != n_assets:
                raise ValueError("market_weights length mismatch.")
            weights /= weights.sum()
            self._pi = risk_aversion * self._sigma @ weights.reshape(-1, 1)
            src = "δ Σ w"
        elif prior_mean is not None:
            self._pi = np.asarray(prior_mean, dtype=float).reshape(-1, 1)
            src = "prior_mean"
        else:
            raise ValueError("Provide exactly one of pi, market_weights or prior_mean.")
        if verbose:
            print(f"[BL] π source: {src}.")

        def _vec_to_dict(vec_like):
            if vec_like is None:
                return {}
            if isinstance(vec_like, dict):
                return vec_like
            vec = np.asarray(vec_like, float).ravel()
            if vec.size != n_assets:
                raise ValueError(f"`mean_views` must have length {n_assets}.")
            return {self._assets[i]: vec[i] for i in range(n_assets)}

        mv_dict = _vec_to_dict(mean_views)
        self._p, self._q, view_keys = self._build_views(mv_dict)
        self._k: int = self._p.shape[0]
        if verbose:
            print(f"[BL] Built P {self._p.shape}, Q {self._q.shape}.")

        # ---------- confidences & Ω -------------------------------------
        self._conf: Optional[np.ndarray] = self._parse_conf(view_confidences, view_keys)
        self._idzorek_use_tau = bool(idzorek_use_tau)
        self._omega: np.ndarray = self._build_omega(omega, verbose)

        # ---------- posterior -------------------------------------------
        self._posterior_mean, self._posterior_cov = self._compute_posterior(verbose)
        if self._is_pandas:
            self._posterior_mean = pd.Series(self._posterior_mean, index=self._assets)
            self._posterior_cov = pd.DataFrame(
                self._posterior_cov, index=self._assets, columns=self._assets
            )

    # ------------------------------------------------------------------ #
    # internal utilities
    # ------------------------------------------------------------------ #
    # asset index lookup
    def _asset_index(self, label: Union[str, int]) -> int:
        if label in self._assets:
            return self._assets.index(label)
        if isinstance(label, str) and label.isdigit():
            idx = int(label)
            if idx < len(self._assets):
                return idx
        raise ValueError(f"Unknown asset label '{label}'.")

    # ---- views --------------------------------------------------------
    def _build_views(
        self, mean_views: Dict[Any, Any]
    ) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
        rows: List[np.ndarray] = []
        targets: List[float] = []
        keys: List[Any] = []
        n = len(self._assets)

        for key, value in mean_views.items():
            # Accept either scalar or single-element tuple/list
            if isinstance(value, Sequence) and not isinstance(value, str):
                if len(value) != 1:
                    raise ValueError(
                        "Inequality views not supported – use scalar value."
                    )
                target = float(value[0])
            else:
                target = float(value)

            if isinstance(key, tuple):  # relative view μ_i − μ_j = target
                asset_i, asset_j = key
                i_idx, j_idx = self._asset_index(asset_i), self._asset_index(asset_j)
                row = np.zeros(n)
                row[i_idx], row[j_idx] = 1.0, -1.0
            else:  # absolute view μ_i = target
                idx = self._asset_index(key)
                row = np.zeros(n)
                row[idx] = 1.0

            rows.append(row)
            targets.append(target)
            keys.append(key)

        p_mat = np.vstack(rows) if rows else np.zeros((0, n))
        q_vec = (
            np.array(targets, dtype=float).reshape(-1, 1)
            if targets
            else np.zeros((0, 1))
        )
        return p_mat, q_vec, keys

    # ---- confidences --------------------------------------------------
    @staticmethod
    def _parse_conf(conf: Any, keys: List[Any]) -> Optional[np.ndarray]:
        if conf is None:
            return None
        if isinstance(conf, (int, float)):
            return np.full(len(keys), float(conf))
        if isinstance(conf, dict):
            return np.array([float(conf.get(k, 1.0)) for k in keys])
        arr = np.asarray(conf, dtype=float).ravel()
        if arr.size != len(keys):
            raise ValueError("view_confidences length mismatch.")
        return arr

    # ---- Ω construction ----------------------------------------------
    def _build_omega(self, omega: Any, verbose: bool) -> np.ndarray:
        if self._k == 0:  # no views → empty Ω
            return np.zeros((0, 0))

        tau_sigma = self._tau * self._sigma

        # -- Idzorek -----------------------------------------------------
        if isinstance(omega, str) and omega.lower() == "idzorek":
            if self._conf is None:
                raise ValueError("Idzorek requires view_confidences.")
            diag = []
            base_sigma = tau_sigma if self._idzorek_use_tau else self._sigma
            for i, conf in enumerate(self._conf):
                p_i = self._p[i : i + 1]  # (1, N)
                var_i = (p_i @ base_sigma @ p_i.T).item()  # σ²(view)
                c = np.clip(conf, 1e-6, 1.0 - 1e-6)
                factor = (1.0 - c) / c
                diag.append(factor * var_i)
            omega_mat = np.diag(diag)
            if verbose:
                suffix = "τ Σ" if self._idzorek_use_tau else "Σ"
                print(f"[BL] Ω from Idzorek confidences (base = {suffix}).")

        # -- default diagonal -------------------------------------------
        elif omega is None:
            omega_mat = np.diag(np.diag(self._p @ tau_sigma @ self._p.T))
            if verbose:
                print("[BL] Ω = τ·diag(P Σ Pᵀ).")

        # -- user-supplied ----------------------------------------------
        else:
            omega_arr = np.asarray(omega, dtype=float)
            if omega_arr.ndim == 1 and omega_arr.size == self._k:
                omega_mat = np.diag(omega_arr)
            elif omega_arr.shape == (self._k, self._k):
                omega_mat = omega_arr
            else:
                raise ValueError(
                    "omega must be 'idzorek', length-K vector, or K×K matrix."
                )
            if verbose:
                print("[BL] Using user-provided Ω.")

        return omega_mat

    # ---- posterior ----------------------------------------------------
    def _compute_posterior(self, verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        tau_sigma = self._tau * self._sigma

        if self._k == 0:  # no views: posterior = prior
            if verbose:
                print("[BL] No views → posterior = prior.")
            return self._pi.flatten(), self._sigma

        # P τ Σ Pᵀ + Ω
        mat_a = self._p @ tau_sigma @ self._p.T + self._omega  # (K, K)

        # Solve rather than invert for numerical stability
        rhs = self._q - self._p @ self._pi  # (K, 1)
        mean_shift = np.linalg.solve(mat_a, rhs)  # (K, 1)

        posterior_mean = (self._pi + tau_sigma @ self._p.T @ mean_shift).flatten()

        middle = tau_sigma @ self._p.T @ np.linalg.solve(mat_a, self._p @ tau_sigma)
        posterior_cov = self._sigma + tau_sigma - middle
        posterior_cov = 0.5 * (posterior_cov + posterior_cov.T)  # enforce symmetry

        if verbose:
            print("[BL] Posterior mean and covariance computed.")
        return posterior_mean, posterior_cov
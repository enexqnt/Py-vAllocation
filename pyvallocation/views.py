# entropy_pooling and _dual_objective functions are adapted from fortitudo-tech https://github.com/fortitudo-tech/fortitudo.tech

import logging
import warnings
from collections.abc import Sequence
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import Bounds, minimize

import pandas as pd

from .bayesian import _cholesky_pd
from .probabilities import normalize_probability_vector

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

    Args:
        lagrange_multipliers (np.ndarray): Current vector of Lagrange multipliers
            :math:`\\lambda` at which the objective and gradient are evaluated.
        log_p_col (np.ndarray): Log prior probabilities :math:`\\log p^{(0)}` as a
            column vector with shape ``(S, 1)``.
        lhs (np.ndarray): Left-hand side matrix :math:`A` with shape ``(K, S)``
            combining equality and inequality constraints.
        rhs_squeezed (np.ndarray): Right-hand side targets :math:`b` with shape ``(K,)``.

    Returns:
        Tuple[float, np.ndarray]: Scaled dual objective value and gradient.
        The tuple entries correspond to ``value`` and ``gradient`` where
        ``value = 1e3 * varphi(lambda)`` and
        ``gradient = 1e3 * nabla varphi(lambda)``.

    Notes:
        Intended for internal use by ``scipy.optimize.minimize`` inside
        :func:`entropy_pooling`. Based on the dual formulation of entropy
        minimization.
    """
    lagrange_multipliers_col = lagrange_multipliers[:, np.newaxis]

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        exponent = log_p_col - 1.0 - lhs.T @ lagrange_multipliers_col
    exponent = np.clip(exponent, -700.0, 700.0)
    x = np.exp(exponent)

    rhs_vec = np.atleast_1d(rhs_squeezed)
    objective_value = -(-np.sum(x) - lagrange_multipliers @ rhs_vec)
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
    r"""Return posterior probabilities via the entropy-pooling algorithm.

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
    (Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm with box
    constraints) are supported, as they allow for box bounds on the Lagrange
    multipliers.

    Args:
        p (np.ndarray): Prior probability vector :math:`p^{(0)} \\in (0,1)^S`,
            shape ``(S,)`` or ``(S, 1)``. Must sum to 1 with strictly positive entries.
        A (np.ndarray): Equality constraint matrix :math:`E` in :math:`Eq=b`,
            shape ``(K_eq, S)``.
        b (np.ndarray): Equality targets :math:`b`, shape ``(K_eq,)``.
        G (np.ndarray, optional): Inequality constraint matrix :math:`G` in
            :math:`Gq \\le h`, shape ``(K_ineq, S)``. Defaults to ``None``.
        h (np.ndarray, optional): Inequality targets :math:`h`,
            shape ``(K_ineq,)``. Defaults to ``None``.
        method (str, optional): Optimizer passed to ``scipy.optimize.minimize``.
            Supported values: ``"TNC"`` or ``"L-BFGS-B"``. Defaults to ``"TNC"``.

    Returns:
        np.ndarray: Posterior probability column vector :math:`q` with shape
        ``(S, 1)`` satisfying the constraints while minimizing relative entropy.

    Raises:
        ValueError: If an unsupported ``method`` is specified.

    Notes:
        Adapted from `fortitudo.tech` (https://github.com/fortitudo-tech/fortitudo.tech).
        The core methodology is described in Meucci (2008). The dual problem is
        minimized and the posterior probabilities are recovered from the optimal
        Lagrange multipliers.
    """
    opt_method = method or "TNC"
    if opt_method not in ("TNC", "L-BFGS-B"):
        raise ValueError(
            f"Method {opt_method} not supported. Choose 'TNC' or 'L-BFGS-B'."
        )

    normalised_prior = normalize_probability_vector(
        p,
        name="prior probabilities",
        strictly_positive=True,
    )
    p_col = normalised_prior.reshape(-1, 1)
    b_col = np.asarray(b, dtype=float).reshape(-1, 1)

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

    solver_options = {"maxfun": 10000}
    if opt_method == "L-BFGS-B":
        solver_options["maxiter"] = 1000

    solution = minimize(
        _entropy_pooling_dual_objective,
        x0=initial_lagrange_multipliers,
        args=(log_p_col, current_lhs, current_rhs_stacked.squeeze()),
        method=opt_method,
        jac=True,
        bounds=optimizer_bounds,
        options=solver_options,
    )

    if not solution.success:
        status = getattr(solution, "status", None)
        message = getattr(solution, "message", "")
        raise RuntimeError(
            "Entropy pooling optimisation failed "
            f"(status={status}): {message}"
        )

    optimal_lagrange_multipliers_col = solution.x[:, np.newaxis]

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        posterior_exponent = log_p_col - 1.0 - current_lhs.T @ optimal_lagrange_multipliers_col
    posterior_exponent = np.clip(posterior_exponent, -700.0, 700.0)
    q_posterior = np.exp(posterior_exponent)

    if not np.all(np.isfinite(q_posterior)):
        raise RuntimeError("Entropy pooling produced non-finite posterior probabilities.")

    q_posterior = np.clip(q_posterior, 0.0, None)
    total_prob = float(np.sum(q_posterior))
    if not np.isfinite(total_prob) or total_prob <= 0.0:
        raise RuntimeError("Entropy pooling posterior probabilities could not be normalised.")
    q_posterior /= total_prob

    return q_posterior

class FlexibleViewsProcessor:
    r"""Entropy-pooling engine with fully flexible moment views.

    The `FlexibleViewsProcessor` class provides a robust framework for adjusting a
    discrete **prior** distribution of multivariate asset returns to incorporate
    user-specified *views*. This is achieved by minimizing the relative entropy
    (Kullback-Leibler divergence) between the prior and the new, "posterior"
    distribution. The class implements the *Fully Flexible Views*
    methodology proposed by Attilio Meucci, supporting views on
    various statistical moments, including **means**, **variances**, **skewnesses**,
    and **correlations**. It supports both *simultaneous* ("one-shot") and
    *iterated* (block-wise) entropy pooling.

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
    * **Plug-and-play Scenario Handling:** Users can either provide historical
        returns directly or specify prior mean and covariance, allowing the
        processor to synthesize scenarios. The output includes posterior moments
        (mean, covariance) and the adjusted probabilities.
    * **Sequential Updating (Iterated EP):** By setting `sequential=True`, the
        class applies view blocks (e.g., mean views, then volatility views) in a
        predefined order (*mean -> vol -> skew -> corr*). This iterated approach,
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

    Args:
        prior_risk_drivers (np.ndarray or pd.DataFrame, optional): Prior
            *risk-driver* scenarios (returns, factors, spreads, vol surfaces, etc.).
            If supplied, ``prior_mean`` and ``prior_cov`` are ignored.
        prior_probabilities (array-like, optional): Prior scenario weights. If omitted,
            uniform probabilities ``1/S`` are used.
        prior_mean (array-like, optional): Mean vector used to synthesize scenarios
            when ``prior_risk_drivers`` is not supplied.
        prior_cov (array-like, optional): Covariance matrix used to synthesize scenarios
            when ``prior_risk_drivers`` is not supplied.
        distribution_fn (callable, optional): Custom scenario generator with signature
            ``f(mu, cov, n, rng) -> np.ndarray`` returning shape ``(n, N)``. Defaults to
            ``numpy.random.Generator.multivariate_normal``.
        num_scenarios (int, optional): Number of synthetic scenarios when
            ``prior_risk_drivers`` is missing. Defaults to ``10000``.
        random_state (int or np.random.Generator, optional): Seed or RNG used by
            ``distribution_fn`` / ``multivariate_normal``.
        mean_views, vol_views, corr_views, skew_views (mapping or array-like, optional):
            View payloads. Keys are asset labels (or pairs for correlations).
            Values are scalars (equalities) or tuples like ``('>=', x)``.
        sequential (bool, optional): If ``True``, apply views sequentially (mean → vol → skew → corr).
            Defaults to ``False`` (single entropy pooling solve).

    Attributes:
        posterior_probabilities (np.ndarray): Optimal posterior probabilities :math:`q`, shape ``(S, 1)``.
        posterior_returns (np.ndarray or pd.Series): Posterior mean.
        posterior_cov (np.ndarray or pd.DataFrame): Posterior covariance.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from scipy.stats import multivariate_normal
        >>>
        >>> # Example with historical returns
        >>> np.random.seed(42)
        >>> returns_data = np.random.randn(100, 3)  # S=100 scenarios, N=3 assets
        >>> return_df = pd.DataFrame(
        ...     returns_data, columns=["Asset A", "Asset B", "Asset C"]
        ... )
        >>>
        >>> # Initialize with historical returns and views
        >>> fp_hist = FlexibleViewsProcessor(
        ...     prior_risk_drivers=return_df,
        ...     mean_views={"Asset A": 0.05, "Asset B": (">=", 0.01)},
        ...     vol_views={"Asset A": ("<=", 0.15)},
        ...     sequential=True,  # Apply views sequentially
        ... )
        >>> q_hist = fp_hist.get_posterior_probabilities()
        >>> mu_post_hist, cov_post_hist = fp_hist.get_posterior()
        >>> print("Posterior Mean (Historical):", mu_post_hist)
        >>> print("Posterior Covariance (Historical):\\n", cov_post_hist)
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
        ...     corr_views={("0", "1"): 0.8},  # Correlation view between asset 0 and 1
        ... )
        >>> q_synth = fp_synth.get_posterior_probabilities()
        >>> mu_post_synth, cov_post_synth = fp_synth.get_posterior()
        >>> print("\\nPosterior Mean (Synthesized):", mu_post_synth)
        >>> print("Posterior Covariance (Synthesized):\\n", cov_post_synth)
    """
    def __init__(
        self,
        prior_risk_drivers: Optional[Union[np.ndarray, "pd.DataFrame"]] = None,
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
        cvar_views: Any = None,
        sequential: bool = False,
    ):
        """Initialise the flexible-views processor.

        Args:
            prior_risk_drivers: Scenario matrix of risk drivers (returns, factors, etc.).
            prior_probabilities: Optional scenario probabilities aligned to scenarios.
            prior_mean: Prior mean vector used when scenarios are not provided.
            prior_cov: Prior covariance matrix used when scenarios are not provided.
            distribution_fn: Optional callable to draw scenarios from moments.
            num_scenarios: Number of scenarios to simulate when needed (default ``10000``).
            random_state: Optional random seed or Generator.
            mean_views: Views on means (absolute or relative).
            vol_views: Views on marginal volatilities.
            corr_views: Views on correlations.
            skew_views: Views on skewness.
            cvar_views: Views on conditional value-at-risk. Specify as
                ``{asset: (target_cvar, gamma)}`` where ``gamma`` is the tail
                probability (e.g. 0.05 for 95% CVaR). Uses recursive entropy
                pooling per Meucci (2011, ssrn-1542083).
            sequential: Whether to apply views sequentially (default ``False``).
        """
        risk_drivers = prior_risk_drivers

        if risk_drivers is not None:
            if isinstance(risk_drivers, pd.DataFrame):
                self.R = risk_drivers.values
                self.assets = list(risk_drivers.columns)
                self._use_pandas = True
            else:
                self.R = np.atleast_2d(np.asarray(risk_drivers, float))
                self.assets = [str(i) for i in range(self.R.shape[1])]
                self._use_pandas = False

            S, N = self.R.shape

            if prior_probabilities is None:
                self.p0 = np.full((S, 1), 1.0 / S)
            else:
                p_array = normalize_probability_vector(
                    prior_probabilities,
                    name="prior_probabilities",
                    strictly_positive=True,
                )
                if p_array.size != S:
                    raise ValueError(
                        "`prior_probabilities` must match the number of scenarios."
                    )
                self.p0 = p_array.reshape(-1, 1)

        else:
            if prior_mean is None or prior_cov is None:
                raise ValueError(
                    "Provide either `prior_risk_drivers` or both `prior_mean` and `prior_cov`."
                )

            if not isinstance(num_scenarios, int) or num_scenarios <= 0:
                raise ValueError("`num_scenarios` must be a positive integer.")

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

            if cov.shape != (N, N):
                raise ValueError(
                    f"`prior_cov` must be a square matrix of shape ({N}, {N})."
                )
            cov = 0.5 * (cov + cov.T)
            cov = cov + np.eye(N) * 1e-6
            chol = _cholesky_pd(cov)

            rng = np.random.default_rng(random_state)

            if distribution_fn is None:
                standard_normals = rng.standard_normal((num_scenarios, N))
                with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                    simulated = standard_normals @ chol.T
                self.R = simulated + mu
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
        self.cov0 = np.cov(self.R.T, aweights=self.p0.flatten(), bias=True)
        self.var0 = np.diag(self.cov0)

        def _vec_to_dict(vec_like, name):
            """Normalize vector-like views into an asset-keyed dictionary.

            Args:
                vec_like: Vector-like or mapping of view targets.
                name: Parameter name for error messages.

            Returns:
                dict: Asset-keyed views.
            """
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
        if cvar_views is not None:
            self.cvar_views = _vec_to_dict(cvar_views, "cvar_views") if not isinstance(cvar_views, dict) else cvar_views
        else:
            self.cvar_views = None
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
        * ``x`` (e.g., `0.03`)   -> `('==', x)`: Implies an equality view.
        * ``('>=', x)`` (e.g., `('>=', 0.05)`) -> `('>=', x)`: Implies a greater-than-or-equal-to inequality view.
        * Similar for `'<=', '>', '<'`.

        For **relative mean views** (e.g., `mean_views={('Asset A', 'Asset B'): ('>=', 0.0)}`),
        the target `x` is interpreted as the desired difference between the means
        (e.g., :math:`\mu_1 - \mu_2 \ge x`).

        Args:
            v (Any): Raw view value (scalar or tuple).

        Returns:
            Tuple[str, float]: Operator string (``'=='``, ``'>='``, ``'<='``, ``'>'``, ``'<'``)
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

        Args:
            key (Any): Asset label. Can be the original label used at initialization
                (e.g., DataFrame column name) or a numeric string (e.g., ``"0"``).

        Returns:
            int: Zero-based integer index of the asset within the scenario matrix.

        Raises:
            ValueError: If the asset label is not recognized.
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
        r"""
        Translates a dictionary of views for a specific moment type into lists of
        equality and inequality constraints suitable for the `entropy_pooling` function.

        This method is crucial for converting high-level user views (e.g., "mean of
        Asset A is 5%") into the low-level linear constraints on probabilities
        (:math:`Eq=b`, :math:`Gq \le h`) required by the entropy pooling solver.
        The conversion leverages properties of expected values and higher moments
        (e.g., :math:`\mathbb{E}[R^2] = \text{Var}[R] + (\mathbb{E}[R])^2`).

        Args:
            view_dict (Dict): Views for the specified moment type (e.g.,
                ``self.mean_views`` or ``self.vol_views``). Keys are asset labels
                (or tuples for combined assets), values are view specifications.
            moment_type (str): Moment type to process. Supported values:
                ``"mean"``, ``"vol"``, ``"skew"``, ``"corr"``.
            mu (np.ndarray): Current mean vector, shape ``(N,)``. In sequential EP,
                this is the posterior mean from prior steps; in simultaneous EP,
                it is the prior mean. Used to linearize higher-order constraints.
            var (np.ndarray): Current variance vector, shape ``(N,)``. In sequential EP,
                this is the posterior variance from prior steps; in simultaneous EP,
                it is the prior variance. Used to linearize higher-order constraints.

        Returns:
            Tuple[List[np.ndarray], List[float], List[np.ndarray], List[float]]:
            ``(A_eq, b_eq, A_ineq, b_ineq)`` constraint lists for entropy pooling,
            where ``A_eq``/``A_ineq`` are row arrays and ``b_eq``/``b_ineq`` are targets.
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
            """Append a linear constraint row based on the operator.

            Args:
                op: Operator string (``==``, ``<=``, ``>=``).
                row: Constraint row vector.
                raw: Right-hand-side scalar.
            """
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
                if tgt <= 0:
                    raise ValueError(f"Volatility target for '{asset}' must be positive (got {tgt}).")
                idx = self._asset_idx(asset)
                raw = tgt**2 + mu[idx] ** 2
                add(op, R[:, idx] ** 2, raw)

        elif moment_type == "skew":
            for asset, vw in view_dict.items():
                op, tgt = self._parse_view(vw)
                idx = self._asset_idx(asset)
                if var[idx] < 1e-12:
                    raise ValueError(f"Variance of '{asset}' is near-zero ({var[idx]:.2e}); skewness view is numerically unstable.")
                s = np.sqrt(var[idx])
                raw = tgt * s**3 + 3 * mu[idx] * var[idx] + mu[idx] ** 3
                add(op, R[:, idx] ** 3, raw)

        elif moment_type == "corr":
            for (a1, a2), vw in view_dict.items():
                op, tgt = self._parse_view(vw)
                if not -1.0 <= tgt <= 1.0:
                    raise ValueError(f"Correlation target for ('{a1}', '{a2}') must be in [-1, 1] (got {tgt}).")
                i = self._asset_idx(a1)
                j = self._asset_idx(a2)
                s_i, s_j = np.sqrt(var[i]), np.sqrt(var[j])
                raw = tgt * s_i * s_j + mu[i] * mu[j]
                add(op, R[:, i] * R[:, j], raw)

        else:
            raise ValueError(f"Unknown moment type '{moment_type}'.")

        return A_eq, b_eq, G_ineq, h_ineq

    def _solve_cvar_view(
        self,
        asset_idx: int,
        gamma: float,
        cvar_target: float,
        prior_probs: np.ndarray,
        other_A: List[np.ndarray] = None,
        other_b: List[float] = None,
        other_G: List[np.ndarray] = None,
        other_h: List[float] = None,
        max_newton_iter: int = 20,
    ) -> np.ndarray:
        """Solve CVaR view via Meucci (2011) recursive EP (Eqs 9-15).

        For a view E[X | X <= VaR_gamma] = cvar_target, we search over
        possible VaR thresholds (index s) using Newton-Raphson on the
        relative entropy profile.

        Args:
            asset_idx: Index of the asset for the CVaR view.
            gamma: Tail probability (e.g. 0.05 for 95% CVaR).
            cvar_target: Target CVaR value (negative = loss).
            prior_probs: Current prior probability vector (S, 1).
            other_A, other_b: Additional equality constraints.
            other_G, other_h: Additional inequality constraints.
            max_newton_iter: Maximum Newton-Raphson iterations.

        Returns:
            np.ndarray: Posterior probability vector (S, 1).
        """
        R = self.R
        x = R[:, asset_idx]  # marginal scenarios for this asset
        S = len(x)
        p = prior_probs.flatten()

        # Sort scenarios ascending
        order = np.argsort(x)
        x_sorted = x[order]
        p_sorted = p[order]

        # Initialize s_bar from prior: find s where cumulative prob first reaches gamma
        cum_p = np.cumsum(p_sorted)
        s_bar = int(np.searchsorted(cum_p, gamma, side='left'))
        s_bar = max(1, min(s_bar, S - 1))

        def _solve_for_s(s):
            """Solve EP with VaR at index s (Eq 9-10)."""
            # Constraints: sum(q_1..q_s * x_1..x_s) = gamma * cvar_target
            #              sum(q_1..q_s) = gamma
            A_cvar = []
            b_cvar = []

            # Constraint 1: sum of tail probs = gamma
            row_sum = np.zeros(S)
            row_sum[:s + 1] = 1.0
            A_cvar.append(row_sum[np.argsort(order)])  # unsort back to original order
            b_cvar.append(gamma)

            # Constraint 2: weighted tail mean = gamma * cvar_target
            row_mean = np.zeros(S)
            row_mean[:s + 1] = x_sorted[:s + 1]
            A_cvar.append(row_mean[np.argsort(order)])  # unsort
            b_cvar.append(gamma * cvar_target)

            # Normalization: sum(q) = 1
            A_cvar.append(np.ones(S))
            b_cvar.append(1.0)

            # Combine with other constraints
            all_A = list(other_A or []) + A_cvar
            all_b = list(other_b or []) + b_cvar
            all_G = list(other_G or [])
            all_h = list(other_h or [])

            try:
                q = entropy_pooling(
                    prior_probs.flatten(),
                    A=np.vstack(all_A) if all_A else None,
                    b=np.array(all_b) if all_b else None,
                    G=np.vstack(all_G) if all_G else None,
                    h=np.array(all_h) if all_h else None,
                )
                # Compute relative entropy
                q_flat = q.flatten()
                mask = q_flat > 0
                kl = float(np.sum(q_flat[mask] * np.log(q_flat[mask] / p[mask])))
                return q, kl
            except Exception:
                return None, np.inf

        # Newton-Raphson search (Eqs 12-15)
        s = s_bar
        best_q, best_kl = _solve_for_s(s)

        for _ in range(max_newton_iter):
            # Compute D(s) = E(s+1) - E(s) and D2(s) = D(s+1) - D(s)
            s_lo = max(1, s - 1)
            s_hi = min(S - 2, s + 1)

            _, kl_lo = _solve_for_s(s_lo)
            _, kl_mid = _solve_for_s(s)
            _, kl_hi = _solve_for_s(s_hi)

            d1 = kl_hi - kl_mid  # first difference
            d2 = (kl_hi - 2 * kl_mid + kl_lo)  # second difference

            if abs(d2) < 1e-14:
                break

            ratio = d1 / d2
            if not np.isfinite(ratio):
                break

            s_new = int(round(s - ratio))
            s_new = max(1, min(s_new, S - 2))

            if s_new == s:
                break

            s = s_new
            q_new, kl_new = _solve_for_s(s)
            if q_new is not None and kl_new < best_kl:
                best_q, best_kl = q_new, kl_new

        if best_q is None:
            raise RuntimeError(f"CVaR view solve failed for asset {asset_idx}.")

        return best_q.reshape(-1, 1)

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
            """Solve entropy pooling for a given constraint set.

            Args:
                prior_probs: Prior probability vector.
                A_eq_list: Equality constraint rows.
                b_eq_list: Equality constraint values.
                G_ineq_list: Inequality constraint rows.
                h_ineq_list: Inequality constraint values.

            Returns:
                np.ndarray: Posterior probability vector.
            """
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

        if not any((self.mean_views, self.vol_views, self.skew_views, self.corr_views, self.cvar_views)):
            return p0

        if self.sequential:
            q_result = p0
            view_blocks = [
                ("mean", self.mean_views),
                ("vol", self.vol_views),
                ("skew", self.skew_views),
                ("corr", self.corr_views),
            ]

            for mtype, vd in view_blocks:
                if vd:
                    Aeq, beq, G, h = self._build_constraints(vd, mtype, mu_cur, var_cur)
                    q_result = do_ep(q_result, Aeq, beq, G, h)

                    mu_cur = (R.T @ q_result).flatten()
                    var_cur = ((R - mu_cur) ** 2).T @ q_result
                    var_cur = var_cur.flatten()

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

            if A_all or G_all:
                q_result = do_ep(p0, A_all, b_all, G_all, h_all)
            else:
                q_result = p0

        # --- CVaR views (recursive EP, Meucci 2011 ssrn-1542083) ---
        if self.cvar_views:
            for asset_key, vw in self.cvar_views.items():
                idx = self._asset_idx(asset_key)
                if isinstance(vw, (list, tuple)) and len(vw) >= 2:
                    cvar_target = float(vw[0])
                    gamma = float(vw[1])
                else:
                    cvar_target = float(vw)
                    gamma = 0.05
                q_result = self._solve_cvar_view(
                    asset_idx=idx, gamma=gamma, cvar_target=cvar_target,
                    prior_probs=q_result,
                )

        return q_result

    def get_posterior_probabilities(self) -> np.ndarray:
        """Return the (S x 1) posterior probability vector.

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
    Bayesian Black-Litterman (BL) updater for *equality* **mean views**.

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

    1. ``pi`` - direct numeric vector.
    2. ``market_weights`` - reverse-optimised
       :math:`\boldsymbol\pi=\delta\boldsymbol\Sigma\mathbf w`
       (CAPM equilibrium) with
       **risk-aversion** :math:`\delta>0` :cite:p:`black1992global`.
    3. ``prior_mean`` - treat the sample mean as :math:`\boldsymbol\pi`.

    Args:
        prior_cov (array-like): Prior covariance :math:`\\boldsymbol\\Sigma`, shape ``(N, N)``.
        prior_mean (array-like, optional): Prior mean vector, exclusive with ``pi`` / ``market_weights``.
        market_weights (array-like, optional): Market-cap weights used for CAPM reverse optimisation.
        risk_aversion (float, optional): Risk-aversion coefficient :math:`\\delta (>0)`. Defaults to ``1.0``.
        tau (float, optional): Shrinkage scalar :math:`\\tau` for the prior covariance.
            Typical values ``0.01``–``0.10``. Defaults to ``0.05``. :cite:p:`he2002intuition`.
        idzorek_use_tau (bool, optional): If ``True``, the Idzorek rule scales by
            :math:`\\tau\\boldsymbol\\Sigma`; otherwise it uses :math:`\\boldsymbol\\Sigma`.
            Defaults to ``True``.
        pi (array-like, optional): Direct prior mean, exclusive with the other options above.
        mean_views (mapping or array-like, optional): Equality mean views.
            Examples: ``{'Asset': 0.02}`` (absolute), ``{('A','B'): 0.00}`` (relative),
            or length-``N`` array (per-asset absolute views).
        view_confidences (float | sequence | dict, optional): Idzorek confidences
            :math:`c_k \\in (0,1]` per view.
        omega ({"idzorek"} | array-like, optional): View covariance :math:`\\boldsymbol\\Omega`.
            ``"idzorek"`` derives it from confidences; vector length ``K`` is diagonal;
            full ``K x K`` matrices are used verbatim.
        verbose (bool, optional): If ``True``, prints intermediate diagnostics. Defaults to ``False``.

    Attributes:
        posterior_mean (np.ndarray or pd.Series): :math:`\\boldsymbol\\mu^{\\star}` from Eq. :eq:`bl_posterior`.
        posterior_cov (np.ndarray or pd.DataFrame): :math:`\\boldsymbol\\Sigma^{\\star}` from Eq. :eq:`bl_posterior`.

    Methods:
        get_posterior() -> (posterior_mean, posterior_cov): Return posterior moments in the same
        NumPy/Pandas flavour as the inputs.

    Examples:
        >>> bl = BlackLittermanProcessor(
        ...     prior_cov=cov,
        ...     market_weights=cap_weights,
        ...     risk_aversion=2.5,
        ...     mean_views={("EM", "DM"): 0.03},
        ...     view_confidences={"EM,DM": 0.60},
        ...     omega="idzorek",
        ... )
        >>> mu_bl, sigma_bl = bl.get_posterior()

    """
    def get_posterior(
        self,
    ) -> Tuple[Union[np.ndarray, "pd.Series"], Union[np.ndarray, "pd.DataFrame"]]:
        """:no-index:

        Return posterior mean and covariance.

        Returns:
            Tuple[ArrayLike, ArrayLike]: Posterior mean and covariance.
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
        """Initialise the Black-Litterman processor.

        Args:
            prior_cov: Prior covariance matrix.
            prior_mean: Optional prior mean vector.
            market_weights: Optional market-cap weights for implied equilibrium mean.
            risk_aversion: Risk-aversion coefficient (default ``1.0``).
            tau: Prior covariance scaling (default ``0.05``).
            idzorek_use_tau: Whether to scale Idzorek Omega by ``tau``.
            pi: Optional direct prior mean.
            mean_views: Mean views (absolute or relative).
            view_confidences: View confidence levels in ``(0, 1]``.
            omega: View covariance (``"idzorek"`` or array-like).
            verbose: Whether to print diagnostics.
        """

        # ---------- \Sigma (prior covariance) --------------------------------
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
            src = "user \\pi"
        elif market_weights is not None:
            weights = np.asarray(market_weights, dtype=float).ravel()
            if weights.size != n_assets:
                raise ValueError("market_weights length mismatch.")
            weights /= weights.sum()
            self._pi = risk_aversion * self._sigma @ weights.reshape(-1, 1)
            src = "\\delta \\Sigma w"
        elif prior_mean is not None:
            self._pi = np.asarray(prior_mean, dtype=float).reshape(-1, 1)
            src = "prior_mean"
        else:
            raise ValueError("Provide exactly one of pi, market_weights or prior_mean.")
        if verbose:
            print(f"[BL] \\pi source: {src}.")

        def _vec_to_dict(vec_like):
            """Normalize mean views into an asset-keyed dictionary.

            Args:
                vec_like: Vector-like or mapping of views.

            Returns:
                dict: Asset-keyed views.
            """
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

        # ---------- confidences & \Omega -------------------------------------
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
        """Resolve an asset label or numeric index to a position.

        Args:
            label: Asset label or integer index.

        Returns:
            int: Asset index.
        """
        if label in self._assets:
            return self._assets.index(label)
        if isinstance(label, (int, np.integer)):
            if 0 <= label < len(self._assets):
                return int(label)
        elif isinstance(label, str) and label.isdigit():
            k_int = int(label)
            if k_int in self._assets:
                return self._assets.index(k_int)
            if 0 <= k_int < len(self._assets):
                return k_int
        raise ValueError(f"Unknown asset label '{label}'.")

    # ---- views --------------------------------------------------------
    def _build_views(
        self, mean_views: Dict[Any, Any]
    ) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
        """Build the P and Q matrices from mean views.

        Args:
            mean_views: Mapping of view keys to targets.

        Returns:
            Tuple[np.ndarray, np.ndarray, list]: P matrix, Q vector, and view keys.
        """
        rows: List[np.ndarray] = []
        targets: List[float] = []
        keys: List[Any] = []
        n = len(self._assets)

        for key, value in mean_views.items():
            # Accept either scalar or single-element tuple/list
            if isinstance(value, Sequence) and not isinstance(value, str):
                if len(value) != 1:
                    raise ValueError(
                        "Inequality views not supported - use scalar value."
                    )
                target = float(value[0])
            else:
                target = float(value)

            if isinstance(key, tuple):  # relative view \mu_i - \mu_j = target
                asset_i, asset_j = key
                i_idx, j_idx = self._asset_index(asset_i), self._asset_index(asset_j)
                row = np.zeros(n)
                row[i_idx], row[j_idx] = 1.0, -1.0
            else:  # absolute view \mu_i = target
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
        """Parse view confidences into a dense array aligned to view keys.

        Args:
            conf: Confidence specification (scalar, dict, or array-like).
            keys: View identifiers in order.

        Returns:
            Optional[np.ndarray]: Confidence vector aligned to views.
        """
        if conf is None:
            return None
        if isinstance(conf, (int, float)):
            return np.full(len(keys), float(conf))
        if isinstance(conf, dict):
            result = []
            for k in keys:
                if k not in conf:
                    logging.getLogger(__name__).warning(
                        "No confidence found for view key %r; defaulting to 1.0. "
                        "Ensure confidence keys match view keys exactly (e.g. tuple for relative views).",
                        k,
                    )
                result.append(float(conf.get(k, 1.0)))
            return np.array(result)
        arr = np.asarray(conf, dtype=float).ravel()
        if arr.size != len(keys):
            raise ValueError("view_confidences length mismatch.")
        return arr

    # ---- \Omega construction ----------------------------------------------
    def _build_omega(self, omega: Any, verbose: bool) -> np.ndarray:
        """Construct the view covariance matrix ``Omega``.

        Args:
            omega: ``"idzorek"`` or array-like Omega specification.
            verbose: Whether to print diagnostics.

        Returns:
            np.ndarray: Omega matrix.
        """
        if self._k == 0:  # no views -> empty \Omega
            return np.zeros((0, 0))

        tau_sigma = self._tau * self._sigma

        # -- Idzorek -----------------------------------------------------
        if isinstance(omega, str) and omega.lower() == "idzorek":
            if self._conf is None:
                raise ValueError("Idzorek requires view_confidences.")
            # Simplified closed-form Idzorek approximation (Walters 2014).
            # For the full iterative 7-step procedure, see Idzorek (2005) Eqs. 12-18.
            # Note: multi-asset basket views in P are not yet supported (pair-wise only).
            diag = []
            base_sigma = tau_sigma if self._idzorek_use_tau else self._sigma
            for i, conf in enumerate(self._conf):
                p_i = self._p[i : i + 1]  # (1, N)
                var_i = (p_i @ base_sigma @ p_i.T).item()  # \sigma^2(view)
                c = np.clip(conf, 1e-6, 1.0 - 1e-6)
                factor = (1.0 - c) / c
                diag.append(factor * var_i)
            omega_mat = np.diag(diag)
            if verbose:
                suffix = "\\tau \\Sigma" if self._idzorek_use_tau else "\\Sigma"
                print(f"[BL] \\Omega from Idzorek confidences (base = {suffix}).")

        # -- default diagonal -------------------------------------------
        elif omega is None:
            omega_mat = np.diag(np.diag(self._p @ tau_sigma @ self._p.T))
            if verbose:
                print("[BL] Omega = tau*diag(P Sigma P^T).")

        # -- user-supplied ----------------------------------------------
        else:
            omega_arr = np.asarray(omega, dtype=float)
            if omega_arr.ndim == 1 and omega_arr.size == self._k:
                omega_mat = np.diag(omega_arr)
            elif omega_arr.shape == (self._k, self._k):
                omega_mat = omega_arr
            else:
                raise ValueError(
                    "omega must be 'idzorek', length-K vector, or KxK matrix."
                )
            if verbose:
                print("[BL] Using user-provided \\Omega.")

        return omega_mat

    # ---- posterior ----------------------------------------------------
    def _compute_posterior(self, verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Compute posterior mean/covariance for the Black-Litterman model.

        Args:
            verbose: Whether to print diagnostics.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Posterior mean and covariance.
        """
        tau_sigma = self._tau * self._sigma

        if self._k == 0:  # no views: posterior = prior
            if verbose:
                print("[BL] No views -> posterior = prior.")
            return self._pi.flatten(), self._sigma

        # P tau Sigma P^T + Omega
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

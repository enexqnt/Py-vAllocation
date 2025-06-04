"""Investor views processing via entropy pooling and Black–Litterman."""

# entropy_pooling and _dual_objective functions are adapted from fortituto-tech https://github.com/fortitudo-tech/fortitudo.tech

from collections.abc import Sequence
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import Bounds, minimize

from .optional import HAS_PANDAS, pd


def _entropy_pooling_dual_objective(
    lagrange_multipliers: np.ndarray,
    log_p_col: np.ndarray,
    lhs: np.ndarray,
    rhs_squeezed: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Objective and gradient for entropy pooling dual optimisation."""

    lagrange_multipliers_col = lagrange_multipliers[:, np.newaxis]

    x = np.exp(log_p_col - 1.0 - lhs.T @ lagrange_multipliers_col)

    rhs_vec = np.atleast_1d(rhs_squeezed)
    objective_value = np.sum(x) + lagrange_multipliers @ rhs_vec
    gradient_vector = rhs_vec - (lhs @ x).squeeze()

    return 1000.0 * objective_value, 1000.0 * gradient_vector


# Backwards compatible alias expected by older tests
def _dual_objective(
    lagrange_multipliers: np.ndarray,
    log_p_col: np.ndarray,
    lhs: np.ndarray,
    rhs_squeezed: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Alias for :func:`_entropy_pooling_dual_objective`."""
    return _entropy_pooling_dual_objective(
        lagrange_multipliers, log_p_col, lhs, rhs_squeezed
    )


def entropy_pooling(
    p: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    G: Optional[np.ndarray] = None,
    h: Optional[np.ndarray] = None,
    method: Optional[str] = None,
) -> np.ndarray:

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
    """
    Generic entropy-pooling engine supporting views on means, vols, skews and
    correlations – all at once (simultaneous EP) or block-wise (iterated EP).

    Parameters
    ----------
    prior_returns : (S × N) ndarray or *DataFrame*, optional
        Historical/simulated return cube.  If omitted you must provide
        `prior_mean` **and** `prior_cov`.
    prior_probabilities : (S,) vector or *Series*, optional
        Scenario probabilities (defaults to uniform).
    prior_mean, prior_cov : vector / matrix (or *Series* / *DataFrame*), optional
        First two moments used to synthesise scenarios when `prior_returns`
        isn’t supplied.
    distribution_fn : callable, optional
        Custom sampler ``f(mu, cov, n[, random_state]) -> (n, N) array``.
        Used only when generating synthetic scenarios.
    num_scenarios : int, default 10000
        Number of synthetic draws if `prior_returns` is *not* given.
    random_state : int or numpy.random.Generator, optional
        Passed to NumPy’s RNG (and to `distribution_fn` if it accepts it).
    mean_views, vol_views, corr_views, skew_views : dict or array-like, optional
        View payloads.  A value can be either ``x`` (equality) or a tuple
        ``('>=', x)``, ``('<', x)`` etc.
        *Keys* are asset names / indices (or pairs thereof for correlations).
    sequential : bool, default *False*
        If *True*, apply view blocks sequentially (iterated EP).
    """

    def __init__(
        self,
        prior_returns: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        prior_probabilities: Optional[Union[np.ndarray, pd.Series]] = None,
        *,
        prior_mean: Optional[Union[np.ndarray, pd.Series]] = None,
        prior_cov: Optional[Union[np.ndarray, pd.DataFrame]] = None,
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
                R = prior_returns.values
                self.assets = list(prior_returns.columns)
                self._use_pandas = True
            else:
                R = np.atleast_2d(np.asarray(prior_returns, float))
                self.assets = list(range(R.shape[1]))
                self._use_pandas = False

            S, N = R.shape

            if prior_probabilities is None:
                p0 = np.full((S, 1), 1.0 / S)
            else:
                p = np.asarray(prior_probabilities, float).ravel()
                if p.size != S:
                    raise ValueError(
                        "`prior_probabilities` must match the number of scenarios."
                    )
                p0 = p.reshape(-1, 1)

        else:
            if prior_mean is None or prior_cov is None:
                raise ValueError(
                    "Provide either `prior_returns` or both `prior_mean` and "
                    "`prior_cov`."
                )

            if isinstance(prior_mean, pd.Series):
                mu = prior_mean.values.astype(float)
                self.assets = list(prior_mean.index)
                self._use_pandas = True
            else:
                mu = np.asarray(prior_mean, float).ravel()
                self.assets = list(range(mu.size))
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
                R = rng.multivariate_normal(mu, cov, size=num_scenarios)
            else:
                try:
                    R = distribution_fn(mu, cov, num_scenarios, rng)
                except TypeError:
                    R = distribution_fn(mu, cov, num_scenarios)

            R = np.atleast_2d(np.asarray(R, float))
            if R.shape != (num_scenarios, N):
                raise ValueError(
                    "`distribution_fn` must return shape "
                    f"({num_scenarios}, {N}), got {R.shape}."
                )

            S = num_scenarios
            p0 = np.full((S, 1), 1.0 / S)

        self.R = R
        self.p0 = p0

        mu0 = (R.T @ p0).flatten()
        cov0 = np.cov(R.T, aweights=p0.flatten())
        var0 = np.diag(cov0)

        self.mu0, self.var0 = mu0, var0

        def _vec_to_dict(vec_like, name):
            if vec_like is None:
                return {}
            if isinstance(vec_like, dict):
                return vec_like

            vec = np.asarray(vec_like, float).ravel()
            if vec.size != len(self.assets):
                raise ValueError(f"`{name}` must have length {len(self.assets)}.")
            return {a: vec[i] for i, a in enumerate(self.assets)}

        self.mean_views = _vec_to_dict(mean_views, "mean_views")
        self.vol_views = _vec_to_dict(vol_views, "vol_views")
        self.skew_views = _vec_to_dict(skew_views, "skew_views")
        self.corr_views = corr_views or {}
        self.sequential = bool(sequential)

        self.posterior_probabilities = self._compute_posterior_probabilities()

        q = self.posterior_probabilities
        mu_post = (R.T @ q).flatten()
        cov_post = np.cov(R.T, aweights=q.flatten(), bias=True)

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
        Convert a view value into *(operator, target)* form.

        Accepted syntaxes
        -----------------
        * ``x``              → ('==', x)
        * ``('>=', x)``      → ('>=', x)         (same for ``<=``, ``>``, ``<``)

        For **relative mean views** the target *x* is interpreted as the
        difference μ₁ − μ₂ compared with *x*.  Example::

            mean_views = {('Asset A', 'Asset B'): ('>=', 0.0)}
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
        Return the position of *key* in ``self.assets``, accepting either
        the exact label or a numeric string that can be cast to int.
        """
        if key in self.assets:
            return self.assets.index(key)
        if isinstance(key, str) and key.isdigit():  # "0", "1", …
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
        Translate *view_dict* → (A_eq, b_eq, G_ineq, h_ineq) lists
        suitable for an entropy-pooling call.
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
                    row = R[:, i] - R[:, j]  # (S,)
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
        EP core: handles “simultaneous” vs “iterated” processing without
        confidence blending (full confidence in views).
        """
        R, p0 = self.R, self.p0
        mu_cur, var_cur = self.mu0.copy(), self.var0.copy()

        def do_ep(prior, A_eq, b_eq, G_ineq, h_ineq):
            S = R.shape[0]
            A_eq.append(np.ones(S))
            b_eq.append(1.0)
            A = np.vstack(A_eq) if A_eq else np.zeros((0, S))
            b = np.array(b_eq, float).reshape(-1, 1) if b_eq else np.zeros((0, 1))

            if G_ineq:
                G = np.vstack(G_ineq)
                h = np.array(h_ineq, float).reshape(-1, 1)
            else:
                G, h = None, None

            return entropy_pooling(prior, A, b, G, h)  # (S × 1)

        if not any((self.mean_views, self.vol_views, self.skew_views, self.corr_views)):
            return p0

        if self.sequential:
            q_last = p0
            for mtype, vd in [
                ("mean", self.mean_views),
                ("vol", self.vol_views),
                ("skew", self.skew_views),
                ("corr", self.corr_views),
            ]:
                if vd:
                    Aeq, beq, G, h = self._build_constraints(vd, mtype, mu_cur, var_cur)
                    q_last = do_ep(q_last, Aeq, beq, G, h)

                    mu_cur = (R.T @ q_last).flatten()
                    var_cur = ((R - mu_cur) ** 2).T @ q_last
                    var_cur = var_cur.flatten()

            return q_last

        A_all, b_all, G_all, h_all = [], [], [], []
        for mtype, vd in [
            ("mean", self.mean_views),
            ("vol", self.vol_views),
            ("skew", self.skew_views),
            ("corr", self.corr_views),
        ]:
            if vd:
                Aeq, beq, G, h = self._build_constraints(vd, mtype, mu_cur, var_cur)
                A_all.extend(Aeq)
                b_all.extend(beq)
                G_all.extend(G)
                h_all.extend(h)

        return do_ep(p0, A_all, b_all, G_all, h_all)

    def get_posterior_probabilities(self) -> np.ndarray:
        """Return the (S × 1) posterior probability vector."""
        return self.posterior_probabilities

    def get_posterior(
        self,
    ) -> Tuple[Union[np.ndarray, pd.Series], Union[np.ndarray, pd.DataFrame]]:
        """Return *(posterior mean, posterior covariance)*."""
        return self.posterior_returns, self.posterior_cov


class BlackLittermanProcessor:
    """
    Black–Litterman engine supporting **equality mean views** (absolute or
    relative).  Inequality views (>, >=, <, <=) are *not* handled.

    Parameters
    ----------
    prior_cov : (N, N) array-like
        Prior (sample- or market-implied) covariance Σ.
    prior_mean, market_weights, pi
        Mutually exclusive ways to provide the prior mean π.  Exactly one
        of them must be supplied (see Notes below).
    risk_aversion : float, default 1.0
        δ in π = δ Σ w when *market_weights* is supplied.
        **Must be positive.**
    tau : float, default 0.05
        Prior shrinkage parameter τ.
    idzorek_use_tau : bool, default True
        When constructing Ω from Idzorek confidences, decide whether to use
        τ Σ (True — original He & Litterman convention) or Σ (False — the
        alternative sometimes found in practice).
    mean_views : dict or array-like, optional
        Equality mean views in *FlexibleViewsProcessor* grammar:
        ``{"Asset": 0.02, ("A", "B"): 0.00}`` or a length-N vector
        with absolute views per asset.
    view_confidences : float | list | dict, optional
        Confidence c ∈ (0, 1] per view (used only if Ω = "idzorek").
    omega : {"idzorek"} | array-like, optional
        View-covariance matrix Ω.  If omitted, Ω = τ·diag(P Σ Pᵀ).
    verbose : bool, default False
        Print processing steps.

    Notes
    -----
    The prior mean π can be supplied in three mutually exclusive ways:

    1. **pi**              – direct numeric vector.
    2. **market_weights**  – CAPM equilibrium π = δ Σ w.
    3. **prior_mean**      – treat the sample mean as π.

    Methods
    -------
    get_posterior() -> (posterior_mean, posterior_cov)
        Return posterior quantities as NumPy arrays or Pandas objects,
        matching the type of the inputs.
    """

    # ------------------------------------------------------------------ #
    # public helper
    # ------------------------------------------------------------------ #
    def get_posterior(
        self,
    ) -> Tuple[Union[np.ndarray, pd.Series], Union[np.ndarray, pd.DataFrame]]:
        """Return *(posterior_mean, posterior_covariance)*."""
        return self._posterior_mean, self._posterior_cov

    # ------------------------------------------------------------------ #
    # constructor
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        prior_cov: Union[np.ndarray, pd.DataFrame],
        prior_mean: Optional[Union[np.ndarray, pd.Series]] = None,
        market_weights: Optional[Union[np.ndarray, pd.Series]] = None,
        risk_aversion: float = 1.0,
        tau: float = 0.05,
        idzorek_use_tau: bool = True,
        pi: Optional[Union[np.ndarray, pd.Series]] = None,
        mean_views: Any = None,
        view_confidences: Any = None,
        omega: Any = None,
        verbose: bool = False,
    ) -> None:
        import warnings

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
            if isinstance(value, Sequence):
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

# entropy_pooling and _dual_objective functions are adapted from fortituto-tech https://github.com/fortitudo-tech/fortitudo.tech

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from scipy.optimize import minimize, Bounds
import numpy as np
import pandas as pd
import warnings

def entropy_pooling(
        p: np.ndarray, A: np.ndarray, b: np.ndarray, G: Optional[np.ndarray] = None,
        h: Optional[np.ndarray] = None, method: Optional[str] = None) -> np.ndarray:
    """Compute Entropy Pooling posterior probabilities.

    Args:
        p: Prior probability vector with shape (S, 1).
        A: Equality constraint matrix with shape (M, S).
        b: Equality constraint vector with shape (M, 1).
        G: Inequality constraint matrix with shape (N, S), optional.
        h: Inequality constraint vector with shape (N, 1), optional.
        method: Optimization method: {'TNC', 'L-BFGS-B'}. Default 'TNC'.

    Returns:
        Posterior probability vector with shape (S, 1).
    """
    if method is None:
        method = 'TNC'
    elif method not in ('TNC', 'L-BFGS-B'):
        raise ValueError(f'Method {method} not supported. Choose TNC or L-BFGS-B.')

    len_b = len(b)
    if G is None or h is None:
        lhs = A
        rhs = b
        bounds = Bounds([-np.inf] * len_b, [np.inf] * len_b)
    else:
        lhs = np.vstack((A, G))
        rhs = np.vstack((b, h))
        len_h = len(h)
        bounds = Bounds([-np.inf] * len_b + [0] * len_h, [np.inf] * (len_b + len_h))

    log_p = np.log(p + 1e-12) # Add a small epsilon to prevent log(0)
    dual_solution = minimize(
        _dual_objective, x0=np.zeros(lhs.shape[0]), args=(log_p, lhs, rhs),
        method=method, jac=True, bounds=bounds, options={'maxfun': 10000})
    
    # Check for numerical stability of the dual solution
    if not dual_solution.success or np.any(np.isnan(dual_solution.x)) or np.any(np.isinf(dual_solution.x)):
        warnings.warn("Optimization for entropy pooling failed or returned unstable results. Returning prior probabilities.")
        return p # Fallback to prior probabilities
    
    # Calculate q and clamp to avoid extreme values and ensure sum to 1
    q = np.exp(log_p - 1 - lhs.T @ dual_solution.x[:, np.newaxis])
    q = np.maximum(q, 1e-12) # Clamp probabilities to be at least a small positive number
    q /= np.sum(q) # Re-normalize to ensure probabilities sum to 1
    return q


def _dual_objective(
        lagrange_multipliers: np.ndarray, log_p: np.ndarray,
        lhs: np.ndarray, rhs: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute Entropy Pooling dual objective and gradient.

    Args:
        lagrange_multipliers: Lagrange multipliers with shape (M,) or (M + N,).
        log_p: Log of prior probability vector with shape (S, 1).
        lhs: Matrix with shape (M, S) or (M + N, S).
        rhs: Vector with shape (M, 1) or (M + N, 1).

    Returns:
        Dual objective value and gradient.
    """
    lagrange_multipliers = lagrange_multipliers[:, np.newaxis]
    
    # Clamp log_x to prevent overflow/underflow before np.exp
    log_x = np.clip(log_p - 1 - lhs.T @ lagrange_multipliers, -700, 700) 
    x = np.exp(log_x)
    
    # Add a small epsilon to x to prevent issues if x becomes exactly zero
    x = np.maximum(x, 1e-12) 

    gradient = rhs - lhs @ x
    objective = x.T @ (log_x - log_p) - lagrange_multipliers.T @ gradient
    return -1000 * objective.item(), 1000 * gradient.flatten()

class FlexibleViewsProcessor:
    r"""
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
    view_confidences : float, array-like or dict, optional
        Global confidence level *c* ∈ [0, 1] (scalar) or object averaged down
        to one scalar.
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
        view_confidences: Any = None,
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

            # Add a small regularization to the covariance matrix
            # to ensure it's positive definite and prevent numerical issues
            # with multivariate_normal.
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
        # Use np.cov for more robust covariance calculation
        cov0 = np.cov(R.T, aweights=p0.flatten())
        var0 = np.diag(cov0)

        self.mu0, self.var0 = mu0, var0

        if view_confidences is None:
            self.c_global = 1.0
        elif isinstance(view_confidences, (int, float)):
            self.c_global = float(view_confidences)
        elif isinstance(view_confidences, dict):
            self.c_global = float(np.mean(list(view_confidences.values())))
        else:
            self.c_global = float(np.asarray(view_confidences, float).mean())
        self.c_global = np.clip(self.c_global, 0.0, 1.0)

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
        # Use np.cov for more robust covariance calculation
        cov_post = np.cov(R.T, aweights=q.flatten())

        if self._use_pandas:
            self.posterior_returns = pd.Series(mu_post, index=self.assets)
            self.posterior_cov = pd.DataFrame(
                cov_post, index=self.assets, columns=self.assets
            )
        else:
            self.posterior_returns = mu_post
            self.posterior_cov = cov_post

    # ====================================================================== #
    #  private helpers
    # ====================================================================== #
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
        the exact label or a numeric string that can be cast to int."""
        if key in self.assets:
            return self.assets.index(key)
        if isinstance(key, str) and key.isdigit():         # "0", "1", …
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
            else:                           # "<=" or "<"
                G_ineq.append(-row)
                h_ineq.append(-raw)

        # ---- mean -------------------------------------------------------- #
        if moment_type == "mean":
            for key, vw in view_dict.items():
                op, tgt = self._parse_view(vw)

                # ------------ NEW: relative view (μ_A – μ_B ▷ tgt) -------- #
                if isinstance(key, tuple) and len(key) == 2:
                    a1, a2 = key
                    i, j   = self._asset_idx(a1), self._asset_idx(a2)
                    row    = R[:, i] - R[:, j]          # (S,)
                    add(op, row, tgt)
                # ------------ existing absolute view ---------------------- #
                else:
                    idx = self._asset_idx(key)
                    add(op, R[:, idx], tgt)

        # ---- volatility (2nd moment) ................................. #
        elif moment_type == "vol":
            for asset, vw in view_dict.items():
                op, tgt = self._parse_view(vw)
                idx = self._asset_idx(asset)
                raw = tgt**2 + mu[idx] ** 2      # convert σ → E[R²]
                add(op, R[:, idx] ** 2, raw)

        # ---- skewness (3rd moment) ................................... #
        elif moment_type == "skew":
            for asset, vw in view_dict.items():
                op, tgt = self._parse_view(vw)
                idx = self._asset_idx(asset)
                s = np.sqrt(var[idx])
                raw = tgt * s**3 + 3 * mu[idx] * var[idx] + mu[idx] ** 3
                add(op, R[:, idx] ** 3, raw)

        # ---- correlation (cross-moment) .............................. #
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

    # ................................................................. #
    def _compute_posterior_probabilities(self) -> np.ndarray:
        """
        EP core: handles “simultaneous” vs “iterated” processing and
        blends the outcome with the prior via the global confidence *c*.
        """
        R, p0 = self.R, self.p0
        mu_cur, var_cur = self.mu0.copy(), self.var0.copy()

        # helper running one single EP call ................................
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

        # ---- a) no views ............................................. #
        if not any((self.mean_views, self.vol_views, self.skew_views, self.corr_views)):
            q_views = p0

        # ---- b) sequential (iterated) EP ............................. #
        elif self.sequential:
            q_last = p0
            for mtype, vd in [
                ("mean", self.mean_views),
                ("vol", self.vol_views),
                ("skew", self.skew_views),
                ("corr", self.corr_views),
            ]:
                if vd:
                    Aeq, beq, G, h = self._build_constraints(
                        vd, mtype, mu_cur, var_cur
                    )
                    q_last = do_ep(q_last, Aeq, beq, G, h)

                    # update running moments
                    mu_cur = (R.T @ q_last).flatten()
                    var_cur = ((R - mu_cur) ** 2).T @ q_last
                    var_cur = var_cur.flatten()

            q_views = q_last

        # ---- c) one-shot simultaneous EP ............................. #
        else:
            A_all, b_all, G_all, h_all = [], [], [], []
            for mtype, vd in [
                ("mean", self.mean_views),
                ("vol", self.vol_views),
                ("skew", self.skew_views),
                ("corr", self.corr_views),
            ]:
                if vd:
                    Aeq, beq, G, h = self._build_constraints(
                        vd, mtype, mu_cur, var_cur
                    )
                    A_all.extend(Aeq)
                    b_all.extend(beq)
                    G_all.extend(G)
                    h_all.extend(h)

            q_views = do_ep(p0, A_all, b_all, G_all, h_all)

        # ---- d) blend with prior using c ∈ [0,1] ..................... #
        q = (1.0 - self.c_global) * p0 + self.c_global * q_views
        return q

    # ====================================================================== #
    #  public helpers
    # ====================================================================== #
    def get_posterior_probabilities(self) -> np.ndarray:
        """Return the (S × 1) posterior probability vector."""
        return self.posterior_probabilities.flatten()

    def get_posterior(
        self,
    ) -> Tuple[Union[np.ndarray, pd.Series], Union[np.ndarray, pd.DataFrame]]:
        """Return *(posterior mean, posterior covariance)*."""
        return self.posterior_returns, self.posterior_cov

    
class BlackLittermanProcessor:
    """
    Canonical Black–Litterman model.

    Combines a prior (user-specified or market-implied returns) with views
    to yield posterior mean and return covariance that fully reflects
    both sample uncertainty and view uncertainty.

    Implicit assumptions:
      - CAPM equilibrium when using `market_weights` to imply pi (π=δΣw).
      - Normal returns when sampling from (mean, cov).
      - View uncertainty proportional to τΣ unless overridden or Idzorek.

    Methods adhere strictly to the Bayesian derivation:
      posterior mean μ* = π + τΣPᵀ(PτΣPᵀ+Ω)⁻¹(Q−Pπ)
      posterior cov Σ* = Σ + [τΣ − τΣPᵀ(PτΣPᵀ+Ω)⁻¹PτΣ]

    Parameters
    ----------
    prior_mean : array-like (N,), optional
        Prior mean returns π; used if `pi` and `market_weights` are None.
    prior_cov : array-like (N×N)
        Covariance Σ of returns.
    market_weights : array-like (N,), optional
        To compute π = δΣw (CAPM assumption).
    risk_aversion : float, default=1.0
        δ in π = δΣw.
    tau : float, default=0.05
        Scalar on prior uncertainty τ.
    pi : array-like (N,), optional
        Direct user-specified π, overrides `prior_mean` and `market_weights`.
    absolute_views : dict or array-like (N,), optional
        Dict {asset: view} or full-vector Q of length N.
    relative_views : dict, optional
        {(asset_i, asset_j): view_diff}.
    view_confidences : dict or list, optional
        If dict: map view-key to confidence c∈[0,1].
        If list: list of length K in view order.
    omega : 'idzorek', array-like (K,), or (K×K), optional
        If 'idzorek', build Ω via Idzorek's formula from confidences.
        If None, set Ω = τ diag(PΣPᵀ); if array, diag or full matrix.
    verbose : bool, default=False
        Print implicit assumptions and processing steps.

    Raises
    ------
    ValueError
        For missing or mismatched inputs.
    RuntimeError
        If underlying solver fails.
    """
    def __init__(
        self,
        prior_mean:       Optional[Union[np.ndarray, pd.Series]] = None,
        prior_cov:        Optional[Union[np.ndarray, pd.DataFrame]] = None,
        market_weights:   Optional[Union[np.ndarray, pd.Series]] = None,
        risk_aversion:    float   = 1.0,
        tau:              float   = 0.05,
        pi:               Optional[Union[np.ndarray, pd.Series]] = None,
        absolute_views:   Any     = None,
        relative_views:   Optional[Dict] = None,
        view_confidences: Any     = None,
        omega:            Any     = None,
        verbose:          bool    = False
    ):
        import warnings
        # 1) Validate and set Σ
        if prior_cov is None:
            raise ValueError("`prior_cov` is required.")
        # detect pandas input for output type
        self._use_pandas = isinstance(prior_cov, pd.DataFrame)
        if self._use_pandas:
            cov = prior_cov.values
            self.assets = list(prior_cov.index)
        else:
            cov = np.atleast_2d(np.asarray(prior_cov, float))
            self.assets = list(range(cov.shape[0]))
        N = cov.shape[0]
        if cov.shape != (N, N):
            raise ValueError("`prior_cov` must be square N×N.")
        if not np.allclose(cov, cov.T, atol=1e-8):
            warnings.warn("`prior_cov` not symmetric; symmetrizing.")
            cov = (cov + cov.T) / 2
        self.Sigma = cov

        # 2) Determine π
        self._pi_source = None
        if pi is not None:
            pi_arr = np.asarray(pi, float).ravel()
            if pi_arr.size != N:
                raise ValueError("`pi` must have length N.")
            self.pi = pi_arr.reshape(-1, 1)
            self._pi_source = 'user'
            if verbose: print("[BL] Using user-supplied π.")
        elif market_weights is not None:
            w = np.asarray(market_weights, float).ravel()
            if w.size != N:
                raise ValueError("`market_weights` must have length N.")
            w = w / w.sum()
            self.pi = risk_aversion * cov.dot(w.reshape(-1, 1))
            self._pi_source = 'market'
            if verbose: print("[BL] Implied π = δΣw (CAPM).")
        elif prior_mean is not None:
            mu = (prior_mean.values
                  if isinstance(prior_mean, pd.Series)
                  else np.asarray(prior_mean, float).ravel())
            if mu.size != N:
                raise ValueError("`prior_mean` must have length N.")
            self.pi = mu.reshape(-1, 1)
            self._pi_source = 'prior_mean'
            if verbose: print("[BL] Using `prior_mean` as π.")
        else:
            raise ValueError("No source for π provided.")

        self.tau   = float(tau)
        self.delta = float(risk_aversion)

        # 3) Parse absolute_views into dict
        if absolute_views is None:
            abs_dict = {}
        elif isinstance(absolute_views, dict):
            abs_dict = absolute_views.copy()
        else:
            arr = np.asarray(absolute_views, float).ravel()
            if arr.size != N:
                raise ValueError("`absolute_views` must have length N.")
            abs_dict = {i: arr[i] for i in range(N)}
            if verbose: print("[BL] Full-vector absolute views.")

        # 4) Parse relative_views
        rel_dict = relative_views or {}

        # 5) Build P and Q
        P_rows, Q_vals = [], []
        for asset, val in abs_dict.items():
            idx = (self.assets.index(asset)
                   if isinstance(asset, str) else asset)
            row = np.zeros(N); row[idx] = 1.0
            P_rows.append(row); Q_vals.append(float(val))
        for (a, b), val in rel_dict.items():
            i = (self.assets.index(a)
                 if isinstance(a, str) else a)
            j = (self.assets.index(b)
                 if isinstance(b, str) else b)
            row = np.zeros(N); row[i] = 1.0; row[j] = -1.0
            P_rows.append(row); Q_vals.append(float(val))
        self.P = np.vstack(P_rows) if P_rows else np.zeros((0, N))
        self.Q = np.array(Q_vals).reshape(-1, 1) if Q_vals else np.zeros((0, 1))
        self.K = self.P.shape[0]
        if verbose: print(f"[BL] Built P {self.P.shape}, Q {self.Q.shape}.")

        # 6) Parse confidences
        if view_confidences is None:
            self.conf = None
        elif isinstance(view_confidences, dict):
            keys = list(abs_dict.keys()) + list(rel_dict.keys())
            self.conf = np.array([
                float(view_confidences.get(k, 1.0)) for k in keys
            ])
        else:
            arr = np.asarray(view_confidences, float).ravel()
            if arr.size != self.K:
                raise ValueError("`view_confidences` must match number of views.")
            self.conf = arr
        if self.conf is not None and verbose:
            print(f"[BL] View confidences: {self.conf}.")

        # 7) Build Omega
        if isinstance(omega, str) and omega.lower() == 'idzorek':
            if self.conf is None:
                raise ValueError("Idzorek requires view_confidences.")
            omegas = []
            for k in range(self.K):
                c = np.clip(self.conf[k], 1e-6, 1-1e-6)
                factor = (1 - c) / c
                Pi_k = self.P[k:k+1, :]
                var_k = Pi_k.dot(self.tau * cov).dot(Pi_k.T).item()
                omegas.append(self.tau * factor * var_k)
            self.Omega = np.diag(omegas)
            if verbose: print("[BL] Ω from Idzorek formula.")
        elif omega is None:
            diag = np.diag(self.P.dot(self.tau * cov).dot(self.P.T))
            self.Omega = np.diag(diag)
            if verbose: print("[BL] Ω = τ diag(PΣPᵀ).")
        else:
            Om = np.asarray(omega, float)
            if Om.ndim == 1 and Om.size == self.K:
                self.Omega = np.diag(Om)
            elif Om.shape == (self.K, self.K):
                self.Omega = Om
            else:
                raise ValueError("omega must be 'idzorek', length-K, or K×K.")
            if verbose: print("[BL] Using user-provided Ω.")

        # 8) Compute posterior
        mu_post, cov_post = self._compute_posterior(verbose)
        # wrap outputs according to original input
        if self._use_pandas:
            self.posterior_returns = pd.Series(mu_post, index=self.assets)
            self.posterior_cov = pd.DataFrame(cov_post, index=self.assets, columns=self.assets)
        else:
            self.posterior_returns = mu_post
            self.posterior_cov = cov_post

    def _compute_posterior(self, verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        tS = self.tau * self.Sigma
        if self.K == 0:
            if verbose: print("[BL] No views: posterior mean = π.")
            mu_post = self.pi.flatten()
            cov_post = self.Sigma
        else:
            A = self.P.dot(tS).dot(self.P.T) + self.Omega
            invA = np.linalg.inv(A)
            diff = self.Q - self.P.dot(self.pi)
            mu_post = (self.pi + tS.dot(self.P.T).dot(invA).dot(diff)).flatten()
            var_mean = tS - tS.dot(self.P.T).dot(invA).dot(self.P.dot(tS))
            cov_post = self.Sigma + var_mean
            if verbose: print("[BL] Posterior mean and covariance computed.")
        return mu_post, cov_post

    def get_posterior(self) -> Tuple[Union[np.ndarray, pd.Series], Union[np.ndarray, pd.DataFrame]]:
        """Return (posterior_returns, posterior_cov)."""
        return self.posterior_returns, self.posterior_cov

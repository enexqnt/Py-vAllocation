"""Scenario-based stress testing helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .moments import estimate_sample_moments
from .probabilities import (
    compute_effective_number_scenarios,
    generate_exp_decay_probabilities,
    generate_gaussian_kernel_probabilities,
    resolve_probabilities,
)
from .utils.functions import portfolio_cvar, portfolio_var

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
ProbabilityLike = Union[np.ndarray, pd.Series, Sequence[float]]
WeightsLike = Union[np.ndarray, pd.Series, pd.DataFrame, Mapping[str, float]]

__all__ = [
    "stress_test",
    "stress_invariants",
    "exp_decay_stress",
    "kernel_focus_stress",
    "entropy_pooling_stress",
    "linear_map",
]


def _align_weights(
    weights: WeightsLike,
    asset_names: Optional[Sequence[str]],
) -> Tuple[np.ndarray, Optional[Sequence[str]], Sequence[str]]:
    """
    Return weight matrix shaped (N, M), the asset order, and portfolio names.

    Args:
        weights: Weight inputs (Series/DataFrame/array/mapping).
        asset_names: Optional asset name ordering to enforce.

    Returns:
        Tuple[np.ndarray, Optional[Sequence[str]], Sequence[str]]: Weight matrix,
        resolved asset names, and portfolio labels.
    """
    if isinstance(weights, pd.DataFrame):
        matrix = weights.to_numpy(dtype=float)
        idx = list(weights.index)
        names = list(weights.columns)
    elif isinstance(weights, pd.Series):
        matrix = weights.to_numpy(dtype=float).reshape(-1, 1)
        idx = list(weights.index)
        names = [weights.name or "portfolio_0"]
    elif isinstance(weights, Mapping):
        series = pd.Series(weights, dtype=float)
        matrix = series.to_numpy(dtype=float).reshape(-1, 1)
        idx = list(series.index)
        names = [series.name or "portfolio_0"]
    else:
        arr = np.asarray(weights, dtype=float)
        if arr.ndim == 1:
            matrix = arr.reshape(-1, 1)
            idx = None
            names = ["portfolio_0"]
        elif arr.ndim == 2:
            matrix = arr
            idx = None
            names = [f"portfolio_{i}" for i in range(arr.shape[1])]
        else:
            raise ValueError("`weights` must be 1D or 2D.")

    if asset_names is not None:
        if idx is None:
            if matrix.shape[0] != len(asset_names):
                raise ValueError("Weight dimension does not match number of assets.")
        else:
            if set(idx) != set(asset_names):
                missing = sorted(set(asset_names) - set(idx))
                extra = sorted(set(idx) - set(asset_names))
                raise ValueError(
                    f"Weight labels do not match asset names. Missing={missing}, extra={extra}"
                )
            order = [idx.index(name) for name in asset_names]
            matrix = matrix[order, :]
            idx = list(asset_names)
    return matrix, idx, names


def _kl_divergence(p_star: np.ndarray, p_nom: np.ndarray) -> float:
    """Compute the KL divergence ``KL(p* || p)`` for probability vectors.

    Args:
        p_star: Stressed probability vector.
        p_nom: Nominal probability vector.

    Returns:
        float: KL divergence value.
    """
    eps = 1e-16
    p1 = np.clip(p_star, eps, None)
    p0 = np.clip(p_nom, eps, None)
    return float(np.sum(p1 * np.log(p1 / p0)))


def stress_test(
    weights: WeightsLike,
    scenarios: ArrayLike,
    *,
    probabilities: Optional[ProbabilityLike] = None,
    stressed_probabilities: Optional[ProbabilityLike] = None,
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    confidence: float = 0.95,
    demean: bool = False,
) -> pd.DataFrame:
    """
    Evaluate allocations under nominal and stressed conditions.

    Args:
        weights: Portfolio weights (Series, DataFrame, NumPy array, or mapping).
            Multiple portfolios can be evaluated at once by passing a 2-D array/DataFrame.
        scenarios: Scenario matrix ``R`` with shape ``(T, N)`` (rows = observations,
            columns = assets). Accepts pandas objects or ndarrays.
        probabilities: Optional nominal scenario probabilities ``p``. Defaults to a
            uniform distribution across scenarios.
        stressed_probabilities: Optional stressed probabilities ``p*`` on the same
            scenario grid. When supplied alongside ``probabilities`` the KL divergence
            ``KL(p* || p)`` is reported.
        transform: Callable applied to ``R`` (after conversion to ``float``) to obtain
            stressed scenarios (e.g., shocks, factor projections).
        confidence: Confidence level for VaR/CVaR metrics (default ``0.95``).
            E.g. 0.95 means 5% tail CVaR. Risk is reported as a positive loss.
        demean: When ``True`` the scenario P&L used for VaR/CVaR is demeaned by the
            respective probabilities.

    Returns:
        pd.DataFrame: Tidy DataFrame indexed by portfolio label containing nominal
        metrics, stressed metrics (when supplied), effective number of scenarios,
        and optional KL divergence.

    Examples:
        >>> import numpy as np
        >>> from pyvallocation.stress import stress_test, linear_map
        >>> weights = np.array([0.6, 0.4])
        >>> scenarios = np.array([[0.01, 0.02], [0.00, 0.01], [-0.02, 0.00]])
        >>> shock = linear_map(scale=1.5)  # magnify returns by 50%
        >>> df = stress_test(weights, scenarios, transform=shock)
    """
    conf = confidence

    if not 0.0 < conf < 1.0:
        raise ValueError("`confidence` must be in (0, 1).")

    if isinstance(scenarios, pd.DataFrame):
        R_arr = scenarios.to_numpy(dtype=float)
        asset_names = list(scenarios.columns)
    else:
        R_arr = np.asarray(scenarios, dtype=float)
        asset_names = None

    if R_arr.ndim != 2:
        raise ValueError("`scenarios` must be a 2D array-like.")

    n_scenarios, n_assets = R_arr.shape
    p_nom = resolve_probabilities(probabilities, n_scenarios)
    W, aligned_assets, portfolio_names = _align_weights(weights, asset_names)
    if W.shape[0] != n_assets:
        raise ValueError("Weight dimension does not match number of assets in `scenarios`.")

    def _metrics(
        R_matrix: np.ndarray, probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute mean, volatility, VaR, CVaR, and ENS for a scenario set.

        Args:
            R_matrix: Scenario matrix.
            probs: Scenario probabilities.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]: Mean, volatility,
            VaR, CVaR, and effective number of scenarios.
        """
        mu, sigma = estimate_sample_moments(R_matrix, probs)
        mu_vec = np.asarray(mu, dtype=float).reshape(-1, 1)
        sigma_mat = np.asarray(sigma, dtype=float)
        mean = (W.T @ mu_vec).reshape(-1)
        var = np.asarray(portfolio_var(W, R_matrix, probs, confidence=conf, demean=demean)).reshape(-1)
        cvar = np.asarray(portfolio_cvar(W, R_matrix, probs, confidence=conf, demean=demean)).reshape(-1)
        stdev = np.sqrt(np.einsum("ik,kl,il->i", W.T, sigma_mat, W.T))
        ens = compute_effective_number_scenarios(probs)
        return mean, stdev, var, cvar, ens

    ret_nom, sd_nom, var_nom, cvar_nom, ens_nom = _metrics(R_arr, p_nom)

    columns = {
        "return_nom": ret_nom,
        "stdev_nom": sd_nom,
        f"VaR{int(round(conf * 100))}_nom": var_nom,
        f"CVaR{int(round(conf * 100))}_nom": cvar_nom,
        "ENS_nom": np.full_like(ret_nom, ens_nom, dtype=float),
    }

    stressed_R = R_arr
    if transform is not None:
        stressed_R = np.asarray(transform(R_arr), dtype=float)
        if stressed_R.shape != R_arr.shape:
            raise ValueError("`transform` must return an array with the same shape as `scenarios`.")
        if not np.all(np.isfinite(stressed_R)):
            raise ValueError("`transform` produced non-finite values in stressed scenarios.")

    if stressed_probabilities is not None or transform is not None:
        p_star = (
            resolve_probabilities(
                stressed_probabilities,
                n_scenarios,
                name="stressed_probabilities",
            )
            if stressed_probabilities is not None
            else p_nom
        )
        ret_st, sd_st, var_st, cvar_st, ens_st = _metrics(stressed_R, p_star)
        columns.update(
            {
                "return_stress": ret_st,
                "stdev_stress": sd_st,
                f"VaR{int(round(conf * 100))}_stress": var_st,
                f"CVaR{int(round(conf * 100))}_stress": cvar_st,
                "ENS_stress": np.full_like(ret_st, ens_st, dtype=float),
            }
        )
        if stressed_probabilities is not None:
            columns["KL_q_p"] = np.full_like(ret_st, _kl_divergence(p_star, p_nom), dtype=float)

    df = pd.DataFrame(columns, index=portfolio_names)
    if aligned_assets is not None:
        df.index.name = "portfolio"
    return df


def exp_decay_stress(
    weights: WeightsLike,
    scenarios: ArrayLike,
    *,
    probabilities: Optional[ProbabilityLike] = None,
    half_life: int = 60,
    confidence: float = 0.95,
    demean: bool = False,
) -> pd.DataFrame:
    """
    Historical-simulation stress with exponential decay weights.

    The helper builds stressed probabilities using
    :func:`pyvallocation.probabilities.generate_exp_decay_probabilities` and
    passes them to :func:`stress_test`.

    Args:
        weights, scenarios: See :func:`stress_test`.
        probabilities: Nominal scenario probabilities ``p``. Defaults to uniform.
        half_life: Half-life (in observations) of the exponential decay kernel.
            Defaults to ``60``.
        confidence, demean: Risk settings forwarded to :func:`stress_test`.

    Returns:
        pd.DataFrame: Tidy comparison between nominal and stressed metrics.

    Examples:
        >>> import numpy as np
        >>> from pyvallocation.stress import exp_decay_stress
        >>> weights = np.array([0.5, 0.5])
        >>> scenarios = np.array([[0.01, 0.00], [-0.02, 0.03], [0.015, -0.01]])
        >>> df = exp_decay_stress(weights, scenarios, half_life=2)
    """
    scenario_array = np.asarray(scenarios, dtype=float)
    p_star = generate_exp_decay_probabilities(scenario_array.shape[0], half_life)
    return stress_test(
        weights,
        scenarios,
        probabilities=probabilities,
        stressed_probabilities=p_star,
        confidence=confidence,
        demean=demean,
    )


def kernel_focus_stress(
    weights: WeightsLike,
    scenarios: ArrayLike,
    *,
    focus_series: ArrayLike,
    probabilities: Optional[ProbabilityLike] = None,
    bandwidth: Optional[float] = None,
    target: Optional[float] = None,
    confidence: float = 0.95,
    demean: bool = False,
) -> pd.DataFrame:
    """
    Gaussian-kernel stress that focuses on a state variable.

    Args:
        weights, scenarios: See :func:`stress_test`.
        focus_series: One-dimensional feature (e.g., realised volatility) with
            length equal to the number of scenarios.
        probabilities: Nominal probabilities ``p`` (defaults to uniform).
        bandwidth: Optional kernel bandwidth ``h`` supplied to
            :func:`pyvallocation.probabilities.generate_gaussian_kernel_probabilities`.
        target: Target state ``x_T`` around which probability mass is concentrated.
            When omitted, the last observation is used.
        confidence, demean: Risk settings forwarded to :func:`stress_test`.

    Returns:
        pd.DataFrame: Tidy comparison between nominal and stressed metrics.

    Examples:
        >>> import numpy as np
        >>> from pyvallocation.stress import kernel_focus_stress
        >>> returns = np.array([[0.01, 0.00], [-0.02, 0.03], [0.015, -0.01]])
        >>> vol_proxy = np.array([0.10, 0.15, 0.30])  # e.g. rolling volatility
        >>> df = kernel_focus_stress([0.5, 0.5], returns, focus_series=vol_proxy, target=0.30)
    """
    v = np.asarray(focus_series, dtype=float).reshape(-1)
    if v.shape[0] != np.asarray(scenarios, dtype=float).shape[0]:
        raise ValueError("`focus_series` length must match the number of scenarios.")
    p_star = generate_gaussian_kernel_probabilities(v, h=bandwidth, x_T=target)
    return stress_test(
        weights,
        scenarios,
        probabilities=probabilities,
        stressed_probabilities=p_star,
        confidence=confidence,
        demean=demean,
    )


def entropy_pooling_stress(
    weights: WeightsLike,
    scenarios: ArrayLike,
    *,
    posterior_probabilities: ProbabilityLike,
    probabilities: Optional[ProbabilityLike] = None,
    confidence: float = 0.95,
    demean: bool = False,
) -> pd.DataFrame:
    """
    Stress test using posterior probabilities produced by entropy pooling.

    Args:
        weights, scenarios: See :func:`stress_test`.
        posterior_probabilities: Probability vector returned by
            :func:`pyvallocation.views.entropy_pooling` or
            :class:`pyvallocation.views.FlexibleViewsProcessor`.
        probabilities: Nominal probabilities ``p``. Defaults to uniform.
        confidence, demean: Risk settings forwarded to :func:`stress_test`.

    Returns:
        pd.DataFrame: Tidy comparison between nominal and stressed metrics including
        KL divergence between ``p*`` and ``p``.

    Examples:
        >>> import numpy as np
        >>> from pyvallocation.stress import entropy_pooling_stress
        >>> scenarios = np.array([[0.01, -0.02], [0.02, 0.01], [-0.03, 0.00]])
        >>> posterior = np.array([0.10, 0.70, 0.20])  # output of entropy_pooling
        >>> df = entropy_pooling_stress([0.4, 0.6], scenarios, posterior_probabilities=posterior)
    """
    return stress_test(
        weights,
        scenarios,
        probabilities=probabilities,
        stressed_probabilities=posterior_probabilities,
        confidence=confidence,
        demean=demean,
    )


def linear_map(
    *,
    mean_shift: Optional[ArrayLike] = None,
    scale: Optional[float] = None,
    matrix: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a linear scenario transform ``R -> R @ matrix.T * scale + mean_shift``.

    Args:
        mean_shift: Optional vector added to every scenario (shape ``(N,)``).
        scale: Scalar multiplier applied after the optional matrix projection.
        matrix: Optional matrix ``B`` with shape ``(N_out, N_in)``. When provided,
            scenarios are multiplied on the right by ``B^T``.

    Returns:
        Callable[[np.ndarray], np.ndarray]: Function that accepts a scenario matrix
        and returns a transformed copy.

    Examples:
        >>> import numpy as np
        >>> from pyvallocation.stress import linear_map
        >>> R = np.array([[0.01, 0.00], [-0.02, 0.03]])
        >>> transform = linear_map(mean_shift=np.array([0.0, -0.01]), scale=1.2)
    """

    def _transform(R: np.ndarray) -> np.ndarray:
        """Apply the configured linear transformation to scenarios.

        Args:
            R: Scenario matrix.

        Returns:
            np.ndarray: Transformed scenarios.
        """
        X = np.asarray(R, dtype=float)
        if matrix is not None:
            mat = np.asarray(matrix, dtype=float)
            X = X @ mat.T
        if scale is not None:
            X = X * float(scale)
        if mean_shift is not None:
            shift = np.asarray(mean_shift, dtype=float)
            if shift.ndim == 1:
                if shift.shape[0] != X.shape[1]:
                    raise ValueError("`mean_shift` length must match the number of assets.")
                X = X + shift
            else:
                raise ValueError("`mean_shift` must be a 1D array-like.")
        return X

    return _transform


def stress_invariants(
    invariants: ArrayLike,
    weights: WeightsLike,
    reprice: Union[Callable, dict],
    *,
    stress_views: Optional[dict] = None,
    horizon: int = 1,
    n_simulations: int = 5000,
    seed: Optional[int] = None,
    confidence: float = 0.95,
    demean: bool = False,
) -> pd.DataFrame:
    """Stress test at the **invariant** level through the full Prayer pipeline.

    Applies stress views to the risk-driver distribution via entropy pooling,
    projects to horizon, reprices to P&L, and compares nominal vs stressed
    risk metrics.  This is the correct way to answer questions like
    *"what happens to my portfolio if HY spreads widen by 200 bp?"* — the
    stress is applied to invariants, not directly to P&L.

    Args:
        invariants: Historical risk-driver scenarios ``(T, K)``.
        weights: Portfolio weights ``(N,)`` or Series.
        reprice: Repricing specification — a callable or dict, same format as
            :meth:`~pyvallocation.portfolioapi.PortfolioWrapper.from_invariants`.
        stress_views: Dict of views on the invariants, passed as
            ``mean_views`` to :class:`~pyvallocation.views.FlexibleViewsProcessor`.
            Example: ``{"credit_spread": at_least(0.02)}``.
        horizon: Investment horizon in invariant time steps.
        n_simulations: Number of projected scenarios.
        seed: Random seed for reproducibility.
        confidence: VaR/CVaR confidence level (default 0.95).
        demean: Demean P&L for VaR/CVaR computation.

    Returns:
        pd.DataFrame: Nominal and stressed metrics (return, stdev, VaR, CVaR, ENS).
    """
    from .utils.projection import compose_repricers, project_scenarios, reprice_exp
    from .views import FlexibleViewsProcessor

    is_df = isinstance(invariants, pd.DataFrame)

    # Build repricing function
    if isinstance(reprice, dict):
        if not is_df:
            raise TypeError("Dict reprice requires DataFrame invariants.")
        combined, inst_names = compose_repricers(reprice, invariants.columns.tolist())
    else:
        combined = reprice if reprice is not None else reprice_exp
        inst_names = None

    # Convert to numpy for consistent project_scenarios calls
    inv_np = invariants.to_numpy(dtype=float) if is_df else np.asarray(invariants, dtype=float)
    p_nom = None  # uniform

    # Nominal P&L
    pnl_nom = project_scenarios(inv_np, investment_horizon=horizon, p=p_nom,
                                n_simulations=n_simulations, reprice=combined, seed=seed)
    pnl_nom = np.asarray(pnl_nom, dtype=float)

    # Stressed P&L
    if stress_views:
        processor = FlexibleViewsProcessor(prior_risk_drivers=invariants, mean_views=stress_views)
        p_stressed = processor.get_posterior_probabilities().flatten()
        seed_s = seed + 1 if seed is not None else None
        pnl_stressed = project_scenarios(inv_np, investment_horizon=horizon, p=p_stressed,
                                         n_simulations=n_simulations, reprice=combined, seed=seed_s)
        pnl_stressed = np.asarray(pnl_stressed, dtype=float)
    else:
        pnl_stressed = None

    # Wrap with instrument names and delegate to stress_test
    if inst_names is not None:
        pnl_df = pd.DataFrame(pnl_nom, columns=inst_names)
    elif is_df:
        pnl_df = pd.DataFrame(pnl_nom, columns=invariants.columns)
    else:
        pnl_df = pnl_nom

    if pnl_stressed is not None:
        # Compute stressed probabilities on the P&L grid — since we bootstrapped
        # with stressed probs the P&L scenarios already embed the stress. We
        # compare the two scenario sets via stress_test with a transform.
        nom_probs = resolve_probabilities(None, pnl_nom.shape[0])
        stressed_probs = resolve_probabilities(None, pnl_stressed.shape[0])

        # Compute metrics for both distributions directly
        def _metrics(R, p, w):
            from .moments import estimate_sample_moments
            mu, sigma = estimate_sample_moments(R, p)
            mu_v = np.asarray(mu, float).ravel()
            sigma_m = np.asarray(sigma, float)
            w_v = np.asarray(w, float).ravel()
            mean_r = float(w_v @ mu_v)
            stdev_r = float(np.sqrt(w_v @ sigma_m @ w_v))
            var_r = float(np.asarray(portfolio_var(w_v, R, p, confidence=confidence, demean=demean)).ravel()[0])
            cvar_r = float(np.asarray(portfolio_cvar(w_v, R, p, confidence=confidence, demean=demean)).ravel()[0])
            ens = compute_effective_number_scenarios(p)
            return mean_r, stdev_r, var_r, cvar_r, ens

        w_arr = np.asarray(weights, float).ravel() if not isinstance(weights, pd.Series) else weights.values
        m_n = _metrics(pnl_nom, nom_probs, w_arr)
        m_s = _metrics(pnl_stressed, stressed_probs, w_arr)
        ci = int(round(confidence * 100))
        return pd.DataFrame({
            "return_nom": [m_n[0]], "stdev_nom": [m_n[1]],
            f"VaR{ci}_nom": [m_n[2]], f"CVaR{ci}_nom": [m_n[3]], "ENS_nom": [m_n[4]],
            "return_stressed": [m_s[0]], "stdev_stressed": [m_s[1]],
            f"VaR{ci}_stressed": [m_s[2]], f"CVaR{ci}_stressed": [m_s[3]], "ENS_stressed": [m_s[4]],
        })

    return stress_test(weights, pnl_df, confidence=confidence, demean=demean)

import logging
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def project_mean_covariance(
    mu: Union[np.ndarray, pd.Series],
    cov: Union[np.ndarray, pd.DataFrame],
    annualization_factor: float,
) -> tuple[Union[np.ndarray, pd.Series], Union[np.ndarray, pd.DataFrame]]:
    """Scale mean and covariance by ``annualization_factor``.

    Args:
        mu: Mean vector.
        cov: Covariance matrix.
        annualization_factor: Scaling factor (e.g., 12 for monthly to annual).

    Returns:
        tuple: Scaled mean vector and covariance matrix.
    """

    return mu * annualization_factor, cov * annualization_factor


def convert_scenarios_compound_to_simple(scenarios: np.ndarray) -> np.ndarray:
    """Convert compound returns to simple returns.

    Args:
        scenarios: Log/compound return scenarios.

    Returns:
        np.ndarray: Simple return scenarios.
    """

    return np.exp(scenarios) - 1


def convert_scenarios_simple_to_compound(scenarios: np.ndarray) -> np.ndarray:
    """Convert simple returns to compound (log) returns.

    Args:
        scenarios: Simple return scenarios. All values must be strictly
            greater than -1 (i.e., ``1 + r > 0``).

    Returns:
        np.ndarray: Log/compound return scenarios.

    Raises:
        ValueError: If any scenario has a return <= -1.
    """
    arr = np.asarray(scenarios, dtype=float)
    if np.any(arr <= -1.0):
        raise ValueError(
            "Simple returns must be > -1 for log transformation "
            f"(min value: {float(np.min(arr)):.6f})."
        )
    return np.log(1 + arr)


def _to_numpy(x):
    """Return the underlying ndarray (no copy for ndarray).

    Args:
        x: NumPy array or pandas object.

    Returns:
        np.ndarray: Dense array view/copy.
    """
    return x.to_numpy() if isinstance(x, (pd.Series, pd.DataFrame)) else np.asarray(x)


def _wrap_vector(x_np, template):
    """Wrap 1-D ndarray in the same type as `template` (Series or ndarray).

    Args:
        x_np: Vector to wrap.
        template: Template object (Series or ndarray).

    Returns:
        Union[np.ndarray, pd.Series]: Wrapped vector.
    """
    return (
        pd.Series(x_np, index=template.index, name=template.name)
        if isinstance(template, pd.Series)
        else x_np
    )


def _wrap_matrix(x_np, template):
    """Wrap 2-D ndarray in the same type as `template` (DataFrame or ndarray).

    Args:
        x_np: Matrix to wrap.
        template: Template object (DataFrame or ndarray).

    Returns:
        Union[np.ndarray, pd.DataFrame]: Wrapped matrix.
    """
    return (
        pd.DataFrame(x_np, index=template.index, columns=template.columns)
        if isinstance(template, pd.DataFrame)
        else x_np
    )


def log2simple(mu_g, cov_g):
    r"""\mu,\Sigma of log-returns -> \mu,\Sigma of simple returns (vectorised, pandas-aware).

    Args:
        mu_g: Mean of log-returns.
        cov_g: Covariance of log-returns.

    Returns:
        Tuple[ArrayLike, ArrayLike]: Mean and covariance in simple-return units.
    """
    mu_g_np = _to_numpy(mu_g)
    cov_g_np = _to_numpy(cov_g)

    d = np.diag(cov_g_np)
    exp_mu = np.exp(mu_g_np + 0.5 * d)
    mu_r_np = exp_mu - 1

    cov_r_np = (
        np.exp(mu_g_np[:, None] + mu_g_np + 0.5 * (d[:, None] + d + 2 * cov_g_np))
        - exp_mu[:, None] * exp_mu
    )

    return (_wrap_vector(mu_r_np, mu_g), _wrap_matrix(cov_r_np, cov_g))


def simple2log(mu_r, cov_r):
    r"""\mu,\Sigma of simple returns -> \mu,\Sigma of log-returns (log-normal assumption).

    Args:
        mu_r: Mean of simple returns.
        cov_r: Covariance of simple returns.

    Returns:
        Tuple[ArrayLike, ArrayLike]: Mean and covariance in log-return units.
    """
    mu_r_np = _to_numpy(mu_r)
    cov_r_np = _to_numpy(cov_r)

    m = mu_r_np + 1.0
    if np.any(m <= 0):
        raise ValueError(
            "Mean simple returns must be > -1 for log transformation "
            f"(min 1+mu: {float(np.min(m)):.6f})."
        )
    var_g = np.log1p(np.diag(cov_r_np) / m**2)
    mu_g_np = np.log(m) - 0.5 * var_g

    cov_g_np = np.log1p(cov_r_np / np.outer(m, m))
    np.fill_diagonal(cov_g_np, var_g)  # keep exact variances

    return (_wrap_vector(mu_g_np, mu_r), _wrap_matrix(cov_g_np, cov_r))


def project_scenarios(R, investment_horizon=2, p=None, n_simulations=1000,
                      reprice=None, seed=None):
    """
    Simulate horizon sums by sampling invariants with replacement.

    Implements **P3 (Projection)** of Meucci's Prayer framework: invariants
    are bootstrapped over ``investment_horizon`` steps and summed (random walk).
    If a ``reprice`` callable is supplied, it is applied to the projected risk
    drivers to obtain P&L scenarios (**P4 Pricing**).

    Args:
        R: Historical or simulated invariants (e.g. log-returns, yield changes).
            One-dimensional inputs represent single-instrument (length ``T``).
            Two-dimensional inputs represent ``T`` scenarios across ``N`` risk drivers.
        investment_horizon: Number of draws (with replacement) per simulated path.
            Defaults to ``2``.
        p: Scenario probabilities. When omitted, draws are uniform. Length must
            match the number of rows in ``R``.
        n_simulations: Number of simulated paths to generate. Defaults to ``1000``.
        reprice: Optional callable ``f(projected_risk_drivers) -> pnl_scenarios``
            that converts projected risk-driver changes into P&L or simple-return
            scenarios.  Built-in options:
            :func:`reprice_exp` (stocks: ``exp(Δy) - 1``),
            :func:`reprice_taylor` (greeks/duration: ``θτ + δΔy + ½γΔy²``).
        seed: Random seed for reproducibility.  Defaults to ``None``.

    Returns:
        numpy.ndarray or pandas.Series or pandas.DataFrame: Simulated sums whose
        structure mirrors the input type:

        * 1-D inputs yield length-``n_simulations`` vectors.
        * 2-D inputs yield ``(n_simulations, n_assets)`` matrices.

    Examples:
        >>> import numpy as np
        >>> project_scenarios(
        ...     np.array([0.01, -0.02, 0.03]),
        ...     investment_horizon=2,
        ...     n_simulations=4,
        ... ).shape
        (4,)
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [0.01, -0.02], "b": [0.0, 0.02]})
        >>> project_scenarios(df, investment_horizon=2, n_simulations=3).shape
        (3, 2)
    """

    if not isinstance(investment_horizon, (int, np.integer)):
        investment_horizon = int(investment_horizon)
        logger.warning("investment_horizon cast to int: %d", investment_horizon)

    if investment_horizon <= 0:
        raise ValueError("`investment_horizon` must be a positive integer.")
    if n_simulations <= 0:
        raise ValueError("`n_simulations` must be a positive integer.")

    is_series = isinstance(R, pd.Series)
    is_dataframe = isinstance(R, pd.DataFrame)
    R_np = _to_numpy(R)

    if R_np.ndim not in (1, 2):
        raise ValueError("`R` must be a 1D or 2D array-like of scenarios.")

    num_rows = R_np.shape[0]
    weights = np.asarray(p, dtype=float).reshape(-1) if p is not None else None
    if weights is None:
        weights = np.full(num_rows, 1.0 / num_rows, dtype=float)
    else:
        if weights.shape[0] != num_rows:
            raise ValueError("Probability vector length must match the number of scenarios.")
        if np.any(weights < 0):
            raise ValueError("Scenario probabilities must be non-negative.")
        weight_sum = weights.sum()
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError("Scenario probabilities must sum to a positive finite value.")
        if not np.isclose(weight_sum, 1.0):
            weights = weights / weight_sum

    rng = np.random.default_rng(seed)
    idx = rng.choice(num_rows, size=(n_simulations, investment_horizon), p=weights)
    scenario_sums = R_np[idx].sum(axis=1)

    if reprice is not None:
        scenario_sums = reprice(scenario_sums)

    if is_series:
        template_ser = pd.Series(dtype=float, index=range(n_simulations), name=R.name)
        return _wrap_vector(scenario_sums, template_ser)

    if is_dataframe:
        template_df = pd.DataFrame(index=range(n_simulations), columns=R.columns)
        return _wrap_matrix(scenario_sums, template_df)
    return scenario_sums

# Alias emphasising that inputs can be generic risk drivers, not only returns.
project_risk_drivers = project_scenarios


def simulate_paths(
    R,
    horizon: int = 2,
    n_paths: int = 1000,
    p=None,
    reprice=None,
    seed=None,
):
    """Simulate full trajectory paths by bootstrapping invariants.

    Unlike :func:`project_scenarios` which returns only the terminal sum,
    this function returns the cumulative risk-driver change at **every**
    intermediate step, enabling drawdown analysis and fan charts.

    Args:
        R: Invariant scenarios ``(T, N)`` or ``(T,)`` (e.g. log-returns).
        horizon: Number of bootstrap steps per path.
        n_paths: Number of simulated paths.
        p: Optional scenario probabilities.
        reprice: Optional callable applied to cumulative risk-driver changes
            at each step (e.g. ``reprice_exp`` for cumulative simple returns).
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray: Shape ``(n_paths, horizon, N)`` — cumulative
        risk-driver changes (or repriced values) at each time step.
    """
    R_np = _to_numpy(R)
    if R_np.ndim == 1:
        R_np = R_np.reshape(-1, 1)
    if R_np.ndim != 2:
        raise ValueError("`R` must be 1D or 2D.")

    T, N = R_np.shape
    weights = np.asarray(p, dtype=float).ravel() if p is not None else np.full(T, 1.0 / T)
    weights = weights / weights.sum()

    rng = np.random.default_rng(seed)
    idx = rng.choice(T, size=(n_paths, horizon), p=weights)
    sampled = R_np[idx]  # (n_paths, horizon, N)
    cumulative = np.cumsum(sampled, axis=1)  # cumulative sum along time

    if reprice is not None:
        # Apply repricing at each time step
        out = np.empty_like(cumulative)
        for t in range(horizon):
            out[:, t, :] = reprice(cumulative[:, t, :])
        return out

    return cumulative


# ------------------------------------------------------------------ #
# Built-in repricing functions  (Prayer Step P4)
# ------------------------------------------------------------------ #

def reprice_exp(delta_y: np.ndarray) -> np.ndarray:
    """Reprice via exponentiation (stocks, equity indices).

    Maps projected log-return invariants to simple returns:
    ``P&L / V_0 = exp(Δy) - 1``.

    This is the **exact** repricing for instruments whose risk driver
    is the log-price (Meucci Prayer P4, Eq. 17 for stocks).

    Args:
        delta_y: Projected risk-driver changes ``Y_{T+τ} - Y_T`` (log-returns).

    Returns:
        np.ndarray: Simple-return scenarios.
    """
    return np.exp(delta_y) - 1.0


def reprice_taylor(
    delta_y: np.ndarray,
    *,
    theta: Union[np.ndarray, float, None] = None,
    delta: Union[np.ndarray, float, None] = None,
    gamma: Union[np.ndarray, float, None] = None,
    tau: float = 0.0,
) -> np.ndarray:
    r"""Reprice via Taylor / Greek approximation (options, bonds).

    Approximates the P&L using a second-order expansion around the
    current risk-driver values (Meucci Prayer P4, Eq. 18):

    .. math::

       \text{P\&L} \approx \theta\,\tau + \delta\,\Delta y
                    + \tfrac12\,\gamma\,(\Delta y)^2.

    The coefficients are instrument-specific sensitivities:

    * **Equities**: ``delta=1``, others zero (reduces to linear return).
    * **Options**: ``theta`` (time decay), ``delta`` (option delta),
      ``gamma`` (option gamma).  For multi-factor options, ``delta``
      and ``gamma`` can be vectors/matrices matching risk-driver columns.
    * **Bonds**: ``delta = -duration * price``, ``gamma = convexity * price``.

    Args:
        delta_y: Projected risk-driver changes (``n_sim × n_drivers``).
        theta: Time-decay coefficient(s).  Scalar or per-instrument array.
        delta: First-order sensitivity (delta, -duration, etc.).
        gamma: Second-order sensitivity (gamma, convexity, etc.).
        tau: Time step (e.g. ``1/252`` for daily, ``1/12`` for monthly).

    Returns:
        np.ndarray: Approximate P&L scenarios, same shape as ``delta_y``.
    """
    pnl = np.zeros_like(delta_y, dtype=float)
    if theta is not None:
        pnl = pnl + np.asarray(theta, dtype=float) * tau
    if delta is not None:
        pnl = pnl + np.asarray(delta, dtype=float) * delta_y
    if gamma is not None:
        pnl = pnl + 0.5 * np.asarray(gamma, dtype=float) * delta_y ** 2
    return pnl


def make_repricing_fn(pricing_fn, current_drivers: np.ndarray):
    """Build a repricing callable from an arbitrary pricing function.

    For **full repricing** (Prayer P4), the user supplies a function
    ``pricing_fn(Y)`` that maps risk-driver levels to instrument prices.
    The returned callable computes:

    ``P&L = pricing_fn(Y_T + Δy) - pricing_fn(Y_T)``

    Args:
        pricing_fn: Callable ``f(Y) -> prices``, where ``Y`` has the same
            columns as the risk-driver scenarios.  Must be vectorised over
            the first (scenario) axis.
        current_drivers: Current risk-driver levels ``Y_T`` (1-D array of
            length ``n_drivers``).

    Returns:
        Callable: Repricing function suitable for the ``reprice`` parameter
        of :func:`project_scenarios`.

    Examples:
        >>> # Bond: price = face * exp(-yield * maturity)
        >>> import numpy as np
        >>> face, maturity = 100, 5
        >>> pricing_fn = lambda Y: face * np.exp(-Y * maturity)
        >>> current_yield = np.array([0.03])
        >>> repricer = make_repricing_fn(pricing_fn, current_yield)
        >>> # repricer(delta_y) returns bond P&L scenarios
    """
    y0 = np.asarray(current_drivers, dtype=float)
    p0 = pricing_fn(y0)

    def _reprice(delta_y: np.ndarray) -> np.ndarray:
        y_horizon = y0 + delta_y
        return pricing_fn(y_horizon) - p0

    return _reprice


def compose_repricers(instruments, invariant_columns):
    """Build a repricing function mapping K invariant columns to N instrument P&Ls.

    For mixed portfolios where each instrument may depend on one or more
    risk drivers (Prayer P4).  Three specification formats are supported:

    * **callable** -- 1-to-1: uses the invariant column with the same name
      as the instrument.
    * **("driver", callable)** or **(["driver"], callable)** -- single named
      driver whose name differs from the instrument.
    * **(["d1", "d2"], callable)** -- multi-driver: the callable receives
      an ``(n_sim, n_drivers)`` array.

    Args:
        instruments: Dict mapping **instrument name** to repricing spec.
        invariant_columns: Ordered column names of the invariant scenario
            matrix (length K).

    Returns:
        Tuple[Callable, List[str]]: ``(repricing_fn, instrument_names)``
        where *repricing_fn* maps ``(n_sim, K) -> (n_sim, N)`` and
        *instrument_names* is the ordered list of output column labels.

    Raises:
        KeyError: If a required driver column is not in *invariant_columns*.
        TypeError: If a spec is neither a callable nor a ``(drivers, callable)`` tuple.

    Examples:
        >>> fn, names = compose_repricers(
        ...     {"Stock": reprice_exp,
        ...      "Bond": (["yield_10y"], lambda dy: reprice_taylor(dy, delta=-7, gamma=50)),
        ...      "Call": (["stock_logret", "iv_chg"], my_option_fn)},
        ...     invariant_columns=["stock_logret", "yield_10y", "iv_chg"],
        ... )
        >>> names
        ['Stock', 'Bond', 'Call']
    """
    inv_list = list(invariant_columns)
    inv_idx = {name: i for i, name in enumerate(inv_list)}
    instrument_names = list(instruments.keys())
    parsed = []  # list of (col_indices, fn)

    for inst, spec in instruments.items():
        if callable(spec) and not isinstance(spec, tuple):
            # Plain callable: 1-to-1, instrument name must match an invariant column
            if inst not in inv_idx:
                raise KeyError(
                    f"Instrument '{inst}' has no matching invariant column. "
                    f"Use (drivers, callable) syntax to specify which column(s) to use."
                )
            parsed.append(([inv_idx[inst]], spec))
        elif isinstance(spec, tuple) and len(spec) == 2:
            drivers, fn = spec
            if not callable(fn):
                raise TypeError(f"Second element of spec for '{inst}' must be callable.")
            if isinstance(drivers, str):
                drivers = [drivers]
            col_indices = []
            for d in drivers:
                if d not in inv_idx:
                    raise KeyError(f"Driver '{d}' for instrument '{inst}' not in invariant_columns {inv_list}.")
                col_indices.append(inv_idx[d])
            parsed.append((col_indices, fn))
        else:
            raise TypeError(
                f"Repricing spec for '{inst}' must be a callable or (drivers, callable) tuple, "
                f"got {type(spec).__name__}."
            )

    def _combined(delta_y):
        n_sim = delta_y.shape[0] if delta_y.ndim == 2 else len(delta_y)
        pnl = np.empty((n_sim, len(parsed)), dtype=float)
        for j, (col_idx, fn) in enumerate(parsed):
            if len(col_idx) == 1:
                chunk = delta_y[:, col_idx[0]] if delta_y.ndim == 2 else delta_y
            else:
                chunk = delta_y[:, col_idx]
            pnl[:, j] = np.asarray(fn(chunk), dtype=float).ravel()
        return pnl

    return _combined, instrument_names

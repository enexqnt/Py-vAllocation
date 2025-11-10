import numpy as np
import pandas as pd
from typing import Union


def project_mean_covariance(
    mu: Union[np.ndarray, pd.Series],
    cov: Union[np.ndarray, pd.DataFrame],
    annualization_factor: float,
) -> tuple[Union[np.ndarray, pd.Series], Union[np.ndarray, pd.DataFrame]]:
    """Scale mean and covariance by ``annualization_factor``."""

    return mu * annualization_factor, cov * annualization_factor


def convert_scenarios_compound_to_simple(scenarios: np.ndarray) -> np.ndarray:
    """Convert compound returns to simple returns."""

    return np.exp(scenarios) - 1


def convert_scenarios_simple_to_compound(scenarios: np.ndarray) -> np.ndarray:
    """Convert simple returns to compound returns."""

    return np.log(1 + scenarios)


def _to_numpy(x):
    """Return the underlying ndarray (no copy for ndarray)."""
    return x.to_numpy() if isinstance(x, (pd.Series, pd.DataFrame)) else np.asarray(x)


def _wrap_vector(x_np, template):
    """Wrap 1-D ndarray in the same type as `template` (Series or ndarray)."""
    return (
        pd.Series(x_np, index=template.index, name=template.name)
        if isinstance(template, pd.Series)
        else x_np
    )


def _wrap_matrix(x_np, template):
    """Wrap 2-D ndarray in the same type as `template` (DataFrame or ndarray)."""
    return (
        pd.DataFrame(x_np, index=template.index, columns=template.columns)
        if isinstance(template, pd.DataFrame)
        else x_np
    )


def log2simple(mu_g, cov_g):
    r"""\mu,\Sigma of log-returns -> \mu,\Sigma of simple returns (vectorised, pandas-aware)."""
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
    r"""\mu,\Sigma of simple returns -> \mu,\Sigma of log-returns (log-normal assumption)."""
    mu_r_np = _to_numpy(mu_r)
    cov_r_np = _to_numpy(cov_r)

    m = mu_r_np + 1.0
    var_g = np.log1p(np.diag(cov_r_np) / m**2)
    mu_g_np = np.log(m) - 0.5 * var_g

    cov_g_np = np.log1p(cov_r_np / np.outer(m, m))
    np.fill_diagonal(cov_g_np, var_g)  # keep exact variances

    return (_wrap_vector(mu_g_np, mu_r), _wrap_matrix(cov_g_np, cov_r))


def project_scenarios(R, investment_horizon=2, p=None, n_simulations=1000):
    """
    Simulate horizon sums by sampling scenarios with replacement.

    Parameters
    ----------
    R : array-like or pandas object
        Historical or simulated scenarios. One-dimensional inputs represent
        single-asset returns (length ``T``). Two-dimensional inputs represent
        ``T`` scenarios across ``N`` assets.
    investment_horizon : int, default ``2``
        Number of draws (with replacement) per simulated path.
    p : array-like, optional
        Scenario probabilities. When omitted, draws are uniform. Length must
        match the number of rows in ``R``.
    n_simulations : int, default ``1000``
        Number of simulated paths to generate.

    Returns
    -------
    numpy.ndarray or pandas.Series or pandas.DataFrame
        Simulated sums whose structure mirrors the input type:

        * 1-D inputs yield length-``n_simulations`` vectors.
        * 2-D inputs yield ``(n_simulations, n_assets)`` matrices.

    Examples
    --------
    >>> import numpy as np
    >>> project_scenarios(np.array([0.01, -0.02, 0.03]), investment_horizon=2, n_simulations=4).shape
    (4,)
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [0.01, -0.02], 'b': [0.0, 0.02]})
    >>> project_scenarios(df, investment_horizon=2, n_simulations=3).shape
    (3, 2)
    """

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

    rng = np.random.default_rng()
    idx = rng.choice(num_rows, size=(n_simulations, investment_horizon), p=weights)
    scenario_sums = R_np[idx].sum(axis=1)

    if is_series:
        template_ser = pd.Series(dtype=float, index=range(n_simulations), name=R.name)
        return _wrap_vector(scenario_sums, template_ser)

    if is_dataframe:
        template_df = pd.DataFrame(index=range(n_simulations), columns=R.columns)
        return _wrap_matrix(scenario_sums, template_df)
    return scenario_sums

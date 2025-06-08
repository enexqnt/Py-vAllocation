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
    """μ,Σ of log-returns → μ,Σ of simple returns (vectorised, pandas-aware)."""
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
    """μ,Σ of simple returns → μ,Σ of log-returns (log-normal assumption)."""
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
    Simulate scenario‐based sums of rows drawn from R over a given horizon.

    Parameters
    ----------
    R : np.ndarray or pd.DataFrame or pd.Series
        If 1-D (shape = (n_rows,)), we treat each entry as a possible return (Series-style).
        If 2-D (shape = (n_rows, n_cols)), each row is one multivariate outcome (DataFrame-style).
    investment_horizon : int, default 2
        Number of time-steps to draw for each simulation. For each of the n_simulations,
        we sample `investment_horizon` rows (with replacement).
    p : array-like or None, default None
        Probability weights for sampling each row. If None, rows are drawn uniformly.
        Length must equal the number of rows in R.
    n_simulations : int, default 1000
        Number of simulated “paths” to generate.

    Returns
    -------
    results : np.ndarray or pd.Series or pd.DataFrame
        ─ If `R` was a NumPy array of shape (n_rows,) or (n_rows, n_cols), returns a
          NumPy array:
            • If R was 1-D, output is shape (n_simulations,) – sums across the horizon.
            • If R was 2-D, output is shape (n_simulations, n_cols).
        ─ If `R` was a pandas Series (length = n_rows), returns a pandas Series of length
          n_simulations (sum-over-horizon for each sim).
        ─ If `R` was a pandas DataFrame (shape = (n_rows, n_cols)), returns a pandas
          DataFrame of shape (n_simulations, n_cols), where each row is the sum over
          the randomly drawn horizon for each column.

    Examples
    --------
    # (a) Passing a NumPy 1-D array
    >>> R_np = np.array([0.01, -0.02, 0.03, 0.00])
    >>> out_np = project_scenarios(R_np, investment_horizon=3, n_simulations=5)
    >>> type(out_np)
    <class 'numpy.ndarray'>
    >>> out_np.shape
    (5,)

    # (b) Passing a pandas Series
    >>> R_ser = pd.Series([0.01, -0.02, 0.03, 0.00], name="daily_return")
    >>> out_ser = project_scenarios(R_ser, investment_horizon=3, n_simulations=5)
    >>> type(out_ser)
    <class 'pandas.core.series.Series'>
    >>> out_ser.shape
    (5,)

    # (c) Passing a pandas DataFrame
    >>> R_df = pd.DataFrame({
    ...     "ret_a": [0.01, -0.02, 0.03, 0.00],
    ...     "ret_b": [0.02, -0.01, 0.01, 0.03]
    ... })
    >>> out_df = project_scenarios(R_df, investment_horizon=2, n_simulations=3)
    >>> type(out_df)
    <class 'pandas.core.frame.DataFrame'>
    >>> out_df.shape
    (3, 2)
    """

    is_series = isinstance(R, pd.Series)
    is_dataframe = isinstance(R, pd.DataFrame)
    R_np = _to_numpy(R)
    idx = np.random.choice(
        R_np.shape[0], size=(n_simulations, investment_horizon), p=p / p.sum()
    )
    scenario_sums = R_np[idx].sum(axis=1)

    if is_series:
        template_ser = pd.Series(dtype=float, index=range(n_simulations), name=R.name)
        return _wrap_vector(scenario_sums, template_ser)

    if is_dataframe:
        template_df = pd.DataFrame(index=range(n_simulations), columns=R.columns)
        return _wrap_matrix(scenario_sums, template_df)
    return scenario_sums

import numpy as np
import pandas as pd

def project_mean_covariance(mu,cov, annualization_factor):
    return mu*annualization_factor, cov *annualization_factor

def convert_scenarios_compound_to_simple(scenarios):
    return np.exp(scenarios)-1

def convert_scenarios_simple_to_compound(scenarios):
    return np.log(1+scenarios)

# ---- helpers ---------------------------------------------------------------
def _to_numpy(x):
    """Return the underlying ndarray (no copy for ndarray)."""
    return x.to_numpy() if isinstance(x, (pd.Series, pd.DataFrame)) else np.asarray(x)

def _wrap_vector(x_np, template):
    """Wrap 1-D ndarray in the same type as `template` (Series or ndarray)."""
    return (pd.Series(x_np, index=template.index, name=template.name)
            if isinstance(template, pd.Series) else x_np)

def _wrap_matrix(x_np, template):
    """Wrap 2-D ndarray in the same type as `template` (DataFrame or ndarray)."""
    return (pd.DataFrame(x_np, index=template.index, columns=template.columns)
            if isinstance(template, pd.DataFrame) else x_np)

# ---- log  →  simple --------------------------------------------------------
def log2simple(mu_g, cov_g):
    """μ,Σ of log-returns → μ,Σ of simple returns (vectorised, pandas-aware)."""
    mu_g_np  = _to_numpy(mu_g)
    cov_g_np = _to_numpy(cov_g)

    d        = np.diag(cov_g_np)
    exp_mu   = np.exp(mu_g_np + 0.5 * d)
    mu_r_np  = exp_mu - 1

    cov_r_np = (
        np.exp(mu_g_np[:, None] + mu_g_np + 0.5*(d[:, None] + d + 2*cov_g_np))
        - exp_mu[:, None] * exp_mu
    )

    # wrap back in pandas if needed
    return (_wrap_vector(mu_r_np, mu_g),
            _wrap_matrix(cov_r_np, cov_g))


# ---- simple  →  log --------------------------------------------------------
def simple2log(mu_r, cov_r):
    """μ,Σ of simple returns → μ,Σ of log-returns (log-normal assumption)."""
    mu_r_np  = _to_numpy(mu_r)
    cov_r_np = _to_numpy(cov_r)

    m        = mu_r_np + 1.0
    var_g    = np.log1p(np.diag(cov_r_np) / m**2)
    mu_g_np  = np.log(m) - 0.5 * var_g

    cov_g_np = np.log1p(cov_r_np / np.outer(m, m))
    np.fill_diagonal(cov_g_np, var_g)   # keep exact variances

    return (_wrap_vector(mu_g_np, mu_r),
            _wrap_matrix(cov_g_np, cov_r))

import pytest
import numpy as np
from pyvallocation.utils.functions import (
    _return_portfolio_risk,
    _var_cvar_preprocess,
    _var_calc,
    portfolio_cvar,
    portfolio_var
)

def test_return_portfolio_risk():
    risk_mat_single = np.array([[0.1]])
    assert _return_portfolio_risk(risk_mat_single) == 0.1
    risk_mat_multi = np.array([[0.1, 0.2]])
    result = _return_portfolio_risk(risk_mat_multi)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 2)

def test_var_cvar_preprocess_defaults_and_errors():
    R = np.array([[1, 2], [3, 4]])
    e = np.array([[0.5], [0.5]])
    pf_pnl, p, alpha = _var_cvar_preprocess(e, R, None, None, None)
    assert pf_pnl.shape == (2, 1)
    assert p.shape == (2, 1)
    assert 0 < alpha < 1
    with pytest.raises(ValueError):
        _var_cvar_preprocess(e, R, None, alpha=1.5, demean=True)
    with pytest.raises(ValueError):
        _var_cvar_preprocess(e, R, None, alpha=0.5, demean="yes")

def test_var_calc_and_portfolio_var_cvar():
    pf_pnl = np.array([[1.0], [2.0]])
    p = np.array([[0.5], [0.5]])
    alpha = 0.5
    var = _var_calc(pf_pnl, p, alpha)
    assert pytest.approx(var[0, 0]) == 1.5

    # Test portfolio_var returns negative VaR
    e = np.array([[1.0], [0.0]])
    R = np.array([[1.0, 0.0], [2.0, 0.0]])
    var_port = portfolio_var(e, R, p=p, alpha=alpha, demean=False)
    assert var_port == -1.5

    # Test portfolio_cvar returns negative CVaR
    e2 = np.array([[1.0, 1.0], [0.0, 0.0]])
    cvar_vals = portfolio_cvar(e2, R, p=p, alpha=alpha, demean=False)
    assert isinstance(cvar_vals, np.ndarray)
    assert cvar_vals.shape == (1, 2)
    # For these exposures, CVaR should equal VaR of the loss (= -1.0)
    assert pytest.approx(cvar_vals[0, 0]) == -1.0

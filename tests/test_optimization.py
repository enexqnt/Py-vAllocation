import pytest
import numpy as np
from pyvallocation import optimization

def test_build_G_h_A_b_long_only():
    G, h, A, b = optimization.build_G_h_A_b(3, long_only=True)
    assert G.shape == (3, 3)
    assert np.allclose(h, 0)
    assert A.shape == (1, 3)
    assert np.allclose(A, 1)
    assert np.allclose(b, 1)

def test_build_G_h_A_b_bounds():
    G, h, A, b = optimization.build_G_h_A_b(2, bounds=[(0, 0.5), (0.1, 0.6)])
    # 2 long-only constraints + 4 bounds constraints = 6 rows
    assert G.shape == (6, 2)
    assert h.shape == (6,)
    assert np.allclose(A, 1)
    assert np.allclose(b, 1)

def test_build_G_h_A_b_invalid_bounds():
    with pytest.raises(ValueError):
        optimization.build_G_h_A_b(2, bounds=[(0.5, 0.1), (0, 1)])

def test_meanvariance_efficient_portfolio():
    mean = np.array([0.1, 0.2])
    cov = np.array([[0.01, 0.002], [0.002, 0.02]])
    G, h, A, b = optimization.build_G_h_A_b(2)
    mv = optimization.MeanVariance(mean, cov, G, h, A, b)
    w = mv.efficient_portfolio()
    assert w.shape == (2, 1)
    assert np.isclose(np.sum(w), 1.0)
    assert np.all(w >= -1e-8)

def test_meanvariance_invalid_G_h():
    mean = np.array([0.1, 0.2])
    cov = np.eye(2)
    G = np.eye(2)
    h = None
    A = np.ones((1, 2))
    b = np.array([1.0])
    with pytest.raises(ValueError):
        optimization.MeanVariance(mean, cov, G, h, A, b)

def test_meancvar_invalid_alpha():
    R = np.random.randn(10, 2)
    G, h, A, b = optimization.build_G_h_A_b(2)
    with pytest.raises(ValueError):
        optimization.MeanCVaR(R, G, h, A, b, alpha=1.5)

def test_meancvar_efficient_portfolio_shape():
    R = np.random.randn(10, 2)
    G, h, A, b = optimization.build_G_h_A_b(2)
    mcvar = optimization.MeanCVaR(R, G, h, A, b, alpha=0.95)
    w = mcvar.efficient_portfolio()
    assert w.shape == (2, 1)
    assert np.isclose(np.sum(w), 1.0, atol=1e-6)


def test_meanvariance_with_tcosts():
    mean = np.array([0.1, 0.2])
    cov = np.array([[0.01, 0.002], [0.002, 0.02]])
    G, h, A, b = optimization.build_G_h_A_b(2)
    prev = np.array([0.4, 0.6])
    mv = optimization.MeanVariance(
        mean,
        cov,
        G,
        h,
        A,
        b,
        tcost_lambda=np.array([0.1, 0.2]),
        prev_weights=prev,
    )
    w = mv.efficient_portfolio()
    assert w.shape == (2, 1)
    assert np.isclose(np.sum(w), 1.0)


def test_meancvar_with_tcosts():
    R = np.random.randn(10, 2)
    G, h, A, b = optimization.build_G_h_A_b(2)
    prev = np.array([0.3, 0.7])
    mcvar = optimization.MeanCVaR(
        R,
        G,
        h,
        A,
        b,
        alpha=0.9,
        tcost_lambda=np.array([0.1, 0.2]),
        prev_weights=prev,
    )
    w = mcvar.efficient_portfolio()
    assert w.shape == (2, 1)
    assert np.isclose(np.sum(w), 1.0, atol=1e-6)


def test_robustbayes_portfolio():
    mean = np.array([0.05, 0.1])
    cov = np.array([[0.02, 0.005], [0.005, 0.03]])
    G, h, A, b = optimization.build_G_h_A_b(2)
    rb = optimization.RobustBayes(mean, cov, rho=0.1, gamma=3.0, G=G, h=h, A=A, b=b)
    w = rb.efficient_portfolio()
    assert w.shape == (2, 1)
    assert np.isclose(np.sum(w), 1.0, atol=1e-6)


def test_robust_risk_monotone():
    np.random.seed(0)
    I = 3
    M = np.random.randn(I, I)
    cov = M @ M.T / I
    mean = np.random.randn(I)
    gamma_vec = optimization.RobustBayes.gamma_grid(n=4)
    risks = []
    for g in gamma_vec:
        rb = optimization.RobustBayes(mean, cov, rho=0.5, gamma=g)
        w = rb.efficient_portfolio().flatten()
        risks.append(float(w @ cov @ w))
    assert np.all(np.diff(risks) <= 1e-8)


def test_robust_equals_markowitz_when_rho_zero():
    mean = np.array([0.02, 0.03])
    cov = np.array([[0.01, 0.002], [0.002, 0.02]])
    G, h, A, b = optimization.build_G_h_A_b(2)
    mv = optimization.MeanVariance(mean, cov, G, h, A, b)
    w_mv = mv.efficient_portfolio().flatten()
    rb = optimization.RobustBayes(mean, cov, rho=0.0, gamma=1000.0, G=G, h=h, A=A, b=b)
    w_rb = rb.efficient_portfolio().flatten()
    np.testing.assert_allclose(w_rb, w_mv, rtol=1e-3, atol=1e-3)

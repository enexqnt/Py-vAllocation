import numpy as np
import pytest

from pyvallocation.optimization import RelaxedRiskParity


def test_relaxed_risk_parity_reduces_to_risk_parity_when_lambda_zero():
    mu = np.array([0.05, 0.06, 0.07])
    cov = np.diag([0.04, 0.09, 0.16])
    optimizer = RelaxedRiskParity(mu, cov)

    result = optimizer.solve(lambda_reg=0.0, return_target=None)

    weights = result.weights
    assert pytest.approx(weights.sum(), abs=1e-8) == 1.0

    contributions = weights * result.marginal_risk
    assert np.all(contributions > 0)
    share = contributions / contributions.sum()
    np.testing.assert_allclose(share, np.full(3, 1.0 / 3.0), atol=1e-4)


def test_relaxed_risk_parity_meets_explicit_target_with_regulator():
    mu = np.array([0.08, 0.05, 0.03])
    cov = np.array(
        [
            [0.05, 0.01, 0.0],
            [0.01, 0.04, 0.0],
            [0.0, 0.0, 0.02],
        ]
    )
    optimizer = RelaxedRiskParity(mu, cov)
    target = 0.06

    result = optimizer.solve(lambda_reg=0.3, return_target=target)
    weights = result.weights
    assert pytest.approx(weights.sum(), abs=1e-8) == 1.0
    achieved = float(mu @ weights)

    assert achieved >= target - 1e-6
    assert result.rho >= 0.0
    assert result.psi >= result.gamma


def test_relaxed_risk_parity_clips_unreachable_target():
    mu = np.array([0.02, 0.015])
    cov = np.array(
        [
            [0.04, 0.01],
            [0.01, 0.03],
        ]
    )
    optimizer = RelaxedRiskParity(mu, cov)

    requested = 0.5  # unattainable
    result = optimizer.solve(lambda_reg=0.4, return_target=requested)

    assert result.max_return is not None
    assert result.target_return is not None
    assert result.target_return < requested
    assert result.target_return <= result.max_return

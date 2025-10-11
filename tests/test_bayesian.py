import numpy as np
import pytest

from pyvallocation.bayesian import _cholesky_pd


def test_cholesky_pd_escalates_jitter_until_positive_definite():
    nearly_singular = np.array(
        [[0.09, 0.12], [0.12, 0.16 - 5e-5]],
        dtype=float,
    )

    with pytest.warns(RuntimeWarning):
        chol = _cholesky_pd(nearly_singular, jitter=1e-8)

    reconstructed = chol @ chol.T
    eigvals = np.linalg.eigvalsh(reconstructed)

    assert np.all(eigvals > 0)
    assert np.allclose(chol, np.tril(chol))

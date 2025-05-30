import pytest
import numpy as np
from pyvallocation.utils.validation import (
    is_psd,
    ensure_psd_matrix,
    check_weights_sum_to_one,
    check_non_negativity
)

def test_is_psd_positive():
    A = np.array([[2, -1], [-1, 2]])
    assert is_psd(A)

def test_is_psd_non_symmetric_square():
    B = np.array([[1, 2], [3, 4]])
    assert not is_psd(B)

def test_is_psd_non_square():
    with pytest.raises(ValueError):
        is_psd(np.array([[1, 2, 3], [4, 5, 6]]))

def test_ensure_psd_matrix_already_psd():
    A = np.array([[2, -1], [-1, 2]])
    out = ensure_psd_matrix(A)
    np.testing.assert_allclose(out, A)

def test_ensure_psd_matrix_corrects_non_psd():
    C = np.array([[0, 1], [1, 0]])
    out = ensure_psd_matrix(C)
    assert is_psd(out)
    np.testing.assert_allclose(out, out.T)

def test_ensure_psd_matrix_non_square():
    with pytest.raises(ValueError):
        ensure_psd_matrix(np.array([[1, 2, 3], [4, 5, 6]]))

def test_check_weights_sum_to_one_true_false():
    assert check_weights_sum_to_one(np.array([0.5, 0.5]))
    assert not check_weights_sum_to_one(np.array([0.5, 0.4]))

def test_check_non_negativity_true_false():
    assert check_non_negativity(np.array([0.0, 1e-9]))
    assert not check_non_negativity(np.array([-1e-8, 0.0]))

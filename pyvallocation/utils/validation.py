import logging

import numpy as np
import numpy.linalg as la

logger = logging.getLogger(__name__)


def is_psd(matrix: np.ndarray, tolerance: float = 1e-8) -> bool:
    """Check if a matrix is positive semi-definite (PSD) within a tolerance."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be a square 2D NumPy array.")
    if not np.allclose(matrix, matrix.T, atol=tolerance):
        return False
    try:
        eigenvalues = la.eigvalsh(matrix)
        return np.all(eigenvalues >= -tolerance)
    except la.LinAlgError:
        return False


def ensure_psd_matrix(matrix: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """Ensure a matrix is PSD by adding jitter to the diagonal if necessary."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be a square 2D NumPy array.")
    matrix = (matrix + matrix.T) / 2.0
    if is_psd(matrix, tolerance=jitter):
        return matrix
    min_eig = 0
    try:
        eigenvalues = la.eigvalsh(matrix)
        min_eig = np.min(eigenvalues)
    except la.LinAlgError:
        pass
    k = max(0, -min_eig) + jitter
    psd_matrix = matrix + k * np.eye(matrix.shape[0])
    if not is_psd(psd_matrix, tolerance=jitter):
        logger.warning(
            f"Matrix still not PSD after initial jittering with k={k}. Attempting larger jitter."
        )
        psd_matrix = matrix + (np.abs(min_eig) + 1e-6) * np.eye(matrix.shape[0])
        if not is_psd(psd_matrix, tolerance=jitter):
            raise ValueError(
                "Failed to make matrix PSD even with increased jitter. Input matrix might be severely ill-conditioned."
            )
    return psd_matrix


def check_weights_sum_to_one(weights: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Check if weights sum approximately to one within a tolerance."""
    return np.isclose(np.sum(weights), 1.0, atol=tolerance)


def check_non_negativity(array: np.ndarray, tolerance: float = -1e-9) -> bool:
    """Check if all elements in the array are non-negative within a tolerance."""
    return np.all(array >= tolerance)

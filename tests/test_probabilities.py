import pytest
import numpy as np
from pyvallocation import probabilities

def test_generate_uniform_probabilities_basic():
    p = probabilities.generate_uniform_probabilities(4)
    assert isinstance(p, np.ndarray)
    assert p.shape == (4,)
    assert np.allclose(p, np.array([0.25, 0.25, 0.25, 0.25]))

def test_generate_uniform_probabilities_invalid():
    with pytest.raises(ValueError):
        probabilities.generate_uniform_probabilities(0)

def test_generate_exp_decay_probabilities_basic():
    p = probabilities.generate_exp_decay_probabilities(3, half_life=1)
    assert isinstance(p, np.ndarray)
    assert p.shape == (3,)
    # Highest weight for most recent observation
    assert p[-1] > p[0]
    assert np.isclose(np.sum(p), 1.0)

def test_generate_exp_decay_probabilities_invalid_half_life():
    with pytest.raises(ValueError):
        probabilities.generate_exp_decay_probabilities(3, half_life=0)

def test_silverman_bandwidth():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    h = probabilities.silverman_bandwidth(x)
    sigma = np.std(x, ddof=1)
    expected = 1.06 * sigma * len(x)**(-1/5)
    assert np.isclose(h, expected)

def test_generate_gaussian_kernel_probabilities_defaults():
    x = np.array([0.0, 1.0, 2.0])
    p = probabilities.generate_gaussian_kernel_probabilities(x)
    assert isinstance(p, np.ndarray)
    assert np.isclose(np.sum(p), 1.0)
    assert p[-1] == np.max(p)

def test_generate_gaussian_kernel_probabilities_with_params():
    x = np.array([0.0, 1.0, 2.0])
    v = np.array([2.0, 1.0, 0.0])
    h = 1.0
    x_T = 0.0
    p = probabilities.generate_gaussian_kernel_probabilities(x, v=v, h=h, x_T=x_T)
    w = np.exp(-((v - x_T)**2) / (2 * h**2))
    expected = w / np.sum(w)
    assert np.allclose(p, expected)

def test_compute_effective_number_scenarios():
    p = np.array([0.5, 0.5])
    ens = probabilities.compute_effective_number_scenarios(p)
    assert np.isclose(ens, 1 / (0.5**2 + 0.5**2))

import numpy as np
from snmfem.measures import mse

import numpy as np


def MSE_map(map1, map2):
    """
    Calculates the mean squared error between two 2D arrays. They have to have the same dimension.
    :map1: first array (np.array 2D)
    :map2: second array (np.array 2D)
    """
    tr_m1_m1 = np.einsum("ij,ij->", map1, map1)
    tr_m2_m2 = np.einsum("ij,ij->", map2, map2)
    tr_m1_m2 = np.trace(map1.T @ map2)
    return tr_m1_m1 - 2 * tr_m1_m2 + tr_m2_m2


def spectral_angle(v1, v2):
    """
    Calculates the angle between two spectra. They have to have the same dimension.
    :v1: first spectrum (np.array 1D)
    :v2: second spectrum (np.array 1D)
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / np.pi


def test_mse():

    a = np.random.randn(10, 34)
    b = np.random.randn(10, 34)

    np.testing.assert_allclose(mse(a, b), MSE_map(a, b))

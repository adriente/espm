import numpy as np
from snmfem.generate_weights import random_weights, laplacian_weights, two_sphere_weights

def test_generate_random_weights():
    shape_2D = [28, 36]
    n_phases = 5
    
    w = random_weights(shape_2D=shape_2D, n_phases=n_phases)
    
    assert(w.shape == (*shape_2D, n_phases))
    np.testing.assert_array_less(-1e-30, w)
    np.testing.assert_array_almost_equal(np.sum(w, axis=2), 1)

def test_generate_laplacian_weights():
    shape_2D = [28, 36]
    n_phases = 5
    
    w = laplacian_weights(shape_2D=shape_2D, n_phases=n_phases)
    
    assert(w.shape == (*shape_2D, n_phases))
    np.testing.assert_array_less(-1e-30, w)
    np.testing.assert_array_almost_equal(np.sum(w, axis=2), 1)
    
def test_generate_two_sphere():

    w = two_sphere_weights()
    
    assert(w.shape == (80, 80, 3))
    np.testing.assert_array_less(-1e-30, w)
    np.testing.assert_array_almost_equal(np.sum(w, axis=2), 1)

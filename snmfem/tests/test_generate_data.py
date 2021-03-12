import numpy as np
from snmfem import generate_data as gd
from snmfem.models import EDXS, Toy
from snmfem.conf import DB_PATH
from snmfem.generate_weights import two_sphere_weights
from snmfem.datasets import three_phases
import os

def test_generate():

    weights = two_sphere_weights()

    # list of densities which will give different total number of events per spectra
    densities = np.array([1.0, 1.33, 1.25])

    phases = three_phases()
    
    spim = gd.ArtificialSpim(phases, densities, weights)
    assert spim.phases.shape == (3, 1980)
    assert spim.weights.shape == (80,80,3)
    np.testing.assert_allclose(np.sum(spim.phases, axis=1), np.ones([3]))


    # Generates a noiseless version of the dataset
    N = 47
    spim.generate_spim_stochastic(N)

    D = spim.phases.T
    A = spim.flatten_weights()
    X = spim.flatten_gen_spim()
    np.testing.assert_allclose(D @ A, X)
    assert spim.generated_spim.shape == (80, 80, 1980)

    np.testing.assert_allclose(np.sum(spim.generated_spim, axis=2), np.ones([80, 80]))

    w = spim.densities
    Xdot = spim.flatten_Xdot()
    # n D W A
    np.testing.assert_allclose(N * D @ np.diag(w) @ A, Xdot)

    filename = "test.npz"
    spim.save(filename)

    dat = np.load(filename)
    X = dat["X"]
    Xdot = dat["Xdot"]
    phases = dat["phases"] 
    densities = dat["densities"]
    weights = dat["weights"]
    N = dat["N"]
    D = phases.T
    A = weights.T.reshape(phases.shape[0],X.shape[1]*X.shape[0])  
    w = densities
    Xdot = Xdot.T.reshape(X.shape[2],X.shape[1]*X.shape[0])

    np.testing.assert_allclose(N * D @ np.diag(w) @ A, Xdot)
    del dat

    os.remove(filename)



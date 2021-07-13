import numpy as np
from snmfem.datasets.generate_data import ArtificialSpim
from snmfem.models import EDXS
from snmfem.conf import DB_PATH, DATASETS_PATH
from snmfem.datasets.generate_weights import generate_weights, random_weights, laplacian_weights, spheres_weights
from snmfem.datasets.generate_EDXS_phases import generate_brem_params, generate_random_phases, DEFAULT_PARAMS, unique_elts
from pathlib import Path
import os

def test_generate():

    model_parameters  = {"params_dict" : {"Abs" : {
        "thickness" : 200e-7,
        "toa" : 22,
        "atomic_fraction" : True,
        "density" : 4.5
    }},
                         "db_name" : "default_xrays.json",
                         "e_offset" : 0.208,
                         "e_scale" : 0.01,
                         "e_size": 1980,
                         "width_slope" : 0.01,
                         "width_intercept" : 0.065,
                         "seed" : 1}
  

    g_parameters = {
            "elements_list" : [8,13,14,12,26,29,31,72,71,62,60,92,20],
                        "brstlg" : 1}

    phases_parameters =  [
        {"b0" : 4.3298e-09 , "b1" : 6.7732e-07, "elements_dict" :  {"8": 1.0, "12": 0.51, "14": 0.61, "13": 0.07, "20": 0.04, "62": 0.02,"26": 0.028, "60": 0.002, "71": 0.003, "72": 0.003, "29": 0.02}},
        {"b0" : 1.3298e-09 , "b1" : 7.7732e-07, "elements_dict" : {"8": 0.54, "26": 0.15, "12": 1.0, "29": 0.038,"92": 0.0052, "60": 0.004, "31": 0.03, "71": 0.003}},
        {"b0" : 5.3298e-09 , "b1" : 3.7732e-07, "elements_dict" : {"8": 1.0, "14": 0.12, "13": 0.18, "20": 0.47,"62": 0.04, "26": 0.004, "60": 0.008, "72": 0.004, "29": 0.01}} 
        ]

    # Generate the phases
    model = EDXS(**model_parameters)
    model.generate_g_matr(**g_parameters)
    model.generate_phases(phases_parameters)
    phases = model.phases
    G = model.G
    
    seed = 0
    n_phases = 3
    weights_parameters = {"weight_type": "sphere",
                            "shape_2D": [80, 80]}

    weights = generate_weights(**weights_parameters, n_phases=n_phases, seed=seed)

    # list of densities which will give different total number of events per spectra
    densities = np.array([1.0, 1.33, 1.25])
    
    spim = ArtificialSpim(phases, densities, weights, G=G)
    assert spim.phases.shape == (3, 1980)
    assert spim.weights.shape == (80,80,3)
    np.testing.assert_allclose(np.sum(spim.phases, axis=1), np.ones([3]))


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
    shape_2D = [80, 80]
    n_phases = 3
    
    w = spheres_weights(shape_2D=shape_2D, n_phases=n_phases)
    
    assert(w.shape == (80, 80, 3))
    np.testing.assert_array_less(-1e-30, w)
    np.testing.assert_array_almost_equal(np.sum(w, axis=2), 1)

def test_gen_EDXS () : 
    
    b_dict = generate_brem_params(42)
    assert b_dict["b0"] <= 1e-7 
    assert b_dict["b1"] <= 1e-7

    phases, dicts = generate_random_phases(n_phases=3,seed = 42)
    np.testing.assert_array_less(-1e-30, phases)
    model = EDXS(**DEFAULT_PARAMS)
    model.generate_phases(dicts)
    np.testing.assert_almost_equal(model.phases,phases)

    unique_list = unique_elts(dicts)
    assert len(unique_list) == len(set(unique_list))






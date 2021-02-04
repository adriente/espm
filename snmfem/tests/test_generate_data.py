import numpy as np
from snmfem import generate_data as gd
from snmfem import EDXS_model as em
from snmfem.conf import DB_PATH
import os

def test_generate():
    # To save the dataset
    filename = "test"

    abs_db_path = None
    # abs_db_path = "Data/wernisch_abs.json"
    abs_elt_dict = None

    # Continuum X-rays parameters
    # They were determine by fitting experimental data from 0.6 to 18 keV. 
    # Since low energies were incorporated, the model is only effective and not quantitative.
    brstlg_pars = {
        "c0": 0.094,
        "c1": 1417,
        "c2": 1e-6,
        "b0": 1.2,
        "b1": -0.06,
        "b2": 0.00683,
    }

    scale = 1

    # Average number of counts in one spectrum of the artificial data
    N = 10

    # Creation of the pure spectra of the different phases.
    phase1 = em.EDXS_Model(DB_PATH, abs_db_path, brstlg_pars)
    # Gaussians corresponding to elements
    phase1.generate_spectrum(
        {
            "8": 1.0,
            "12": 0.51,
            "14": 0.61,
            "13": 0.07,
            "20": 0.04,
            "62": 0.02,
            "26": 0.028,
            "60": 0.002,
            "71": 0.003,
            "72": 0.003,
            "29": 0.02,
        },
        scale,
    )

    phase2 = em.EDXS_Model(DB_PATH, abs_db_path, brstlg_pars)
    phase2.generate_spectrum(
        {
            "8": 0.54,
            "26": 0.15,
            "12": 1.0,
            "29": 0.038,
            "92": 0.0052,
            "60": 0.004,
            "31": 0.03,
            "71": 0.003,
        },
        scale,
    )

    phase3 = em.EDXS_Model(DB_PATH, abs_db_path, brstlg_pars)
    phase3.generate_spectrum(
        {
            "8": 1.0,
            "14": 0.12,
            "13": 0.18,
            "20": 0.47,
            "62": 0.04,
            "26": 0.004,
            "60": 0.008,
            "72": 0.004,
            "29": 0.01,
        },
        scale,
    )

    # Objects needed for the creation of data
    # list of spectra
    phases = np.array([phase1.spectrum, phase2.spectrum, phase3.spectrum])
    # list of densities which will give different total number of events per spectra
    densities = np.array([1.0, 1.33, 1.25])

    spim = gd.AritificialSpim(phases, densities, (20, 20))
    assert spim.phases.shape == (3, 1980)
    np.testing.assert_allclose(np.sum(spim.phases, axis=1), np.ones([3]))

    # We add two particles belonging to two different phases to the data
    spim.sphere((5, 10), 3.5, 3.5, 0.0, 0.5, 1)
    spim.sphere((15, 12), 3.5, 3.5, 0.0, 0.5, 2)

    # Generates a noiseless version of the dataset
    spim.generate_spim_stochastic(N)



    D = spim.phases.T
    A = spim.weights.reshape(-1, 3).T
    X = spim.generated_spim.reshape(-1, 1980).T
    np.testing.assert_allclose(D @ A, X)
    assert spim.generated_spim.shape == (20, 20, 1980)

    np.testing.assert_allclose(np.sum(spim.generated_spim, axis=2), np.ones([20, 20]))

    w = spim.densities
    Xdot = spim.continuous_spim.reshape(-1, 1980).T
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
    A = weights.reshape(-1, 3).T  
    w = densities
    Xdot = Xdot.reshape(-1, 1980).T

    np.testing.assert_allclose(N * D @ np.diag(w) @ A, Xdot)
    del dat

    os.remove(filename)



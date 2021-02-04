import snmfem.generate_data as gd
import snmfem.EDXS_model as em
import numpy as np
from snmfem.conf import DB_PATH, DATASETS_PATH
from pathlib import Path
from tqdm import tqdm


def generate_synthetic_dataset(seeds=[0], N=100):
    """Generate a synthetic dataset.
    
    # Input parameters
    seeds: (List[Int]) list of the seeds for the dataset
    N: (Int) Average number of counts in one spectrum of the artificial data
    """
    abs_db_path = None
    #abs_db_path = "Data/wernisch_abs.json"
    abs_elt_dict = None

    # Continuum X-rays parameters
    # They were determine by fitting experimental data from 0.6 to 18 keV. Since low energies 
    # were incorporated, the model is only effective and not quantitative.
    brstlg_pars = {"c0": 0.094, "c1": 1417, "c2": 1e-6,
                   "b0": 1.2, "b1": -0.06, "b2": 0.00683}

    scale = 1

    # Creation of the pure spectra of the different phases.
    phase1 = em.EDXS_Model(DB_PATH, abs_db_path, brstlg_pars)
    # Gaussians corresponding to elements
    phase1.generate_spectrum({"8": 1.0, "12": 0.51, "14": 0.61, "13": 0.07, "20": 0.04,
                              "62": 0.02, "26": 0.028, "60": 0.002, "71": 0.003, "72": 0.003, "29": 0.02}, scale)

    phase2 = em.EDXS_Model(DB_PATH, abs_db_path, brstlg_pars)
    phase2.generate_spectrum({"8": 0.54, "26": 0.15, "12": 1.0, "29": 0.038,
                              "92": 0.0052, "60": 0.004, "31": 0.03, "71": 0.003}, scale)

    phase3 = em.EDXS_Model(DB_PATH, abs_db_path, brstlg_pars)
    phase3.generate_spectrum({"8": 1.0, "14": 0.12, "13": 0.18, "20": 0.47,
                              "62": 0.04, "26": 0.004, "60": 0.008, "72": 0.004, "29": 0.01}, scale)

    # Objects needed for the creation of data
    # list of spectra
    phases = np.array([phase1.spectrum, phase2.spectrum, phase3.spectrum])
    # list of densities which will give different total number of events per spectra
    densities = np.array([1.0, 1.33, 1.25])

    spim = gd.AritificialSpim(phases, densities, (80, 80))

    spim.sphere((25, 30), 3.5, 3.5, 0.0, 0.5, 1)
    spim.sphere((55, 30), 3.5, 3.5, 0.0, 0.5, 2)

    for seed in tqdm(seeds):
        filename = DATASETS_PATH / \
            Path("aspim037_N{}_2ptcls_brstlg_seed{}".format(N, seed))
        spim.generate_spim_stochastic(N, seed=seed)
        spim.save(filename=filename)


if __name__ == "__main__":
    generate_synthetic_dataset(range(10))

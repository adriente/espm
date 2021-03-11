import snmfem.generate_data as gd
import snmfem.models as EDXS
import numpy as np
from snmfem.conf import DB_PATH, DATASETS_PATH
from pathlib import Path
from tqdm import tqdm
from snmfem.generate_weights import two_sphere_weights


def generate_three_phases():
    abs_db_path = None
    db_name = "simple_xrays_threshold.json"

    e_offset = 0.208
    e_size = 1980
    e_scale = 0.01

    # Continuum X-rays parameters
    # They were determine by fitting experimental data from 0.6 to 18 keV. Since low energies 
    # were incorporated, the model is only effective and not quantitative.
    brstlg_pars = {"c0": 0.094, "c1": 1417, "c2": 1e-6,
                   "b0": 1.2, "b1": -0.06, "b2": 0.00683}
    scale = 1
    e_scale = 0.01
    e_offset = 0.208
    e_size = 1980
    db_name = "simple_xrays_threshold.json"

    # Creation of the pure spectra of the different phases.
    phase1 = EDXS(e_offset, e_size, e_scale, params_dict = brstlg_pars,db_name = db_name,abs_db_path= abs_db_path)
    # Gaussians corresponding to elements
    phase1.generate_spectrum({"8": 1.0, "12": 0.51, "14": 0.61, "13": 0.07, "20": 0.04,
                              "62": 0.02, "26": 0.028, "60": 0.002, "71": 0.003, "72": 0.003, "29": 0.02}, scale)

    phase2 = EDXS(e_offset, e_size, e_scale, params_dict = brstlg_pars,db_name = db_name,abs_db_path= abs_db_path)
    phase2.generate_spectrum({"8": 0.54, "26": 0.15, "12": 1.0, "29": 0.038,
                              "92": 0.0052, "60": 0.004, "31": 0.03, "71": 0.003}, scale)

    phase3 = EDXS(e_offset, e_size, e_scale, params_dict = brstlg_pars,db_name = db_name,abs_db_path= abs_db_path)
    phase3.generate_spectrum({"8": 1.0, "14": 0.12, "13": 0.18, "20": 0.47,
                              "62": 0.04, "26": 0.004, "60": 0.008, "72": 0.004, "29": 0.01}, scale)
    
    phases = np.array([phase1.spectrum, phase2.spectrum, phase3.spectrum])
    return phases

def generate_edxs_dataset(seeds=[0], N=100):
    """Generate a synthetic dataset.
    
    # Input parameters
    seeds: (List[Int]) list of the seeds for the dataset
    N: (Int) Average number of counts in one spectrum of the artificial data
    """

    phases = generate_three_phases()
    
    # Objects needed for the creation of data
    # list of spectra

    # list of densities which will give different total number of events per spectra
    densities = np.array([1.0, 1.33, 1.25])
    for seed in tqdm(seeds):
        weights = two_sphere_weights(seed)
        
        spim = gd.ArtificialSpim(phases, densities, weights)

        
        folder = DATASETS_PATH / Path("aspim037_N{}_2ptcls_brstlg".format(N))
        folder.mkdir(exist_ok=True, parents=True)

    
        filename = folder / Path("sample_{}".format(seed))
        spim.generate_spim_stochastic(N, seed=seed)
        spim.save(filename=filename)
        
        


# def generate_toy_dataset(seeds=[0], N=100):
#     """Generate a synthetic dataset.
    
#     # Input parameters
#     seeds: (List[Int]) list of the seeds for the dataset
#     N: (Int) Average number of counts in one spectrum of the artificial data
#     """


#     # Objects needed for the creation of data
#     # list of spectra
#     phases = np.array([phase1.spectrum, phase2.spectrum, phase3.spectrum])
#     # list of densities which will give different total number of events per spectra
#     densities = np.array([1.0, 1.33, 1.25])
#     for seed in tqdm(seeds):
#         spim = gd.ArtificialSpim(phases, densities, (80, 80))

#         if seed == 0:
#             spim.sphere((25, 30), 3.5, 3.5, 0.0, 0.5, 1)
#             spim.sphere((55, 30), 3.5, 3.5, 0.0, 0.5, 2)
#         else:
#             np.random.seed(seed)
#             for _ in range(2):
#                 p = np.random.randint(1, 80, [2])
#                 spim.sphere(p, 3.5, 3.5, 0.0, 0.5, 1)            

        
#         folder = DATASETS_PATH / Path("aspim037_N{}_2ptcls_brstlg".format(N))
#         folder.mkdir(exist_ok=True, parents=True)

    
#         filename = folder / Path("sample_{}".format(seed))
#         spim.generate_spim_stochastic(N, seed=seed)
#         spim.save(filename=filename)

import snmfem.generate_data as gd
from snmfem import models
import numpy as np
from snmfem.conf import DB_PATH, DATASETS_PATH
from pathlib import Path
from tqdm import tqdm
from snmfem.generate_weights import generate_weights
from snmfem.conf import SCRIPT_CONFIG_PATH
import json


# def three_phases():
#     abs_db_path = None
#     db_name = "simple_xrays_threshold.json"

#     e_offset = 0.208
#     e_size = 1980
#     e_scale = 0.01

#     # Continuum X-rays parameters
#     # They were determine by fitting experimental data from 0.6 to 18 keV. Since low energies 
#     # were incorporated, the model is only effective and not quantitative.
#     brstlg_pars = {"c0": 0.094, "c1": 1417, "c2": 1e-6,
#                    "b0": 1.2, "b1": -0.06, "b2": 0.00683}
#     scale = 1
#     e_scale = 0.01
#     e_offset = 0.208
#     e_size = 1980
#     db_name = "simple_xrays_threshold.json"

#     # Creation of the pure spectra of the different phases.
#     phase1 = EDXS(e_offset, e_size, e_scale, params_dict = brstlg_pars,db_name = db_name,abs_db_path= abs_db_path)
#     # Gaussians corresponding to elements
#     phase1.generate_spectrum({"8": 1.0, "12": 0.51, "14": 0.61, "13": 0.07, "20": 0.04,
#                               "62": 0.02, "26": 0.028, "60": 0.002, "71": 0.003, "72": 0.003, "29": 0.02}, scale)

#     phase2 = EDXS(e_offset, e_size, e_scale, params_dict = brstlg_pars,db_name = db_name,abs_db_path= abs_db_path)
#     phase2.generate_spectrum({"8": 0.54, "26": 0.15, "12": 1.0, "29": 0.038,
#                               "92": 0.0052, "60": 0.004, "31": 0.03, "71": 0.003}, scale)

#     phase3 = EDXS(e_offset, e_size, e_scale, params_dict = brstlg_pars,db_name = db_name,abs_db_path= abs_db_path)
#     phase3.generate_spectrum({"8": 1.0, "14": 0.12, "13": 0.18, "20": 0.47,
#                               "62": 0.04, "26": 0.004, "60": 0.008, "72": 0.004, "29": 0.01}, scale)
    
#     phases = np.array([phase1.spectrum, phase2.spectrum, phase3.spectrum])
#     return phases

# def generate_edxs_dataset(folder=None, seeds=[0], N=100):
#     """Generate a synthetic dataset.
    
#     # Input parameters
#     seeds: (List[Int]) list of the seeds for the dataset
#     N: (Int) Average number of counts in one spectrum of the artificial data
#     """

#     phases = three_phases()
    
#     # Objects needed for the creation of data
#     # list of spectra
#     if folder is None:
#         folder_name = "aspim037_N{}_2ptcls_brstlg".format(N)
#         folder = DATASETS_PATH / Path(folder_name)
#     folder.mkdir(exist_ok=True, parents=True)

#     # list of densities which will give different total number of events per spectra
#     densities = np.array([1.0, 1.33, 1.25])
#     for seed in tqdm(seeds):
#         weights = two_sphere_weights(seed)
        
#         spim = gd.ArtificialSpim(phases, densities, weights)

#         filename = folder / Path("sample_{}".format(seed))
#         spim.generate_spim_stochastic(N, seed=seed)
#         spim.save(filename=filename)
        

# def toy_phases(e_size = 50,  k=5, seed=0, 
#     e_offset = 0
#     e_scale = 1
#     pars_dict = {"c" : 25, "k" : k}):
    
#     toy = Toy(e_offset=e_offset, e_size=e_size, e_scale=e_scale, params_dict=pars_dict, seed = seed)
    
#     toy.generate_g_matr()
#     # toy.generate_spectrum()
#     toy.generate_phases()
#     phases = toy.phases

#     return phases

# def generate_toy_dataset(folder=None, seeds=[0], shape_2D = (15, 15), k = 5, N = 200, laplacian=True):
#     """Generate a toy dataset.
    
#     # Input parameters
#     folder_name: (str) 
#     seeds: (List[Int])
#     shape_2D: ([Int, Int]) shape of the image
#     N: (Int) Average number of counts in one spectrum of the artificial data
#     laplacian: (bool) use laplacian instead of random image
#     """

#     phases = toy_phases(k=k)
#     densities = k*[1.0]
    
#     if folder is None:
#         if laplacian:
#             folder_name = "Toy_{}_N{}".format("laplacian", N)
#         else: 
#             folder_name = "Toy_{}_N{}".format("random", N)
#         folder = DATASETS_PATH / Path(folder_name)
#     folder.mkdir(exist_ok=True, parents=True)
    
#     # list of densities which will give different total number of events per spectra
#     for seed in tqdm(seeds):
#         if laplacian:
#             weights = laplacian_weights(shape_2D=shape_2D, n_phases=k, seed=seed)
#         else:
#             weights = random_weights(shape_2D=shape_2D, n_phases=k, seed=seed)
                
#         spim = gd.ArtificialSpim(phases, densities, weights)
    
#         filename = folder / Path("sample_{}".format(seed))
#         spim.generate_spim_stochastic(N, seed=seed)
#         spim.save(filename=filename)
        

def generate_dataset_from_json(json_file):
    """Generate a toy dataset.
    
    # Input parameters
    folder_name: (str) 
    seeds: (List[Int])
    shape_2D: ([Int, Int]) shape of the image
    N: (Int) Average number of counts in one spectrum of the artificial data
    laplacian: (bool) use laplacian instead of random image
    """

    json_path = SCRIPT_CONFIG_PATH / Path(json_file)

    with open(json_path,"r") as f :
        json_dict = json.load(f)

    generate_dataset(**json_dict)


def generate_dataset(**kwargs):
    
    # Handle paramters
    data_folder = kwargs["data_folder"]
    weights_parameters = kwargs["weights_parameters"]
    N = kwargs["N"]
    model_parameters = kwargs["model_parameters"]
    densities = kwargs["densities"]
    g_parameters = kwargs["g_parameters"]
    phases_parameters = kwargs["phases_parameters"]

    seeds = range(kwargs["seeds_range"])
    Model = getattr(models, kwargs["model"]) 
    n_phases = len(phases_parameters)

    # Generate the phases
    model = Model(**model_parameters)
    model.generate_g_matr(**g_parameters)
    model.generate_phases(phases_parameters)
    phases = model.phases
    G = model.G

    # Ensure the data folder exist
    folder = DATASETS_PATH / Path(data_folder)
    folder.mkdir(exist_ok=True, parents=True)
    
    # list of densities which will give different total number of events per spectra
    for seed in tqdm(seeds):

        weights = generate_weights(**weights_parameters, n_phases=n_phases, seed=seed)
        spim = gd.ArtificialSpim(phases, densities, weights)
        filename = folder / Path("sample_{}".format(seed))
        spim.generate_spim_stochastic(N, seed=seed)
        spim.save(filename=filename)

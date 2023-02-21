from espm.utils import arg_helper
from espm.datasets.base import generate_dataset
from espm.datasets.generate_EDXS_phases import generate_modular_phases
import hyperspy.api as hs
import numpy as np
from espm.datasets.base import generate_spim, save_generated_spim
from espm.datasets.generate_weights import chemical_maps_weights
from espm.conf import DATASETS_PATH, BASE_PATH, DEFAULT_SYNTHETIC_DATA_DICT
from pathlib import Path

n_phases = 3
seed = 43

file_path = BASE_PATH.parent / Path("experiments/71GPa_experimental_data.hspy")


data = hs.load(str(file_path))

weights = chemical_maps_weights(file_path,["Fe_Ka","Ca_Ka"],conc_max = 0.7)

# elts_dicts = [
#     {
#         "Mg" : 0.522, "Fe" : 0.104, "O" : 0.374, "Cu" : 0.05
#     },
#     {
#         "Mg" : 0.020, "Fe" : 0.018, "Ca" : 0.188, "Si" : 0.173, "Al" : 0.010, "O" : 0.572, "Ti" : 0.004, "Cu" : 0.05, "Sm" : 0.007, "Lu" : 0.006, "Nd" : 0.006 
#     },
#     {
#         "Mg" : 0.245, "Fe" : 0.035, "Ca" : 0.031, "Si" : 0.219, "Al" : 0.024, "O" : 0.436, "Cu" : 0.05, "Hf" : 0.01
#     }]

elts_dicts = [
    {
        "Mg" : 0.522, "Fe" : 0.104, "O" : 0.374, "Cu" : 0.05
    },
    {
        "Mg" : 0.020, "Fe" : 0.018, "Ca" : 0.188, "Si" : 0.173, "Al" : 0.010, "O" : 0.572, "Ti" : 0.004, "Cu" : 0.05, "Sm" : 0.007, "Lu" : 0.006, "Nd" : 0.006 
    },
    {
        "Mg" : 0.445, "Fe" : 0.035, "Ca" : 0.031, "Si" : 0.419, "Al" : 0.074, "O" : 1.136, "Cu" : 0.05, "Hf" : 0.01
    }]

brstlg_pars = [
    {"b0" : 0.0001629, "b1" : 0.0009812},
    {"b0" : 0.0007853, "b1" : 0.0003658},
    {"b0" : 0.0003458, "b1" : 0.0006268}
]

model_params = {
        "e_offset" : 0.3,
        "e_size" : 1980,
        "e_scale" : 0.01,
        "width_slope" : 0.01,
        "width_intercept" : 0.065,
        "db_name" : "default_xrays.json",
        "E0" : 200,
        "params_dict" : {
            "Abs" : {
                "thickness" : 100.0e-7,
                "toa" : 35,
                "density" : 4.5,
                "atomic_fraction" : False
            },
            "Det" : "SDD_efficiency.txt"
        }
    }

data_dict = {
    "N" : 176,
    "densities" : [1.2,1.0,0.8],
    "data_folder" : "71GPa_synthetic_N176",
    "model" : "EDXS",
    "seed" : seed
}


phases, full_dict = generate_modular_phases(elts_dicts=elts_dicts, brstlg_pars = brstlg_pars, scales = [1, 1, 1], model_params= model_params, seed = seed)

data_dict.update(full_dict)

input_dict = arg_helper(data_dict,DEFAULT_SYNTHETIC_DATA_DICT)


generate_dataset(seeds_range=5, **input_dict, weights=weights)


# spim = generate_spim(phases, weights, [1.2,1.0,0.8], N = 176, seed=seed,continuous = False)
# filename = DATASETS_PATH / Path("71GPa_synthetic_N176.hspy")
# save_generated_spim(filename, spim, model_params, full_dict["phases_parameters"], data_dict)

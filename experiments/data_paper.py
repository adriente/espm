from espm.utils import arg_helper
from espm.datasets.base import generate_dataset, generate_weights
from espm.datasets.generate_EDXS_phases import generate_modular_phases
from espm.conf import DEFAULT_SYNTHETIC_DATA_DICT
import numpy as np

n_phases = 3
seed = 43

weights_dict = {
    "weight_type" : "sphere",
    "shape_2d" : [128,128],
    "weights_params" : {"radius" : 4.0}
}

elts_dicts = [
    {
        "Mg" : 0.245, "Fe" : 0.035, "Ca" : 0.031, "Si" : 0.219, "Al" : 0.024, "O" : 0.436, "Cu" : 0.05, "Hf" : 0.01
    },
    {
        "Mg" : 0.522, "Fe" : 0.104, "O" : 0.374, "Cu" : 0.05
    },
    {
        "Mg" : 0.020, "Fe" : 0.018, "Ca" : 0.188, "Si" : 0.173, "Al" : 0.010, "O" : 0.572, "Ti" : 0.004, "Cu" : 0.05, "Sm" : 0.007, "Lu" : 0.006, "Nd" : 0.006 
    }]

brstlg_pars = [
    {"b0" : 0.0003458, "b1" : 0.0006268},
    {"b0" : 0.0001629, "b1" : 0.0009812},
    {"b0" : 0.0007853, "b1" : 0.0003658}
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

phases, full_dict = generate_modular_phases(elts_dicts=elts_dicts, brstlg_pars = brstlg_pars, scales = [1, 1, 1], model_params= model_params, seed = seed)

data_dict = {
    "N" : 18,
    "densities" : [1.0,0.8,1.2],
    "data_folder" : "FpBrgCaPv_N18_paper",
    "seed" : seed
}

data_dict.update(weights_dict)
data_dict.update(full_dict)

input_dict = arg_helper(data_dict,DEFAULT_SYNTHETIC_DATA_DICT)

print(input_dict)

generate_dataset(**input_dict, seeds_range=7)

data_dict["N"] = 73
data_dict["data_folder"] = "FpBrgCaPv_N73_paper"

input_dict = arg_helper(data_dict,DEFAULT_SYNTHETIC_DATA_DICT)


generate_dataset(**input_dict, seeds_range=7)

data_dict["N"] = 293
data_dict["data_folder"] = "FpBrgCaPv_N293_paper"

input_dict = arg_helper(data_dict,DEFAULT_SYNTHETIC_DATA_DICT)

generate_dataset(**input_dict, seeds_range=7)

##########
# Metals #
##########

# n_phases = 3
# seed = 400

# weights_dict = {
#     "weight_type" : "sphere",
#     "shape_2d" : [128,128],
#     "weights_params" : {"radius" : 4.0}
# }

# elts_dicts = [
#     {
#         "Hf" : 0.245, "Ti" : 0.34, "Al" : 0.275, "Si" : 0.01, "Cu" : 0.1, "O" : 0.03
#     },
#     {
#         "W" : 0.522, "Co" : 0.104, "Ni" : 0.194, "Cu" : 0.1, "Si" : 0.03, "O" : 0.05
#     },
#     {
#         "Ti" : 0.138, "Hf" : 0.312, "W" : 0.188, "Ta" : 0.232, "Si" : 0.02, "O" : 0.01, "Cu" : 0.1
#     }]

# brstlg_pars = [
#     {"b0" : 0.0006458, "b1" : 0.0002268},
#     {"b0" : 0.0003129, "b1" : 0.0009812},
#     {"b0" : 0.0005353, "b1" : 0.0004658}
# ]

# model_params = {
#         "e_offset" : 0.3,
#         "e_size" : 1980,
#         "e_scale" : 0.01,
#         "width_slope" : 0.01,
#         "width_intercept" : 0.065,
#         "db_name" : "default_xrays.json",
#         "E0" : 200,
#         "params_dict" : {
#             "Abs" : {
#                 "thickness" : 100.0e-7,
#                 "toa" : 35,
#                 "density" : 6.5,
#                 "atomic_fraction" : False
#             },
#             "Det" : "SDD_efficiency.txt"
#         }
#     }

# phases, full_dict = generate_modular_phases(elts_dicts=elts_dicts, brstlg_pars = brstlg_pars, scales = [1, 1, 1], model_params= model_params, seed = seed)

# data_dict = {
#     "N" : 18,
#     "densities" : [1.1,0.9,1.4],
#     "data_folder" : "Nisuperalloy_N18_paper",
#     "seed" : seed
# }

# data_dict.update(weights_dict)
# data_dict.update(full_dict)

# input_dict = arg_helper(data_dict,DEFAULT_SYNTHETIC_DATA_DICT)

# generate_dataset(**input_dict, seeds_range=7)

# data_dict["N"] = 73
# data_dict["data_folder"] = "Nisuperalloy_N73_paper"

# input_dict = arg_helper(data_dict,DEFAULT_SYNTHETIC_DATA_DICT)

# generate_dataset(**input_dict, seeds_range=7)

# data_dict["N"] = 293
# data_dict["data_folder"] = "Nisuperalloy_N293_paper"

# input_dict = arg_helper(data_dict,DEFAULT_SYNTHETIC_DATA_DICT)

# generate_dataset(**input_dict, seeds_range=7)
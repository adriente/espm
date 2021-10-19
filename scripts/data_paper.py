from snmfem.utils import arg_helper
from snmfem.datasets.base import generate_dataset
from snmfem.datasets.generate_EDXS_phases import generate_modular_phases
from snmfem.conf import DEFAULT_SYNTHETIC_DATA_DICT

n_phases = 3
seed = 91

weights_dict = {
    "weight_type" : "sphere",
    "shape_2D" : [80,80],
}

elts_dicts = [
    {
        "Mg" : 0.245, "Fe" : 0.035, "Ca" : 0.031, "Si" : 0.219, "Al" : 0.024, "O" : 0.446, "Cu" : 0.05
    },
    {
        "Mg" : 0.522, "Fe" : 0.104, "O" : 0.374, "Cu" : 0.05
    },
    {
        "Mg" : 0.020, "Fe" : 0.018, "Ca" : 0.188, "Si" : 0.173, "Al" : 0.010, "O" : 0.591, "Ti" : 0.004, "Cu" : 0.05
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
    "N" : 30,
    "densities" : [1.0,0.8,1.2],
    "data_folder" : "FpBrgCaPv_N30_paper",
    "seed" : seed
}

data_dict.update(weights_dict)
data_dict.update(full_dict)

input_dict = arg_helper(data_dict,DEFAULT_SYNTHETIC_DATA_DICT)

generate_dataset(**input_dict, seeds_range=10)
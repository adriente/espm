from espm.datasets.base import generate_dataset
from espm.conf import DATASETS_PATH
from pathlib import Path
import hyperspy.api as hs
import os

dataset_particles  = {'N': 500,
    'seed' : 91,
    'densities': [0.6030107883539217, 0.9870613994765459, 0.8894990661032164],
    'data_folder': 'built_in_particules',
    'weight_type': 'sphere',
    'weights_params' : {
      "radius" : 2.5
    },
    'shape_2d': [80, 80],
    'phases_parameters': [{'b0': 7.408513360414626e-05,
    'b1': 6.606903143185911e-03,
    'elements_dict': {'23': 0.04704309583693933,
    '37': 0.914954275584854,
    '74': 0.06834271611694454},
    'scale': 1.0},
    {'b0': 7.317975736931391e-05,
      'b1': 5.2126092148294355e-03,
      'elements_dict': {'7': 0.4517936299777999,
      '70': 0.39973013314240835,
      '78': 0.08298592142537742},
      'scale': 1.0},
    {'b0': 2.2664567599307173e-05,
      'b1': 3.1627208027847766e-03,
      'elements_dict': {'13': 0.43306626599937914,
      '22': 0.3985896640183708,
      '57': 0.8994030840372912},
      'scale': 1.0}],
    'model_parameters': {'e_offset': 0.2,
    'e_size': 1980,
    'e_scale': 0.01,
    'width_slope': 0.01,
    'width_intercept': 0.065,
    'db_name': 'default_xrays.json',
    'E0': 200,
    'params_dict': {'Abs': {'thickness': 1e-05,
    'toa': 22,
    'density': None,
    'atomic_fraction': False},
    'Det': 'SDD_efficiency.txt'}},
    'shape_2d': (80, 80),
    'model': 'EDXS'}

dataset_grain_boundary = {
    "model_parameters" : {
        "e_offset" : 1.27,
        "e_size" : 3746,
        "e_scale" : 0.005,
        "width_slope" : 0.01,
        "width_intercept" : 0.065,
        "db_name" : "default_xrays.json",
        "E0" : 200,
        "params_dict" : {
            "Abs" : {
                "thickness" : 140.0e-7,
                "toa" : 22,
                "density" : 3.124,
                "atomic_fraction" : False
            },
            "Det" : "SDD_efficiency.txt"
        }
    },
    "N" : 15,
    "densities" : [1,1],
    "data_folder" : "built_in_grain_boundary",
    "seed" : 0,
    "weight_type" : "gaussian_ripple",
    "shape_2d" : (100,400),
    "weights_params" : {
        "width" : 5
    },
    "model" : "EDXS",
    "phases_parameters" : [{"b0" : 5.5367e-4,
                            "b1" : 0.0192181,
                            "scale" : 0.05,
                            "elements_dict" : {"Ca" : 0.54860348,
                                      "P" : 0.38286879,
                                      "Sr" : 0.03166235,
                                      "Cu" : 0.03686538}},
                            {"b0" : 5.5367e-4,
                            "b1" : 0.0192181,
                            "scale" : 0.05,
                            "elements_dict" : {"Ca" : 0.54860348,
                                      "P" : 0.38286879,
                                      "Sr" : 0.12166235,
                                      "Cu" : 0.03686538}}]
}

def generate_built_in_datasets (seeds_range = 2) : 
    if not(os.path.isdir(DATASETS_PATH / Path(dataset_particles["data_folder"]))) : 
        print("Generating 2 particles + one matrix built-in dataset. This will take a minute.")
        generate_dataset(**dataset_particles,seeds_range=seeds_range)
    if not(os.path.isdir(DATASETS_PATH / Path(dataset_grain_boundary["data_folder"]))) :
        print("Generating a grain boundary with Sr segregation. This will take a minute.")
        generate_dataset(**dataset_grain_boundary, seeds_range = seeds_range)

def load_particules (sample = 0) : 
    filename = DATASETS_PATH / Path("{}/sample_{}.hspy".format(dataset_particles["data_folder"],sample))
    spim = hs.load(filename)
    return spim

def load_grain_boundary (sample = 0) :
    filename = DATASETS_PATH / Path("{}/sample_{}.hspy".format(dataset_grain_boundary["data_folder"],sample))
    spim = hs.load(filename)
    return spim
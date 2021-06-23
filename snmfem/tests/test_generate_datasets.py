from snmfem.datasets import generate_dataset
from snmfem.conf import DATASETS_PATH
from pathlib import Path
import shutil


def test_generate_edxs_dataset():
    
    params = {
    "seeds_range" : 3,
    "N": 30,
    "model" : "EDXS",
    "weights_parameters" : {"weight_type": "sphere",
                            "shape_2D": [20, 15]},
    "model_parameters" :{"params_dict" : {"c0" : 4.8935e-05, 
                                          "c1" : 1464.19810,
                                          "c2" : 0.04216872,
                                          "b0" : 0.15910789,
                                          "b1" : -0.00773158,
                                          "b2" : 8.7417e-04},
                         "db_name" : "default_xrays.json",
                         "e_offset" : 0.208,
                         "e_scale" : 0.01,
                         "e_size": 1980,
                         "width_slope" : 0.01,
                         "width_intercept" : 0.065,
                         "seed" : 1},
    "densities" : [1.0, 1.33, 1.25], 

    "g_parameters" : {"thickness": 2e-05,
            "density": 3.5,
            "toa": 22,
            "elements_list" : [8,13,14,12,26,29,31,72,71,62,60,92,20],
                        "brstlg" : 1},
    "phases_parameters" : [
        {"thickness": 2e-05,
            "density": 3.5,
            "toa": 22,
            "atomic_fraction": True,
            "elements_dict":{"8": 1.0 , "12": 0.51  , "14": 0.61  , "13": 0.07  , "20": 0.04  , "62": 0.02  ,
                            "26": 0.028  , "60": 0.002  , "71": 0.003  , "72": 0.003  , "29": 0.02  }, 
            "scale" : 0.01},
        {"thickness": 2e-05,
            "density": 3.5,
            "toa": 22,
            "atomic_fraction": True,
            "elements_dict":{"8": 0.54  , "26": 0.15  , "12": 1.0  , "29": 0.038  ,
                            "92": 0.0052  , "60": 0.004  , "31": 0.03  , "71": 0.003  },
            "scale" : 0.01},   
            {"thickness": 2e-05,
            "density": 3.5,
            "toa": 22,
            "atomic_fraction": True,
            "elements_dict":{"8": 1.0  , "14": 0.12  , "13": 0.18  , "20": 0.47  ,
                            "62": 0.04  , "26": 0.004  , "60": 0.008  , "72": 0.004  , "29": 0.01  }, 
            "scale" : 0.01} 
        ],
    "data_folder" : "test"
    }


    folder = DATASETS_PATH / Path(params["data_folder"])
    generate_dataset(**params)
    shutil.rmtree(folder)

def test_generate_toy_dataset():
    params = {
        "seeds_range" : 4,
        "weights_parameters" : {"weight_type": "laplacian",
                                "shape_2D": [14, 23]},
        "N": 35,
        "model" : "Toy",
        "model_parameters" :{"params_dict" : {"c" : 25, 
                                            "k" : 4},
                            "db_name" : "default_xrays.json",
                            "e_offset" : 0.208,
                            "e_scale" : 0.01,
                            "e_size": 50,
                            "seed" : 1},
        "densities" : [1,1,1,1], 
        "g_parameters" : {"elements_list" : [8,13,14,12,26,29,31,72,71,62,60,92,20],
                        "brstlg" : 1},
        "phases_parameters" : [{}, {}, {}, {}],
        "data_folder" : "test"
    }
    folder = DATASETS_PATH / Path(params["data_folder"])
    generate_dataset(**params)    
    shutil.rmtree(folder)

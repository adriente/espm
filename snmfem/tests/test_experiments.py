import snmfem.experiments as exps
from pathlib import Path
from snmfem.conf import DATASETS_PATH, BASE_PATH
from snmfem.datasets import generate_dataset
import shutil
import json
import os
import numpy as np

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

base_path = BASE_PATH / Path("tests/ressources/") 
folder = base_path / Path(params["data_folder"])
script_file = Path("test.json")
    
def test_generate_dataset():
    with open(base_path / script_file,"w") as f :
        json.dump(params,f)
    generate_dataset(base_path=base_path , **params)
    
def test_load_samples() : 
    samples , k  = exps.load_samples(script_file, base_path_conf=base_path, base_path_dataset=base_path)
    assert(k == params["model_parameters"]["params_dict"]["k"])
    assert(len(samples) == params["seeds_range"])

def test_load_data() : 
    samples , k  = exps.load_samples(script_file, base_path_conf=base_path, base_path_dataset=base_path)
    for r in range(params["seeds_range"]):
        Xflat, true_spectra_flat, true_maps_flat, G, shape_2d = exps.load_data(samples[r])
<<<<<<< HEAD
        
=======
        # The following line does not work because the order is not the same!
>>>>>>> 68d0509cf138f50988af87ed57b8eb24faa8176f
        data_path = folder / Path("sample_{}.npz".format(r))
        data = np.load(data_path)
        shape_2d_test = tuple(params["weights_parameters"]["shape_2D"])
        Xflat_test = (data["X"].reshape(shape_2d_test[0]*shape_2d_test[1],params["model_parameters"]["e_size"])).T

        true_spectra_test = data["phases"]*params["N"]*np.array(params["densities"])[:,np.newaxis]
        true_maps_test = (data["weights"].reshape(shape_2d_test[0]*shape_2d_test[1],k)).T
<<<<<<< HEAD
        
=======
>>>>>>> 68d0509cf138f50988af87ed57b8eb24faa8176f
        assert(tuple(shape_2d) == shape_2d_test)
        np.testing.assert_array_equal(Xflat, Xflat_test)
        np.testing.assert_array_equal(data["G"],G)
        np.testing.assert_array_equal(true_spectra_test,true_spectra_flat)
        np.testing.assert_array_equal(true_maps_test,true_maps_flat)

def test_run_experiments () :
    samples , k  = exps.load_samples(script_file, base_path_conf=base_path, base_path_dataset=base_path)
    r = np.random.randint(params["seeds_range"])
    Xflat, true_spectra_flat, true_maps_flat, G, shape_2d = exps.load_data(samples[r])
    default_params = {
    "n_components" : k,
    "tol" : 1e-3,
    "max_iter" : 10000,
    "init" : "random",
    "random_state" : 1,
    "verbose" : 0
    }

    params_snmf = {
        "force_simplex" : True,
        "skip_G" : False,
        "mu": 0
    }

    params_evalution = {
        "u" : True,
    }
    experiment = {"name": "snmfem smooth 3", "method": "SmoothNMF", "params": {**default_params, **params_snmf, "lambda_L" : 3.0}}
    m, (GP, A), loss = exps.run_experiment(Xflat, true_spectra_flat, true_maps_flat, G, experiment, params_evalution, shape_2d = shape_2d)
    
    assert(len(m) == 3)
    assert(len(m[0])==k)
    assert(np.array(m[0]).dtype=="float64")
    assert(np.array(m[2]).dtype=="int64")

    assert(GP.shape == (params["model_parameters"]["e_size"],k))
    assert(A.shape == (k,shape_2d[0]*shape_2d[1]))
    
    l_loss = 0
    for i in loss.dtype.names : 
        l_loss += 1
    assert(l_loss == 7 + 2*k)
    
def test_delete_dataset():
    shutil.rmtree(folder)
    os.remove(base_path / script_file)





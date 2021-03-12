import json
import sys
import numpy as np
from pathlib import Path
from snmfem import estimators
from snmfem import models 
from snmfem.conf import SCRIPT_CONFIG_PATH, DATASETS_PATH, RESULTS_PATH
import snmfem.measures as measures


def run_experiment(json_file,save=False,data_filename = None) : 
    json_path = SCRIPT_CONFIG_PATH / Path(json_file)

    with open(json_path,"r") as f :
        json_dict = json.load(f)

    if data_filename is None : 
        data_file = DATASETS_PATH / Path(json_dict["data_file"])
    else : 
        data_file = DATASETS_PATH / Path(data_filename)
    data = np.load(data_file)
    X = data["X_flat"]
    true_spectra = data["densities"][:,np.newaxis]*data["phases"]

    true_maps = data["flat_weights"]
    Model = getattr(models, json_dict["model"]) 
    model = Model(**json_dict["model_parameters"])
    model.generate_g_matr(**json_dict["g_parameters"])

    G = model.G
    Estimator = getattr(estimators, json_dict["estimator"]) 

    estimator = Estimator(**json_dict["hyperparameters"])
    estimator.fit(X,G=G)

    d = {}  # dictionary of everything we would like to save
    d["G"] = estimator.G_
    d["P"] = estimator.P_
    d["A"] = estimator.A_
    save_data_path = RESULTS_PATH / Path("data/" + json_dict["filename"])
    np.savez(save_data_path, **d)

    angles = measures.find_min_angle(true_spectra,(estimator.G_@estimator.P_).T)
    mse = measures.find_min_MSE(true_maps,estimator.A_)
    if save :
        save_dict = {}
        save_dict["inputs"] = json_dict
        save_dict["results"] = {"angles" : angles,"mse" : mse}
        save_meta_path = RESULTS_PATH / Path("summaries/" + json_dict["filename"]+".json")

        with open(save_meta_path, 'w') as outfile:
            json.dump(save_dict, outfile, sort_keys=True, indent=4)
    else : 
        return angles, mse


if __name__ == "__main__" : 
    json_file = sys.argv[1]
    run_experiment(json_file,save=True)
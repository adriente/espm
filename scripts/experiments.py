import json
import sys
import numpy as np
from pathlib import Path

from snmfem import conf
from snmfem.estimator import nmf
import snmfem.models.EDXS_model as em
import snmfem.models.toy_model as toy
import snmfem.measures as measures

if __name__ == "__main__" : 
    FUNC_MAP = {"ToyModel" : toy.ToyModel, "EDXS_Model" : em.EDXS_Model}

    json_file = sys.argv[1]
    json_path = conf.SCRIPT_CONFIG_PATH / Path(json_file)

    with open(json_path,"r") as f :
        json_dict = json.load(f)

    data_file = conf.DATASETS_PATH / Path(json_dict["data_file"])
    data = np.load(data_file)
    X = data["X"]
    true_spectra = data["densities"][:,np.newaxis]*data["phases"]
    true_maps = data["weights"]

    model = FUNC_MAP[json_dict["model"]](**json_dict["model_parameters"])
    model.generate_g_matr(**json_dict["g_parameters"])

    G = model.G

    estimator = nmf.NMF(**json_dict["hyperparameters"])
    estimator.fit(X,G=G)

    d = {}  # dictionary of everything we would like to save
    d["G"] = estimator.G_
    d["P"] = estimator.P_
    d["A"] = estimator.A_
    save_data_path = conf.RESULTS_PATH / Path("data/" + json_dict["filename"])
    np.savez(save_data_path, **d)

    angles = measures.find_min_angle(true_spectra,(estimator.G_@estimator.P_).T)
    mse = measures.find_min_MSE(true_maps,estimator.A_)

    save_dict = {}
    save_dict["inputs"] = json_dict
    save_dict["results"] = {"angles" : angles,"mse" : mse}
    save_meta_path = conf.RESULTS_PATH / Path("summaries/" + json_dict["filename"]+".json")

    with open(save_meta_path, 'w') as outfile:
        json.dump(save_dict, outfile, sort_keys=True, indent=4)
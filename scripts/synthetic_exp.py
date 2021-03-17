import json
import sys
import numpy as np
from pathlib import Path
from snmfem import estimators
from snmfem import models 
from snmfem.conf import SCRIPT_CONFIG_PATH, DATASETS_PATH, RESULTS_PATH
import snmfem.measures as measures


def run_experiment(sample, json_dict) : 
    # load data
    data = np.load(sample)
    X = data["X"]
    nx, ny, ns = X.shape
    Xflat = X.transpose([2,0,1]).reshape(ns, nx*ny)
    densities = data["densities"]
    phases = data["phases"]
    true_spectra = np.expand_dims(densities, axis=1) * phases * data["N"]
    true_maps = data["weights"]
    k = true_maps.shape[2]
    true_maps_flat = true_maps.transpose([2,0,1]).reshape(k,nx*ny)
    assert(true_maps.shape[:2] == (nx, ny))
    G = data["G"]

    # Load estimator
    Estimator = getattr(estimators, json_dict["estimator"]) 

    estimator = Estimator(**json_dict["hyperparameters"])
    estimator.fit(Xflat,G=G)

    G = estimator.G_
    P = estimator.P_
    A = estimator.A_
    angle = measures.find_min_angle(true_spectra,(G@P).T)
    mse = measures.find_min_MSE(true_maps_flat,A)
    return angle, mse, G, P, A


if __name__ == "__main__" : 
    json_file = sys.argv[1]
    
    if len(sys.argv)<3:
        save = False
    else:
        save = sys.argv[2]=="True"

    if len(sys.argv)<4:
        number = None
    else:
        number = [int(sys.argv[3])]

    # Open experiment file
    json_path = SCRIPT_CONFIG_PATH / Path(json_file)
    with open(json_path,"r") as f :
        json_dict = json.load(f)

    # Open dataset file
    dataset_json = json_dict["dataset"]
    json_dataset_path = SCRIPT_CONFIG_PATH / Path(dataset_json)
    with open(json_dataset_path,"r") as f :
        json_dataset_dict = json.load(f)

    # Load the list of samples
    data_folder = DATASETS_PATH / Path(json_dataset_dict["data_folder"])

    samples = list(data_folder.glob("sample_*.npz"))

    if number is not None:
        sample2process = []
        for sample in samples:
            if int(str(sample).split("_")[-1][:-4]) in number:
                sample2process.append(sample)
    else:
        sample2process = samples

    angles = []
    mses = []
    for sample in sample2process:
        sample_number = int(str(sample).split("_")[-1][:-4])
        angle, mse, G, P, A = run_experiment(sample, json_dict)
        angles.append(angle)
        mses.append(mses)
        if save :
            d = {}  # dictionary of everything we would like to save
            d["G"] = G
            d["P"] = P
            d["A"] = A
            data_foler = RESULTS_PATH / Path("data") 
            data_foler.mkdir(exist_ok=True, parents=True)
            save_data_path = data_foler / Path(json_dict["filename"]+"_{}".format(sample_number))
            np.savez(save_data_path, **d)

        save_dict = {}
        save_dict["dataset"] = json_dataset_dict
        save_dict["experiment"] = json_dict
        save_dict["samples"] = [str(s) for s in sample2process]
        save_dict["results"] = {"angles" : angles,"mse" : mse}
        summaries_foler = RESULTS_PATH / Path("summaries") 
        summaries_foler.mkdir(exist_ok=True, parents=True)
        save_meta_path = summaries_foler / Path(json_dict["filename"]+".json")

        with open(save_meta_path, 'w') as outfile:
            json.dump(save_dict, outfile, sort_keys=True, indent=4)
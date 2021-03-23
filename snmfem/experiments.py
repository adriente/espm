from snmfem import estimators 
from  snmfem import  measures
import numpy as np
import snmfem.conf as conf
from pathlib import Path
import json

def compute_metrics(true_spectra, true_maps, GP, A, get_ind = False, u = False):
    angle = measures.find_min_angle(true_spectra, GP.T, get_ind, unique=u)
    mse = measures.find_min_MSE(true_maps, A, get_ind, unique=u)
    return angle, mse

def run_experiment(Xflat, true_spectra, true_maps, G, experiment, params_evalution) : 
    
    Estimator = getattr(estimators, experiment["method"]) 

    estimator = Estimator(**experiment["params"])
    
    estimator.fit(Xflat,G=G)
    
    G = estimator.G_
    P = estimator.P_
    A = estimator.A_
    
    # if isinstance(estimator, estimators.NMFEstimator):
    #     losses = np.hstack((np.expand_dims(np.array(estimator.losses1), 1), np.array(estimator.detailed_losses), np.array(estimator.rel)))
    # else:
    losses = None
    metrics = compute_metrics(true_spectra, true_maps, G@P, A, **params_evalution)
    return metrics, (G@P, A), losses

def load_data(sample) : 
    data = np.load(sample)
    X = data["X"]
    nx, ny, ns = X.shape
    Xflat = X.transpose([2,0,1]).reshape(ns, nx*ny)
    densities = data["densities"]
    phases = data["phases"]
    true_spectra_flat = np.expand_dims(densities, axis=1) * phases * data["N"]
    true_maps = data["weights"]
    k = true_maps.shape[2]
    true_maps_flat = true_maps.transpose([2,0,1]).reshape(k,nx*ny)
    G = data["G"]
    return Xflat, true_spectra_flat, true_maps_flat, G

def load_samples(dataset):
    data_json = conf.SCRIPT_CONFIG_PATH / Path(dataset)
    with open(data_json,"r") as f :
        data_dict = json.load(f)
    k = data_dict["model_parameters"]["params_dict"]["k"]
    shape_2D = data_dict["weights_parameters"]["shape_2D"]
    data_folder = conf.DATASETS_PATH / Path(data_dict["data_folder"])
    samples = list(data_folder.glob("sample_*.npz"))
    return samples, k, shape_2D

def perform_simulations(samples, exp_list, params_evalution):
    
    metrics = []
    for s in samples: 
        Xflat, true_spectra, true_maps, G = load_data(s)
        m = []
        for exp in exp_list : 
            m.append(run_experiment(Xflat, true_spectra, true_maps, G, exp, params_evalution)[0])
        metrics.append(m)
    metrics = np.array(metrics)
    return metrics


def print_results(exp_list, metrics, metrics_names=["Phase angles", "Map MSE"]):
    std = np.std(metrics, axis=0)
    mean = np.mean(metrics, axis=0)
    tc = 25
    ec = 18
    nc = len(exp_list)*ec//2
    txt = ""
    for j, metrics_name in enumerate(metrics_names):
        txt += "="*nc + metrics_name.center(tc) +"="*nc +"=\n"
        for i, exp in enumerate(exp_list):
            txt += "| {}".format(exp["name"]).ljust(tc)
            for a,b in zip(mean[i,j], std[i,j]):
                txt += "| {:.2f} ± {:.2f}".format(a, b).ljust(ec)
            txt += "|\n"
        txt += "\n"
    print(txt)

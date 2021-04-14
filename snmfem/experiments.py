from snmfem import estimators 
from  snmfem import  measures
import numpy as np
import snmfem.conf as conf
from pathlib import Path
import json
from argparse import ArgumentParser, Namespace
import os

def compute_metrics(true_spectra, true_maps, GP, A, u = False):
    angle, ind1 = measures.find_min_angle(true_spectra, GP.T, True, unique=u)
    mse, ind2 = measures.find_min_MSE(true_maps, A, True, unique=u)
    return angle, mse, (ind1, ind2)

def run_experiment(Xflat, true_spectra, true_maps, G, experiment, params_evalution, shape_2d = None) : 
    
    Estimator = getattr(estimators, experiment["method"]) 

    estimator = Estimator(**experiment["params"])
    
    estimator.fit(Xflat,G=G,shape_2d = shape_2d,true_D = true_spectra.T, true_A = true_maps )
    
    G = estimator.G_
    P = estimator.P_
    A = estimator.A_
    
    losses = estimator.get_losses()
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
    shape_2d = data["shape_2d"]
    return Xflat, true_spectra_flat, true_maps_flat, G, shape_2d

def load_samples(dataset, base_path_conf=conf.SCRIPT_CONFIG_PATH, base_path_dataset = conf.DATASETS_PATH):
    data_json = base_path_conf / Path(dataset)
    print(data_json)
    with open(data_json,"r") as f :
        data_dict = json.load(f)
    k = data_dict["model_parameters"]["params_dict"]["k"]
    data_folder = base_path_dataset / Path(data_dict["data_folder"])
    samples = list(data_folder.glob("sample_*.npz"))
    return samples, k

def perform_simulations(samples, exp_list, params_evalution):
    
    metrics = []
    for s in samples: 
        Xflat, true_spectra, true_maps, G, shape_2d = load_data(s)
        m = []
        for exp in exp_list : 
            m.append(run_experiment(Xflat, true_spectra, true_maps, G, exp, params_evalution, shape_2d=shape_2d)[0][:-1])
        metrics.append(m)
    metrics = np.array(metrics)
    return metrics


def print_results(exp_list, metrics, metrics_names=["Phase angles", "Map MSE"]):
    std = np.std(metrics, axis=0)
    mean = np.mean(metrics, axis=0)
    tc = 25
    ec = 18
    nc = metrics.shape[3]*ec //2
    txt = ""
    for j, metrics_name in enumerate(metrics_names):
        txt += "="*nc + metrics_name.center(tc) +"="*nc +"=\n"
        for i, exp in enumerate(exp_list):
            txt += "| {}".format(exp["name"]).ljust(tc)
            for a,b in zip(mean[i,j], std[i,j]):
                txt += "| {:.2f} ± {:.2f}".format(a, b).ljust(ec)
            txt += "|\n"
        txt += "\n"
    # print(txt)
    return txt


def experiment_parser (argv) : 
    parser = ArgumentParser()
    
    pos_g = parser.add_argument_group("positional_group")
    for key in conf.POS_ARGS.keys() : 
        pos_g.add_argument(conf.POS_ARGS[key][0],**conf.POS_ARGS[key][1])
    
    eval_g = parser.add_argument_group("evaluation_group")
    for key in conf.EVAL_ARGS.keys(): 
        eval_g.add_argument(conf.EVAL_ARGS[key][0],conf.EVAL_ARGS[key][1],**conf.EVAL_ARGS[key][2])
    
    est_g = parser.add_argument_group("estimator_group")
    for key in conf.ESTIMATOR_ARGS.keys(): 
        est_g.add_argument(conf.ESTIMATOR_ARGS[key][0],conf.ESTIMATOR_ARGS[key][1],**conf.ESTIMATOR_ARGS[key][2])
    
    args = parser.parse_args(argv)
    arg_groups = {}
    for group in parser._action_groups:
        group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title]=Namespace(**group_dict)
    
    return vars(arg_groups["positional_group"]), vars(arg_groups["estimator_group"]), vars(arg_groups["evaluation_group"])

def build_exp(k,pos_dict,est_dict) : 
    d = {}
    d["name"] = pos_dict["method"]
    d["method"] = pos_dict["method"]
    k_dict = {"n_components" : k}
    estimator_dict = {}
    for key in est_dict.keys() : 
        if conf.ESTIMATOR_ARGS[key][3] is None : 
            estimator_dict[key] = est_dict[key]
        elif pos_dict["method"] in conf.ESTIMATOR_ARGS[key][3] : 
            estimator_dict[key] = est_dict[key]
        else : 
            pass
    d["params"] = {**k_dict, **estimator_dict}
    return d

def store_in_file(file,metrics,matrices_tuple,losses) :
    d = {}
    d["GP"] = matrices_tuple[0]
    d["A"] = matrices_tuple[1]
    d["metrics"] = metrics
    d["losses"] = losses
    filename = conf.RESULTS_PATH / Path(file)
    np.savez(filename, **d)

def print_in_file(exp_list,metrics,out_file,estimator_dict = {}) :
    output_file = conf.RESULTS_PATH / Path(out_file)
    results = print_results(exp_list,metrics) 
    txt =100*"#" + "\n"
    if estimator_dict == {} :
        txt += "Using default parameters\n"
        txt += "\n"
    else : 
        txt += "Estimator parameters\n"
        txt += 20*"-" + "\n"
        for elt in estimator_dict.keys() :
            txt += "|{:10} : {:>5}|".format(elt,estimator_dict[elt])
        txt +="\n"

    print(txt)
    print(results)
    with open(output_file,"a") as f : 
        f.write(txt)
        f.write(results)

def gather_results(file_list,output,folder = conf.RESULTS_PATH,output_folder = conf.RESULTS_PATH) : 
    string_list = []
    path_list = [folder / Path(a) for a in file_list]
    
    for file in path_list : 
        with open(file,"r") as f : 
            string_list.append(f.read())

    output_path = output_folder / Path(output)
    for elt in string_list : 
        with open(output_path, "a") as o :
            o.write(elt)
            o.write("\n")

    for file in path_list : 
        os.remove(file)


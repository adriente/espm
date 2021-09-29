from snmfem import estimators 
from  snmfem import  measures
import numpy as np
import snmfem.conf as conf
import snmfem.utils as u
from pathlib import Path
from argparse import ArgumentParser, Namespace
import hyperspy.api as hs
from snmfem.utils import number_to_symbol_list
from snmfem.models.EDXS_function import elts_dict_from_dict_list

@number_to_symbol_list
def zeros_dict (elements = "Si") :
    d = {} 
    for elt in elements : 
        d[elt] = 0.0
    return d

def build_fixed_P (spim, col1 = False) : 
    phases_pars = spim.metadata.Truth.phases
    unique_elts = list((elts_dict_from_dict_list([x["elements_dict"] for x in phases_pars])).keys())
    P_dict = {}
    for i, phase in enumerate(phases_pars) : 
        elts_list = list(phase["elements_dict"].keys())
        other_elts = list(set(unique_elts) - set(elts_list))
        if col1 and (i!=0) : 
            d = {}
        else : 
            d = zeros_dict(elements=other_elts)
        P_dict[str(i)] = d
        print(P_dict)
    P = spim.set_fixed_P(P_dict)
    return P

def compute_metrics(true_spectra, true_maps, GP, A, u = True):
    angle, ind1 = measures.find_min_angle(true_spectra, GP.T, True, unique=u)
    mse, ind2 = measures.find_min_MSE(true_maps, A, True, unique=u)
    return angle, mse, (ind1, ind2)

def run_experiment(spim,estimator,experiment,simulated = False) : 
    out = spim.decomposition(algorithm = estimator, return_info = True)
    
    P = out.P_
    A = out.A_
    G = out.G_
    
    losses = estimator.get_losses()
    if simulated :
        true_spectra, true_maps = spim.phases, spim.weights
        metrics = compute_metrics(true_spectra.T, true_maps, G@P, A)
    else : 
        temp = np.zeros((experiment["params"]["n_components"],))
        metrics = (temp, temp, (temp,temp))
    return metrics, (G, P, A), losses

def simulation_quick_load(experiment, P_type = None) : 
    spim = hs.load(experiment["input_file"])
    # spim.set_signal_type("EDXSsnmfem")
    G = spim.build_G(problem_type = experiment["g_type"])
    shape_2d = spim.shape_2d
    if P_type == "full" : 
        P_in = build_fixed_P(spim)
    elif P_type == "partial" : 
        P_in = build_fixed_P(spim, col1 = True)
    else : 
        P_in = None
    Estimator = getattr(estimators, experiment["method"])  
    D, A = spim.phases, spim.weights
    estimator = Estimator(G = G, shape_2d = shape_2d, true_D = D, true_A = A, **experiment["params"],fixed_P = P_in,hspy_comp = True)
    return spim, estimator

def experimental_quick_load(experiment, P_dict = None) : 
    spim = hs.load(experiment["input_file"])
    G = spim.build_G(problem_type = experiment["g_type"])
    shape_2d = spim.shape_2d
    P = spim.set_fixed_P(P_dict)
    Estimator = getattr(estimators, experiment["method"])  
    estimator = Estimator(G = G, shape_2d = shape_2d, **experiment["params"],fixed_P = P,hspy_comp = True)
    return spim, estimator

# To be verified ...
# def perform_simulations(exp_list,n_samples = 10, simulated = False):
#     augm_exp_list = []
#     for exp in exp_list : 
#         files = [exp["input_file"] / Path("sample_{}.hspy".format(i)) for i in range(n_samples)]
#         new_exps = []
#         for f in files : 
#             exp["input_file"] = f
#             new_exps.append(exp)
#         augm_exp_list.append(new_exps)
    
#     metrics = []
#     for augm_exp in augm_exp_list :
#         m = []
#         for exp in augm_exp :
#             if simulated :  
#             estim = simulation_quick_load(exp)
#             m.append(run_experiment(estimator = estim, experiment=exp, simulated= simulated)[0][:-1])
#             metrics.append(m)
#     metrics = np.array(metrics)
#     return metrics


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

def build_exp(pos_dict,est_dict, name = None) : 
    d = {}
    if name is None : 
        d["name"] = pos_dict["method"]
    else : 
        d["name"] = name
    d["method"] = pos_dict["method"]
    d["input_file"] = pos_dict["input_file"]
    d["g_type"] = pos_dict["g_type"]
    estimator_dict = {}
    for key in est_dict.keys() : 
        if conf.ESTIMATOR_ARGS[key][3] is None : 
            estimator_dict[key] = est_dict[key]
        elif pos_dict["method"] in conf.ESTIMATOR_ARGS[key][3] : 
            estimator_dict[key] = est_dict[key]
        else : 
            pass
    d["params"] = {"n_components" : pos_dict["k"], **estimator_dict}
    return d

def store_in_file(file,metrics,matrices_tuple,losses) :
    d = {}
    d["G"] = matrices_tuple[0]
    d["P"] = matrices_tuple[1]
    d["A"] = matrices_tuple[2]
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

def fill_exp_dict(input_dict) :

    parser = ArgumentParser()
    for key in conf.ESTIMATOR_ARGS.keys(): 
        parser.add_argument(conf.ESTIMATOR_ARGS[key][0],conf.ESTIMATOR_ARGS[key][1],**conf.ESTIMATOR_ARGS[key][2])
    args = parser.parse_args([])
    default_dict =  vars(args)

    return u.arg_helper(input_dict,default_dict)

def get_ROI_ranges(spim,ROI) : 
    scale_h, offset_h = spim.axes_manager[0].scale, spim.axes_manager[0].offset
    scale_v, offset_v = spim.axes_manager[1].scale, spim.axes_manager[1].offset
    left = int((ROI.left - offset_h)/scale_h)
    right = int((ROI.right - offset_h)/scale_h)
    top = int((ROI.top - offset_v)/scale_v)
    bottom = int((ROI.bottom - offset_v)/scale_v)
    return left, right, top, bottom

def build_index_list (left,right,top,bottom, size_h) : 
    ind_list = []
    for i in range(left,right) : 
        for j in range(top,bottom) : 
            ind_list.append(num_indices((i,j),size_h))
    ind_list.sort()
    return ind_list

def num_indices (ij, size_h) : 
    return ij[0] + size_h*ij[1]

# def gather_results(file_list,output,folder = conf.RESULTS_PATH,output_folder = conf.RESULTS_PATH) : 
#     string_list = []
#     path_list = [folder / Path(a) for a in file_list]
    
#     for file in path_list : 
#         with open(file,"r") as f : 
#             string_list.append(f.read())

#     output_path = output_folder / Path(output)
#     for elt in string_list : 
#         with open(output_path, "a") as o :
#             o.write(elt)
#             o.write("\n")

#     for file in path_list : 
#         os.remove(file)


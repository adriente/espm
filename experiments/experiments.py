from espm import estimators 
from  espm import  measures
import numpy as np
import conf
import espm.utils as u
from pathlib import Path
from argparse import ArgumentParser, Namespace
import hyperspy.api as hs
from espm.utils import number_to_symbol_list
from espm.models.EDXS_function import elts_dict_from_dict_list
import numpy.lib.recfunctions as rfn

#####################################################################
# For HPC use, parser and building experiments to run the algorithm #
#####################################################################

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

def fill_exp_dict(input_dict) :

    parser = ArgumentParser()
    for key in conf.ESTIMATOR_ARGS.keys(): 
        parser.add_argument(conf.ESTIMATOR_ARGS[key][0],conf.ESTIMATOR_ARGS[key][1],**conf.ESTIMATOR_ARGS[key][2])
    args = parser.parse_args([])
    default_dict =  vars(args)

    return u.arg_helper(input_dict,default_dict)

##########################################
# Loading data and running the algorithm #
##########################################

def run_experiment(spim,estimator,experiment,sim = False) : 
    out = spim.decomposition(algorithm = estimator, return_info = True)
    
    W = out.W_
    H = out.H_
    G = out.G_
    
    losses = estimator.get_losses()
    if sim :
        true_spectra, true_maps = spim.phases, spim.maps
        metrics = measures.find_min_config(true_maps, true_spectra.T, H, (G@W).T, angles = True)
    else : 
        temp = np.zeros((experiment["params"]["n_components"],))
        metrics = (temp, temp, temp, False)
    return metrics, (G, W, H), losses

def quick_load(experiment,sim = True, W_dict = None) : 
    spim = hs.load(experiment["input_file"])
    spim.change_dtype("float64")
    # spim.set_signal_type("EDXSsnmfem")
    G = spim.build_G(problem_type = experiment["g_type"])
    shape_2d = spim.shape_2d
    Estimator = getattr(estimators, experiment["method"]) 
    if sim :  
        D, H = spim.phases, spim.maps
    else : 
        D, H = None, None
    if W_dict is None : 
        W = None
    else : 
        W = spim.set_fixed_W(W_dict)
    estimator = Estimator(G = G, shape_2d = shape_2d, true_D = D, true_H = H, **experiment["params"],fixed_W = W,hspy_comp = True)
    return spim, estimator

def run_several_experiments(experiment,output_file,n_samples = 10, W_dict = None) :
    metrics_summary = []
    folder = experiment["input_file"]
    d = {}
    for i in range(n_samples) : 
        # Load parameters and data
        file = str(Path(folder) / Path("sample_{}.hspy".format(i)))
        spim = hs.load(file)
        G = spim.build_G(problem_type = experiment["g_type"])
        shape_2d = spim.shape_2d
        Estimator = getattr(estimators, experiment["method"])
        D, H = spim.phases, spim.maps
        if W_dict is None : 
            W = None
        else : 
            W = spim.set_fixed_W(W_dict)
        estimator = Estimator(G = G, shape_2d = shape_2d, true_D = D, true_H = H, **experiment["params"],fixed_W = W,hspy_comp = True)
        
        # Decomposition
        out = spim.decomposition(algorithm = estimator, return_info = True)

        # Results
        W = out.W_
        H = out.H_
        G = out.G_
        losses = out.get_losses()

        d["G_{}".format(i)] = G
        d["W_{}".format(i)] = W
        d["H_{}".format(i)] = H
        d["losses_{}".format(i)] = losses

        true_spectra, true_maps = spim.phases, spim.maps
        metrics_summary.append(measures.find_min_config(true_maps, true_spectra.T, H, (G@W).T, angles = True))

    np.savez(output_file, **d)

    k = experiment["params"]["n_components"]
    names_a = []
    names_m = []
    formats = (k)*["float64"]
    for i in range(k) : 
        names_a.append("angle_p{}".format(i))
        names_m.append("mse_p{}".format(i))
    
    angles_array = np.zeros((n_samples,),dtype = {"names" : names_a, "formats" : formats})
    mse_array = np.zeros((n_samples,),dtype = {"names" : names_m, "formats" : formats})
    for j,metrics in enumerate(metrics_summary) :
        for i in range(k) : 
            key_a = "angle_p{}".format(i)
            key_m = "mse_p{}".format(i)
            angles_array[key_a][j] = metrics[0][metrics[2][i]]
            mse_array[key_m][j] = metrics[1][i]

    warn_array = np.zeros((n_samples,),dtype={"names" : ["warning"], "formats" : ["bool"]})
    for j,metrics in enumerate(metrics_summary) : 
        warn_array["warning"][j] = metrics[3]

    output = rfn.merge_arrays((angles_array,mse_array,warn_array), flatten = True, usemask = False)

    return output

#################################################
# Function to save the results or printing them #
#################################################

def struct_arr_mean (struct_array) : 
    '''
    Function to compute the mean of elements of every column of a named array
    Note : In the future it may be better to use pandas
    '''
    mean_array = np.zeros((1,),dtype = struct_array.dtype)
    for name in struct_array.dtype.names : 
        mean_array[name] = np.mean(struct_array[name])
    return mean_array

def struct_arr_std (struct_array) : 
    '''
    Function to compute the std of elements of every column of a named array
    Note : In the future it may be better to use pandas
    '''
    std_array = np.zeros((1,),dtype = struct_array.dtype)
    for name in struct_array.dtype.names : 
        std_array[name] = np.std(struct_array[name])
    return std_array


def results_string(experiment, metrics):
    std = struct_arr_std(metrics)
    mean = struct_arr_mean(metrics)

    title = ' ' + experiment["name"] + '\n'
    top = ' Metrics |'
    sep = '%----------'
    means = ' Means   |'
    stds = ' StDs    |'
    for name in metrics.dtype.names : 
        len_name = len(name)
        top += ' ' + name + ' '
        sep += (2 + len_name)*'-' 
        fmt = "{:" + str(len_name)+ '.4f}'
        means += ' ' + str(fmt.format(mean[name][0])) + ' '
        stds +=  ' ' + str(fmt.format(std[name][0])) + ' '

    middle = ''
    for i in range(metrics.shape[0]) : 
        middle += '         |'
        for name in metrics.dtype.names  : 
            len_name = len(name)
            fmt = "{:" + str(len_name)+ '.4f}'
            middle += ' ' + str(fmt.format(metrics[name][i])) + ' '
        middle += "|\n"

    
    top += '|\n'
    sep += '%\n'
    means += '|\n'
    stds += '|\n'

    return title + top + sep + middle + means + stds
    

def store_in_file(file,metrics,matrices_tuple,losses) :
    d = {}
    d["G"] = matrices_tuple[0]
    d["W"] = matrices_tuple[1]
    d["H"] = matrices_tuple[2]
    d["metrics"] = metrics
    d["losses"] = losses
    filename = conf.RESULTS_PATH / Path(file)
    np.savez(filename, **d)

def print_in_file(experiment, metrics, out_file) :
    output_file = conf.RESULTS_PATH / Path(out_file)
    results = results_string(experiment, metrics)
    with open(output_file,"a") as f : 
        f.write(results)


###################################################
# Functions for selecting and initalising fixed_A #
###################################################

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

###################################################
# Function to initalise fixed_P of simulated data #
###################################################

@number_to_symbol_list
def zeros_dict (elements = "Si") :
    d = {} 
    for elt in elements : 
        d[elt] = 0.0
    return d

def build_fixed_W (spim, col1 = False) : 
    phases_pars = spim.metadata.Truth.phases
    unique_elts = list((elts_dict_from_dict_list([x["elements_dict"] for x in phases_pars])).keys())
    W_dict = {}
    for i, phase in enumerate(phases_pars) : 
        elts_list = list(phase["elements_dict"].keys())
        other_elts = list(set(unique_elts) - set(elts_list))
        if col1 and (i!=0) : 
            d = {}
        else : 
            d = zeros_dict(elements=other_elts)
        W_dict[str(i)] = d
        print(W_dict)
    W = spim.set_fixed_W(W_dict)
    return W

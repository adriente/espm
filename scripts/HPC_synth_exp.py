import numpy as np
from snmfem.experiments import perform_simulations, load_samples, print_results, load_data, run_experiment
from argparse import ArgumentParser,Namespace
import sys
from snmfem.conf import RESULTS_PATH
from pathlib import Path


default_params = {
    # Common parms for estimators
    "tol" : 1e-6,
    "max_iter" : 10000,
    "init" : "random",
    "random_state" : 1,
    "verbose" : 0,

    # Evaluation params
    "u" : True,
    "file" : "dump.txt",

    # SNMFEM params
    "force_simplex" : True,
    "skip_G" : False,
    "mu" : 0.0,
    "lambda_L" : 1.0,

    # Scikit-learn params
    "beta_loss" : "frobenius",
    "alpha" : 0.0,
    "l1_ratio" : 1.0,
    "solver" : "mu",
    "regularization" : "components"
    }

dataset = "dataset_EDXS_small.json"

def parse_args(argv,default_params) : 
    parser = ArgumentParser()
    pos_g = parser.add_argument_group("positional_group")
    pos_g.add_argument("json",help = "str : Name of the json file containing info about the data")
    pos_g.add_argument("method",choices = ["NMF","SmoothNMF","SKNMF"],help = "str : Name of the estimator for the decomposition")
    pos_g.add_argument("-f","--file",default=default_params["file"],help="str : Name of the txt file where the data are stored")

    default_g = parser.add_argument_group('default_group')
    default_g.add_argument("-mi","--max_iter",type=int,default=default_params["max_iter"],help="int : Max number of iterations for the algorithm")
    default_g.add_argument("-v","--verbose",action="store_true",default=default_params["verbose"], help = "None : Activate to display details about the algorithm")
    init_list = ["random","nndsvd","nndsvda","nndsvdar","custom"]
    default_g.add_argument("-i","--init",choices=init_list,default=default_params["init"], help="str : Initialisation method")
    default_g.add_argument("-t","--tol",type=float,default=default_params["tol"],help="float : Stopping criterion")

    evaluation_g = parser.add_argument_group("evaluation_group")
    evaluation_g.add_argument("-u","--u",action="store_false",default=default_params["u"], help="None : Activate so that each result is uniquely matched with a ground truth.")
    
    
    subparsers = parser.add_subparsers(dest = "estimator",help='choose between snmfem, smooth_snmfem, sk. Then add the relevant options for the corresponding algorithm')
    
    snmfem = subparsers.add_parser("snmfem",help="snmfem.NMF algorithm || -sG, --skip_G --> None : Activate G matrix || -fs, --force_simplex --> None : Activate simplex constraint || -mu --> float : strenght of the mu reg")
    snmfem_g = snmfem.add_argument_group("estimator_group")
    snmfem_g.add_argument("-sG","--skip_G",action="store_true",default=default_params["skip_G"])
    snmfem_g.add_argument("-fs","--force_simplex",action="store_false",default = default_params["force_simplex"])
    snmfem_g.add_argument("-mu",type=float,default=default_params["mu"])
    
    smooth_snmfem = subparsers.add_parser("smooth_snmfem",help="snmfem.SmoothNMF algorithm || -sG, --skip_G --> None : Activate G matrix || -fs, --force_simplex --> None : Activate simplex constraint || -mu --> float : strenght of the mu reg || -l, --lambda_L --> float : strengh of the Laplacian reg")
    smooth_snmfem_g = smooth_snmfem.add_argument_group("estimator_group")
    smooth_snmfem_g.add_argument("-sG","--skip_G",action="store_true",default=default_params["skip_G"])
    smooth_snmfem_g.add_argument("-fs","--force_simplex",action="store_false",default = default_params["force_simplex"])
    smooth_snmfem_g.add_argument("-mu",type=float,default=default_params["mu"])
    smooth_snmfem_g.add_argument("-l","--lambda_L",default=default_params["lambda_L"])

    sk = subparsers.add_parser("sk",help = "snmfem.SKNMF algorithm || -bl, --beta_loss || -s, --solver || -a, --alpha || -l1, --l1_ratio || -r, --regularization")
    sk_g = sk.add_argument_group("estimator_group")
    sk_g.add_argument("-bl","--beta_loss",choices=["frobenius","kullback-leibler"],default=default_params["beta_loss"])
    sk_g.add_argument("-s","--solver",choices=["mu","cd"],default=default_params["solver"])
    sk_g.add_argument("-a","--alpha",type=float,default=default_params["alpha"])
    sk_g.add_argument("-l1","--l1_ratio",type=float,default=default_params["l1_ratio"])
    sk_g.add_argument("-r","--regularization",choices=["both","components","transformation"],default=default_params["regularization"])

    args = parser.parse_args(argv)
    main_arg_groups = {}
    sub_arg_groups = {}

    for group in parser._action_groups:
        group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        main_arg_groups[group.title]=Namespace(**group_dict)

    chosen_one = vars(args)["estimator"]
    if chosen_one is None : 
        return vars(main_arg_groups["positional_group"]), vars(main_arg_groups["default_group"]), \
         vars(main_arg_groups["evaluation_group"]), {}
    else : 
        for group in parser._subparsers._group_actions[0].choices[chosen_one]._action_groups:
            group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
            sub_arg_groups[group.title]=Namespace(**group_dict)
            
        return vars(main_arg_groups["positional_group"]), vars(main_arg_groups["default_group"]), \
            vars(main_arg_groups["evaluation_group"]), vars(sub_arg_groups["estimator_group"])

def build_exp_list(k,shape_2d,positional_dict,default_dict,estimator_dict = {}) : 
    dict = {}
    dict["name"] = positional_dict["method"]
    dict["method"] = positional_dict["method"]
    dict["params"] = default_dict.copy()
    dict["params"].update(estimator_dict)
    dict["params"].update({"n_components": k})
    if dict["method"] == "smooth_snmfem" :
        dict["params"].update({"shape_2d" : shape_2d})
    return [dict]

def print_in_file(exp_list,metrics,out_file,estimator_dict = {}) :
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
    with open(out_file,"a") as f : 
        f.write(txt)
        f.write(results)

def run_exp(positional_dict,default_dict,evaluation_dict,estimator_dict,output) :
    samples, k, shape_2d = load_samples(positional_dict["json"])
    exp_list = build_exp_list(k,shape_2d,positional_dict,default_dict,estimator_dict)
    metrics = perform_simulations(samples,exp_list,evaluation_dict)
    print_in_file(exp_list,metrics,output,estimator_dict)

if __name__ == "__main__" : 
    p_dict, d_dict, eval_dict, est_dict = parse_args(sys.argv[1:],default_params)
    output_file = RESULTS_PATH / Path(p_dict["file"])
    metrics = run_exp(p_dict,d_dict,eval_dict,est_dict,output=output_file)


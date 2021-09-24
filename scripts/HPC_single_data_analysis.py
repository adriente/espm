import numpy as np
import snmfem.experiments as e
import sys

ind_list_file = "somefilename"
with open(ind_list_file, "r") as f :
    ind_list = [int(i) for i in f.read().split()]


def run_single_exp(pos_dict,est_dict,eval_dict) :
    file = pos_dict["json"]
    Xflat,shape_2d, mod_pars, g_pars = da.load_hs_data(file)
    G = da.build_model(mod_pars,g_pars,G_func = True)
    experiment = da.build_analysis(pos_dict, est_dict)
    G,P,A,losses = da.run_analysis(Xflat, G, experiment, shape_2d = shape_2d,mod_pars = mod_pars,g_pars = g_pars, ind_list = ind_list)
    da.save_results(eval_dict["file"], G, P, A, losses)

if __name__ == "__main__" : 
    pos, est, eval = e.experiment_parser(sys.argv[1:])
    print(pos,est,eval)
    run_single_exp(pos,est,eval)
import numpy as np
import snmfem.experiments as e
import sys
from snmfem.conf import RESULTS_PATH
from pathlib import Path


def run_single_exp(pos_dict,est_dict,eval_dict) :
    samples, k = e.load_samples(pos_dict["json"])
    Xflat, true_spectra_flat, true_maps_flat, G, shape_2d = e.load_data(samples[0])
    exp = e.build_exp(k,pos_dict,est_dict)
    eval = {"u" : eval_dict["u"]}
    metrics, matrices_tuple, losses = e.run_experiment(Xflat,true_spectra_flat,true_maps_flat,G,exp,eval,shape_2d=shape_2d) 
    e.store_in_file(eval_dict["file"], metrics[0], matrices_tuple,losses)

if __name__ == "__main__" : 
    pos, est, eval = e.experiment_parser(sys.argv[1:])
    print(pos,est,eval)
    run_single_exp(pos,est,eval)
    # run_exp(a,default_opt)
    # params_dict = parse_args(sys.argv[1:],default_params)
    # run_exp(params_dict)


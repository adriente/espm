import numpy as np
import snmfem.experiments as e
import snmfem.data_analysis as da
import sys

def run_single_exp(pos_dict,est_dict,eval_dict) :
    file = da.load_filename(pos_dict["json"])
    Xflat, shape_2d, offset, scale, size = da.load_hs_data(file)
    G = da.build_model(pos_dict["json"], offset, scale, size)
    experiment = da.build_analysis(pos_dict, est_dict)
    G,P,A,losses = da.run_analysis(Xflat, G, experiment)
    da.save_results(eval_dict["file"], G, P, A, losses)

if __name__ == "__main__" : 
    pos, est, eval = e.experiment_parser(sys.argv[1:])
    print(pos,est,eval)
    run_single_exp(pos,est,eval)
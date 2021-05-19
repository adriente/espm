import numpy as np
import snmfem.experiments as e
import snmfem.data_analysis as da
import sys
import json


def run_single_exp(jsonfile) :
    with open(jsonfile,"r") as f : 
        dict = json.load(f)
    file = da.load_filename(jsonfile)
    Xflat, shape_2d, offset, scale, size = da.load_hs_data(file)
    G = da.build_model(jsonfile, offset, scale, size)
    experiment = da.build_analysis(dict["main"],dict["estimator"] )
    G,P,A,losses = da.run_analysis(Xflat, G, experiment,shape_2d)
    da.save_results(dict["output_file"], G, P, A, losses)

if __name__ == "__main__" : 
    filename = "config/example.json"
    run_single_exp(filename)
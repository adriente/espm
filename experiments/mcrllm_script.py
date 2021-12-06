import hyperspy.api as hs
import numpy as np
from conf import RESULTS_PATH, DATASETS_PATH
from pathlib import Path
import sys
from mcrllm_wrapper import MCRLLM

def experiment (input_file, output_file) : 
    dataset_path = DATASETS_PATH / Path(input_file)
    true_spim = hs.load(str(dataset_path))


    estimator = MCRLLM(n_components=3, init="Kmeans", max_iter=50000,hspy_comp=True)

    true_spim.decomposition(algorithm=estimator,verbose=True)


    d = {}
    d["G"] = estimator.G_
    d["P"] = estimator.P_
    d["A"] = estimator.A_

    full_output_file = RESULTS_PATH / Path(output_file)

    np.savez(str(full_output_file), **d)

if __name__ == "__main__" : 
    experiment(sys.argv[1],sys.argv[2])


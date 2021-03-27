import numpy as np
import snmfem.experiments as e
from argparse import ArgumentParser,Namespace
import sys
from snmfem.conf import RESULTS_PATH
from pathlib import Path
from glob import glob
import os




def run_batch_exp(positional_dict,evaluation_dict,estimator_dict,output) :
    samples, k = e.load_samples(positional_dict["json"])
    exp_list = [e.build_exp(k,positional_dict,estimator_dict)]
    eval = {"u" : evaluation_dict["u"]}
    metrics = e.perform_simulations(samples,exp_list,eval)
    e.print_in_file(exp_list,metrics,output,estimator_dict)

if __name__ == "__main__" : 
    p_dict, est_dict, eval_dict = e.experiment_parser(sys.argv[1:])
    print(p_dict,eval_dict,est_dict)
    
    int_seq = np.random.randint(low=0,high = 9, size = 10)
    file_name = "dump_"
    for elt in int_seq : 
        file_name += str(elt)
    file_name += ".txt"
    
    run_batch_exp(p_dict,eval_dict,est_dict,output=file_name)
    
    if eval_dict["gather_files"] :
        os.chdir(RESULTS_PATH)
        files = glob("*dump*.txt")
        e.gather_results(files,eval_dict["file"])



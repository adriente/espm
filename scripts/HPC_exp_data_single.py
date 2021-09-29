import snmfem.experiments as e
import sys

fixed_P_dict = {"p0" : {"Al" : 0.0}, "p1" : {"Nd" : 0.0, "Ni" : 0.0}}

def run_single_exp(pos_dict,est_dict,eval_dict, fixed_P = "True") :
    exp = e.build_exp(pos_dict, est_dict)
    if fixed_P == "True"  : 
        spim, estim = e.experimental_quick_load(exp,P_dict = fixed_P_dict)
    else : 
        spim, estim = e.experimental_quick_load(exp,P_dict = None)
    metrics, mat_tuple, losses = e.run_experiment(spim, estim, exp, simulated=False)
    e.store_in_file(eval_dict["output_file"], metrics, mat_tuple, losses)

if __name__ == "__main__" : 
    print(sys.argv[1:])
    pos, est, eval = e.experiment_parser(sys.argv[1:-1])
    print(pos,est,eval)
    run_single_exp(pos,est,eval, fixed_P=sys.argv[-1])
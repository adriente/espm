from snmfem import estimators
import snmfem.experiments as e
import sys

fixed_P = "full"

def run_single_exp(pos_dict,est_dict,eval_dict) :
    exp = e.build_exp(pos_dict, est_dict)
    spim, estim = e.quick_load(exp,P_type = fixed_P, simulated=True)
    metrics, mat_tuple, losses = e.run_experiment(spim, estim, exp, simulated=True)
    e.store_in_file(eval_dict["output_file"], metrics, mat_tuple, losses)

if __name__ == "__main__" : 
    print(sys.argv[1:])
    pos, est, eval = e.experiment_parser(sys.argv[1:])
    print(pos,est,eval)
    run_single_exp(pos,est,eval)
    # run_exp(a,default_opt)
    # params_dict = parse_args(sys.argv[1:],default_params)
    # run_exp(params_dict)


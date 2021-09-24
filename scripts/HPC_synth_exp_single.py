from snmfem import estimators
import snmfem.experiments as e
import sys

fixed_P = 0

def run_single_exp(pos_dict,est_dict,eval_dict) :
    exp = e.build_exp(pos_dict, est_dict)
    if fixed_P == 0 : 
        P_in = e.build_fixed_P(spim)
    elif fixed_P == 1 : 
        P_in = e.build_fixed_P(spim, col1 = True)
    else : 
        P_in = None
    spim, estim = e.quick_load(exp,P_input = P_in, simulated=True)
    metrics, mat_tuple, losses = e.run_experiment(spim, estim, exp, simulated=True)
    e.store_in_file(eval_dict["file"], metrics, mat_tuple, losses)

if __name__ == "__main__" : 
    print(sys.argv[1:])
    pos, est, eval = e.experiment_parser(sys.argv[1:])
    print(pos,est,eval)
    run_single_exp(pos,est,eval)
    # run_exp(a,default_opt)
    # params_dict = parse_args(sys.argv[1:],default_params)
    # run_exp(params_dict)


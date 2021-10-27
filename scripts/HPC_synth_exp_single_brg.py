from snmfem import estimators
import snmfem.experiments as e
import sys

cst_fixed_P = {"p0" : {"Fe" : 3.29160862e+00, "Al" : 1.57731834e+00, "Nd" : 0.0, "Mg" : 1.56346985e+01, "Sm" : 0.0, "Cu" : 5.92151062e+00, "Si" : 1.44423294e+01, "Ca" : 2.14438707e+00, "Lu" : 0.0, "Hf" : 1.98268294e+00, "O" : 1.67923235e+01, "Ti" : 0.0}, "p1" : {}, "p2" : {}}

def run_single_exp(pos_dict,est_dict,eval_dict, fixed_P = "None") :
    exp = e.build_exp(pos_dict, est_dict)
    spim, estim = e.simulation_quick_load(exp,P_type = fixed_P)
    metrics, mat_tuple, losses = e.run_experiment(spim, estim, exp, simulated=True)
    e.store_in_file(eval_dict["output_file"], metrics, mat_tuple, losses)

if __name__ == "__main__" : 
    print(sys.argv[1:])
    pos, est, eval = e.experiment_parser(sys.argv[1:-1])
    print(pos,est,eval)
    run_single_exp(pos,est,eval, fixed_P=cst_fixed_P)

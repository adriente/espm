import experiments as e
import sys
import json

def run_single_exp(pos_dict,est_dict,eval_dict) :
    exp = e.build_exp(pos_dict, est_dict)
    
    if eval_dict["fixed_W_json"] == "None" : 
        spim, estim = e.quick_load(experiment = exp,sim = eval_dict["simulated"], W_dict = None)
    else : 
        with open(eval_dict["fixed_W_json"],"r") as f :
            json_dict = json.load(f)
        spim, estim = e.quick_load(experiment = exp,sim = eval_dict["simulated"], W_dict = json_dict)
    metrics, mat_tuple, losses = e.run_experiment(spim, estim, exp, sim=eval_dict["simulated"])
    e.store_in_file(eval_dict["output_file"], metrics, mat_tuple, losses)

if __name__ == "__main__" : 
    print(sys.argv[1:])
    pos, est, eval = e.experiment_parser(sys.argv[1:])
    print(pos,est,eval)
    run_single_exp(pos,est,eval)



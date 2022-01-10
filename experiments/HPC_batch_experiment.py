import experiments as e
import sys
import json

def run_batch_exp(pos_dict,est_dict,eval_dict) :
    exp = e.build_exp(pos_dict, est_dict)
    txt = eval_dict["output_file"] + ".txt"
    npz = eval_dict["output_file"] + ".npz"
    
    if eval_dict["fixed_W_json"] == "None" :  
        metrics = e.run_several_experiments(exp, npz, n_samples = 6, W_dict = None)
    else : 
        with open(eval_dict["fixed_W_json"],"r") as f :
            json_dict = json.load(f)
        metrics = e.run_several_experiments(exp, npz, n_samples = 6, W_dict = json_dict)

    e.print_in_file(exp, metrics, txt)

if __name__ == "__main__" : 
    print(sys.argv[1:])
    pos, est, eval = e.experiment_parser(sys.argv[1:])
    print(pos,est,eval)
    run_batch_exp(pos,est,eval)


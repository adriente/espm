from pathlib import Path

BASE_PATH = Path(__file__).parent

RESULTS_PATH = BASE_PATH.parent / Path("results")
# Ensure that the folder DATASETS_PATH exists
RESULTS_PATH.mkdir(exist_ok=True, parents=True)

SCRIPT_CONFIG_PATH = BASE_PATH.parent / Path("scripts/config/")


POS_ARGS = {
    "input_file" : ["input_file",{"help" : "str : Name of the file containing the data."}],
    "method" : ["method",{"choices":  ["SmoothNMF"], "help" : "str : Name of the estimator for the decomposition"}],
    "g_type" : ["g_type", {"choices" : ["bremsstrahlung","no_brstlg", "identity"], "default" : "bremsstrahlung", "help" : "str : method to generate the G matrix from the metadata"}],
    "k" : ["k",{"type" : int,"help" : "int : expected number of phases"}]
}

EVAL_ARGS = {
    "output_file" : ["-of","--output_file",{"default" : "dump.npz", "help" : "str : Name of the npz file where the data are stored"}],
    "simulated" : ["-sim", "--simulated", {"action" : "store_true", "help" : "None : Activate to use the ground truth stored in the spim object. It will produce errors if no ground truth is available."}],
    "fixed_W_json" : ["-fwjs","--fixed_W_json",{"default" : "None", "help" : "str : Name of the json file containing the dictionnary to build fixed_W"}]
}

ESTIMATOR_ARGS = {
    # Common parameters
    "max_iter" : ["-mi","--max_iter",{"type" : int, "default" : 10000, "help" : "int : Max number of iterations for the algorithm"},None],
    "verbose" : ["-v","--verbose",{"action" : "store_false", "help" : "None : Activate to prevent display details about the algorithm"},None],
    "init" : ["-i","--init",{"choices" : ["random","nndsvd","nndsvda","nndsvdar","custom"],"default" : "nndsvdar", "help" : "str : Initialisation method"}, None],
    "random_state" : ["-rs","--random_state",{"type" : int,"default" : 42, "help" : "int : seed for the random initialisations"}, None],
    "tol" : ["-t", "--tol", {"type" : float, "default" : 1e-6, "help" : "float : Stopping criterion"}, None],
    "normalize" : ["-n", "--normalize", {"action" : "store_true", "help" : "None : Activate the normalization of the data, it is mostly useful for having stable values of lambda_L among datasets."}, None],

    # SNMFEM parameters
    "mu" : ["-mu","--mu",{"type" : float, "nargs" : "+","default" : 0.0, "help" : "float : strenght of the log regularization"},["SmoothNMF"]],
    "epsilon_reg" : ["-er","--epsilon_reg",{"type" : float, "default" : 1.0, "help" : "float : slope of the log regularization"}, ["SmoothNMF"]],
    "force_simplex" : ["-fs","--force_simplex",{"action" : "store_false", "help" : "None : Activate simplex constraint"},["SmoothNMF"]],
    "lambda_L" : ["-l","--lambda_L",{"type" : float, "default" : 0.0,"help":"float : strengh of the Laplacian reg"},["SmoothNMF"]],
    "l2" : ["-l2","--l2",{"action" : "store_true","help" : "None : Sets the loss function to frobenius when activated"}, ["NMF","SmoothNMF"]],
    "accelerate" : ["-acc","--accelerate",{"action" : "store_true","help" : "None : Sets the algorithm type to the accelerated one"}, ["SmoothNMF"]],
    "linesearch" : ["-ls","--linesearch",{"action" : "store_true","help" : "None : activates the linesearch for the accelerated algorithm"}, ["SmoothNMF"]],

    # Scikit parameters
    # "beta_loss" : ["-bl","--beta_loss",{"choices" : ["frobenius","kullback-leibler"], "default" : "frobenius", "help" : "str : Type of loss to be optimized"},["SKNMF"]],
    # "solver" : ["-s", "--solver", {"choices" : ["mu", "cd"], "default" : "mu", "help" : "str : Type of updates for the optimization"},["SKNMF"]],
    # "alpha" : ["-a", "--alpha",{"type" : float, "default" : 0.0, "help" : "float : strength of the regularization L1, L2 or both depending on the value of l1_ratio"}, ["SKNMF"]],
    # "l1_ratio" : ["-l1","--l1_ratio",{"type" : float, "default" : 1.0, "help" : "float : ratio between L1 and L2 regularization, 1.0 is full L1 regularization"},["SKNMF"]],
    # "regularization" : ["-r","--regularization",{"choices" : ["both","components","transformation"], "default" : "components", "help" : "str : determines on what the regularization is applied, W, H or both"},["SKNMF"]],

    # # MCR
    # "mcr_method" : ["-mm","--mcr_method",{"action" : "store_false","help" : "None : to be written"},["MCRLLM"]]
}

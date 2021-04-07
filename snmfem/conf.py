from pathlib import Path

# Path of the base
BASE_PATH = Path(__file__).parent

# Path of the db
DB_PATH = BASE_PATH / Path("Data/")

# Path of the generated datasets
DATASETS_PATH = BASE_PATH.parent / Path("generated_datasets")
# Ensure that the folder DATASETS_PATH exists
DATASETS_PATH.mkdir(exist_ok=True, parents=True)

RESULTS_PATH = BASE_PATH.parent / Path("results")
# Ensure that the folder DATASETS_PATH exists
RESULTS_PATH.mkdir(exist_ok=True, parents=True)

SCRIPT_CONFIG_PATH = BASE_PATH.parent / Path("scripts/config/")

log_shift = 1e-14
dicotomy_tol = 1e-4
seed_max = 4294967295

POS_ARGS = {
    "json" : ["json",{"help" : "str : Name of the json file containing info about the data"}],
    "method" : ["method",{"choices":  ["NMF","SmoothNMF","SKNMF","MCRLLM"], "help" : "str : Name of the estimator for the decomposition"}]
    # "k" : ["k",{"type" : int,"help" : "int : expected number of phases"}]
}

EVAL_ARGS = {
    "u" : ["-u", "--u", {"action" : "store_false", "help" : "None : Activate so that each result is uniquely matched with a ground truth."}],
    "file" : ["-f","--file",{"default" : "dump.npz", "help" : "str : Name of the npz file where the data are stored"}],
    "gather_files" : ["-gf","--gather_files",{"action" : "store_true","help" : "None : When relevant this parameter is used to gather txt results in one file"}]
}

ESTIMATOR_ARGS = {
    # Common parameters
    "max_iter" : ["-mi","--max_iter",{"type" : int, "default" : 1000, "help" : "int : Max number of iterations for the algorithm"},None],
    "verbose" : ["-v","--verbose",{"action" : "store_false", "help" : "None : Activate to prevent display details about the algorithm"},None],
    "init" : ["-i","--init",{"choices" : ["random","nndsvd","nndsvda","nndsvdar","custom","Kmeans","MBKmeans","NFindr","RobustNFindr","ATGP","FIPPI","nKmeans"],"default" : "random", "help" : "str : Initialisation method"}, None],
    "tol" : ["-t", "--tol", {"type" : float, "default" : 1e-6, "help" : "float : Stopping criterion"}, None],

    # SNMFEM parameters
    "mu" : ["-mu","--mu",{"type" : float,"default" : 0.0, "help" : "float : strenght of the log regularization"},["NMF","SmoothNMF"]],
    "skip_G" : ["-sG","--skip_G",{"action" : "store_true", "help" : "None : Activate G matrix"},["NMF","SmoothNMF"]],
    "force_simplex" : ["-fs","--force_simplex",{"action" : "store_false", "help" : "None : Activate simplex constraint"},["NMF","SmoothNMF"]],
    "lambda_L" : ["-l","--lambda_L",{"type" : float, "default" : 0.0,"help":"float : strengh of the Laplacian reg"},["SmoothNMF"]],
    "l2" : ["-l2","--l2",{"action" : "store_true","help" : "None : Sets the loss function to frobenius when activated"}, ["NMF","SmoothNMF"]],

    # Scikit parameters
    "beta_loss" : ["-bl","--beta_loss",{"choices" : ["frobenius","kullback-leibler"], "default" : "frobenius", "help" : "str : Type of loss to be optimized"},["SKNMF"]],
    "solver" : ["-s", "--solver", {"choices" : ["mu", "cd"], "default" : "mu", "help" : "str : Type of updates for the optimization"},["SKNMF"]],
    "alpha" : ["-a", "--alpha",{"type" : float, "default" : 0.0, "help" : "float : strength of the regularization L1, L2 or both depending on the value of l1_ratio"}, ["SKNMF"]],
    "l1_ratio" : ["-l1","--l1_ratio",{"type" : float, "default" : 1.0, "help" : "float : ratio between L1 and L2 regularization, 1.0 is full L1 regularization"},["SKNMF"]],
    "regularization" : ["-r","--regularization",{"choices" : ["both","components","transformation"], "default" : "components", "help" : "str : determines on what the regularization is applied, W, H or both"},["SKNMF"]],

    # MCR
    "mcr_method" : ["-mm","--mcr_method",{"action" : "store_false","help" : "None : to be written"},["MCRLLM"]]
}


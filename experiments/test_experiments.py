import experiments as exps
from pathlib import Path
from espm.conf import DATASETS_PATH
from espm.datasets import generate_dataset, spim
import shutil
from espm import estimators
from espm import datasets
import numpy as np

DATA_DICT = {
    "model_parameters" : {
        "e_offset" : 0.2,
        "e_size" : 1000,
        "e_scale" : 0.02,
        "width_slope" : 0.02,
        "width_intercept" : 0.065,
        "db_name" : "default_xrays.json",
        "E0" : 200,
        "params_dict" : {
            "Abs" : {
                "thickness" : 100.0e-7,
                "toa" : 35,
                "density" : 5
            },
            "Det" : "SDD_efficiency.txt"
        }
    },
    "N" : 40,
    "densities" : [1.3,1.6,1.9,1.0],
    "data_folder" : "test_experiment_data",
    "seed" : 42,
    "weight_type" : "laplacian",
    "shape_2d" : (30,40),
    "weights_params" : {},
    "model" : "EDXS",
    "phases_parameters" : [{"b0" : 5e-5,
                            "b1" : 3e-4,
                            "scale" : 3e-6,
                            "elements_dict" : {"Fe" : 0.54860348,
                                      "Pt" : 0.38286879,
                                      "Mo" : 0.03166235,
                                      "O" : 0.03686538}},
                            {"b0" : 7e-4,
                            "b1" : 5e-4,
                            "scale" : 3e-6,
                            "elements_dict" : {"Ca" : 0.54860348,
                                      "Si" : 0.38286879,
                                      "O" : 0.15166235}},
                            {"b0" : 3e-5,
                            "b1" : 5e-5,
                            "scale" : 3e-6,
                            "elements_dict" : {
                                "Cu" : 0.34,
                                "Mo" : 0.12,
                                "Au" : 0.54
                            }},
                             {"b0" : 1e-4,
                            "b1" : 5e-6,
                            "scale" : 3e-6,
                            "elements_dict" : {
                                "Fe" : 0.14,
                                "Ni" : 0.12,
                                "S" : 0.74
                            }}]
}

# base_path = BASE_PATH / Path("tests/ressources/") 
folder = DATASETS_PATH / Path(DATA_DICT["data_folder"])

def test_experiment_parser () : 
    inputs = ["file.hspy", "SmoothNMF", "bremsstrahlung", "3", "-mi" , '100', "--verbose", "--tol", '200', "-fwjs", "/path/to/json"]
    result_dicts = {'input_file': 'file.hspy', 'method': 'SmoothNMF', 'g_type': 'bremsstrahlung', 'k': 3}, {'max_iter': 100, 'verbose': False, 'init': 'nndsvdar', 'tol': 200, 'mu': 0.0, 'force_simplex': True, 'lambda_L': 0.0, 'l2': False, "accelerate" : False, "linesearch" : False, "normalize" : False, "random_state" : 42, "epsilon_reg" : 1.0}, {'output_file': 'dump.npz', "simulated" : False, "fixed_W_json" : "/path/to/json"}

    assert exps.experiment_parser(inputs) == result_dicts

def test_build_exp () : 
    inputs = ["file.hspy", "SmoothNMF", "bremsstrahlung", "3", "-mi" , '1000', "--verbose", "--tol", '200']
    pos_dict, est_dict, _ = exps.experiment_parser(inputs)
    exp = exps.build_exp(pos_dict,est_dict,name = "dummy")
    assert exp == {'g_type' : 'bremsstrahlung', 'input_file' : 'file.hspy', 'name' : 'dummy', 'method' : 'SmoothNMF', 
    'params' : {'force_simplex' : True, 'init' : 'nndsvdar', 'l2' : False, 'max_iter' : 1000, 'mu' : 0.0, 'n_components' : 3,
    'tol' : 200.0, 'verbose' : False, "normalize" : False, "random_state" : 42, 'epsilon_reg' : 1.0, 'accelerate' : False, 'lambda_L' : 0.0, 'linesearch' : False}}

def test_fill_exp_dict () : 
    exp = {"force_simplex" : True, "max_iter" : 1000, "linesearch" : True}
    filled_exp = exps.fill_exp_dict(exp)
    assert filled_exp == {'force_simplex' : True, "verbose" : True, 'init' : 'nndsvdar', 'l2' : False, 'max_iter' : 1000, 'mu' : 0.0, 'tol' : 1e-6, "lambda_L" : 0.0, "accelerate" : False, "linesearch" : True, "normalize" : False, "random_state" : 42, "epsilon_reg" : 1.0}

def test_quick_load () : 
    
    generate_dataset(seeds_range=1,**DATA_DICT)
    file = folder / Path("sample_0.hspy")
    experiment1 = {'g_type' : 'bremsstrahlung', 'input_file' : file , 'name' : 'dummy', 'method' : 'SmoothNMF', 
    'params' : {'force_simplex' : True, 'init' : 'random', 'l2' : False, 'max_iter' : 1000, 'mu' : 0.0, 'n_components' : 3,
    'tol' : 200.0, 'verbose' : False}}

    spim1, estimator1 = exps.quick_load(experiment1)
    assert callable(estimator1.G)
    assert isinstance(estimator1,estimators.SmoothNMF)
    assert estimator1.shape_2d == (30,40)
    assert estimator1.max_iter == 1000
    assert isinstance(spim1, datasets.EDS_espm)
    assert spim1.data.shape == (30,40,1000)
    assert estimator1.true_H.shape == (4,30*40)
    assert estimator1.true_D.shape == (1000,4)

    shutil.rmtree(folder)

def test_run_experiment () : 
    generate_dataset(seeds_range=1,**DATA_DICT)
    file = folder / Path("sample_0.hspy")
    experiment1 = {'g_type' : 'bremsstrahlung', 'input_file' : file , 'name' : 'dummy', 'method' : 'SmoothNMF', 
    'params' : {'force_simplex' : True, 'init' : 'random', 'l2' : False, 'max_iter' : 3, 'mu' : 0.0, 'n_components' : 4,
    'tol' : 0.0001, 'verbose' : False}}

    spim1, estimator1 = exps.quick_load(experiment1)
    metrics, (G, P, A), losses = exps.run_experiment(spim1, estimator1, experiment1, sim=True)

    assert G.shape == (1000, 13)
    assert P.shape == (13,4)
    assert A.shape == (4, 1200)
    assert losses.shape == (3,)
    assert losses.dtype.names == ('full_loss', 'KL_div_loss', 'log_reg_loss', 'Lapl_reg_loss','rel_W', 'rel_H', 'ang_p0', 'ang_p1', 'ang_p2', 'ang_p3', 'mse_p0', 'mse_p1', 'mse_p2', 'mse_p3', 'true_KL_loss', 'sigma_L')
    assert len(metrics) == 4
    assert len(metrics[0]) == 4
    assert len(metrics[1]) == 4
    assert len(metrics[2]) == 4
    assert np.array(metrics[2]).dtype == int
    assert np.all(np.diff(losses['full_loss'])) >= 0

    shutil.rmtree(folder)

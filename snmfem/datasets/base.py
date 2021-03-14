import snmfem.datasets.generate_data as gd
from snmfem import models
import numpy as np
from snmfem.conf import DB_PATH, DATASETS_PATH
from pathlib import Path
from tqdm import tqdm
from snmfem.datasets.generate_weights import generate_weights
from snmfem.conf import SCRIPT_CONFIG_PATH
import json

def generate_dataset_from_json(json_file):
    """Generate a toy dataset.
    
    # Input parameters
    folder_name: (str) 
    seeds: (List[Int])
    shape_2D: ([Int, Int]) shape of the image
    N: (Int) Average number of counts in one spectrum of the artificial data
    laplacian: (bool) use laplacian instead of random image
    """

    json_path = SCRIPT_CONFIG_PATH / Path(json_file)

    with open(json_path,"r") as f :
        json_dict = json.load(f)

    generate_dataset(**json_dict)


def generate_dataset(**kwargs):
    
    # Handle paramters
    data_folder = kwargs["data_folder"]
    weights_parameters = kwargs["weights_parameters"]
    N = kwargs["N"]
    model_parameters = kwargs["model_parameters"]
    densities = kwargs["densities"]
    g_parameters = kwargs["g_parameters"]
    phases_parameters = kwargs["phases_parameters"]

    seeds = range(kwargs["seeds_range"])
    Model = getattr(models, kwargs["model"]) 
    n_phases = len(phases_parameters)

    # Generate the phases
    model = Model(**model_parameters)
    model.generate_g_matr(**g_parameters)
    model.generate_phases(phases_parameters)
    phases = model.phases
    G = model.G

    # Ensure the data folder exist
    folder = DATASETS_PATH / Path(data_folder)
    folder.mkdir(exist_ok=True, parents=True)
    
    # list of densities which will give different total number of events per spectra
    for seed in tqdm(seeds):

        weights = generate_weights(**weights_parameters, n_phases=n_phases, seed=seed)
        spim = gd.ArtificialSpim(phases, densities, weights, G=G)
        filename = folder / Path("sample_{}".format(seed))
        spim.generate_spim_stochastic(N, seed=seed)
        spim.save(filename=filename)

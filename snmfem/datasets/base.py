import snmfem.datasets.generate_data as gd
from snmfem import models
import numpy as np
from snmfem.conf import DB_PATH, DATASETS_PATH
from pathlib import Path
from tqdm import tqdm
from snmfem.datasets.generate_weights import generate_weights
from snmfem.conf import SCRIPT_CONFIG_PATH
import json
import hyperspy.api as hs

def generate_spim(phases, weights, densities, N, seed=0,continuous = False):
        """
        Function to generate a noisy spectrum image based on an ideal one. For each pixel,
        local_N random spectroscopic events are drown from the probabilities given by the
        ideal spectra. local_N is drawn from a poisson distribution with average N weighted
        by the local density. The density of the matrix is set similarly as for
        generate_spim_deterministic.

        Inputs :
            N : (integer) average number of events
            seed : (integer) the seed for reproducible result. Default: 0.
            old : (boolean) use the old way to generate data. Default: False.
        """
        # Set the seed
        np.random.seed(seed)

        shape_2d = weights.shape[:2]
        spectral_len = phases.shape[1]
        phases = phases / np.sum(phases, axis=1, keepdims=True)
        # n D W A
        if continuous : 
        
            continuous_spim = N * (
                weights.reshape(-1, weights.shape[-1])
                @ (phases * np.expand_dims(densities, axis=1))
            ).reshape(*shape_2d, -1)
            return continuous_spim

        else : 
            stochastic_spim = np.zeros([*shape_2d, spectral_len])
            for k, w in enumerate(densities):
                # generating the spectroscopic events
                for i in range(shape_2d[0]):
                    for j in range(shape_2d[1]):
                        # Draw a local_N based on the local density
                        local_N = np.random.poisson(N * w * weights[i, j, k])
                        # draw local_N events from the ideal spectrum
                        counts = np.random.choice(
                            spectral_len, local_N, p=phases[k]
                        )
                        # Generate the spectrum based on the drawn events
                        hist = np.bincount(counts, minlength=spectral_len)
                        stochastic_spim[i, j] += hist
            return stochastic_spim

def save_generated_spim(filename, spim, model_params, g_params, phases_params, misc_params) : 
    s = hs.signals.Signal1D(spim)
    s.set_signal_type("EDS_TEM")
    s.axes_manager[-1].offset = model_params["e_offset"]
    s.axes_manager[-1].scale = model_params["e_scale"]

    s.set_microscope_parameters(beam_energy = model_params["E0"])
    s.add_elements(g_params["elements"])
    s.metadata.Sample.thickness = model_params["params_dict"]["Abs"]["thickness"]
    s.metadata.Sample.density = model_params["params_dict"]["Abs"]["density"]
    s.metadata.Acquisition_instrument.TEM.Detector.take_off_angle = model_params["params_dict"]["Abs"]["toa"]
    s.metadata.Acquisition_instrument.TEM.Detector.width_slope = model_params["width_slope"]
    s.metadata.Acquisition_instrument.TEM.Detector.width_intercept = model_params["width_intercept"]
    s.metadata.xray_db = model_params["db_name"]
    s.metadata.g_type = g_params["brstlg"]
    
    s.metadata.Truth = {}
    s.metadata.Truth.phases = phases_params
    s.metadata.Truth.Params = misc_params

    s.save(filename)

def generate_dataset(base_path=DATASETS_PATH, **kwargs):
    
    # Handle paramters
    data_folder = kwargs["data_folder"]
    model_parameters = kwargs["model_parameters"]
    g_parameters = kwargs["g_parameters"]
    phases_parameters = kwargs["phases_parameters"]
    misc_parameters = {
        "weight_type" : kwargs["weight_type"],
        "N" : kwargs["N"],
        "densities" : kwargs["densities"],
        "seeds" : range(kwargs["seeds_range"]),
        "model" : kwargs["model"]
    }

    Model = getattr(models, kwargs["model"]) 
    n_phases = len(phases_parameters)

    # Generate the phases
    model = Model(**model_parameters)
    model.generate_g_matr(**g_parameters)
    model.generate_phases(phases_parameters)
    phases = model.phases

    # Ensure the data folder exist
    folder = base_path / Path(data_folder)
    folder.mkdir(exist_ok=True, parents=True)
    
    # list of densities which will give different total number of events per spectra
    for seed in tqdm(range(kwargs["seeds_range"])):

        weights = generate_weights(kwargs["weight_type"],kwargs["shape_2d"], n_phases=n_phases, seed=seed)
        spim = generate_spim(phases, weights, kwargs["densities"], kwargs["N"], seed=seed)
        filename = folder / Path("sample_{}".format(seed))
        save_generated_spim(filename, spim, model_parameters, g_parameters, phases_parameters, misc_parameters)

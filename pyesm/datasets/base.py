from espm import models
import numpy as np
from espm.conf import DATASETS_PATH
from pathlib import Path
from tqdm import tqdm
from espm.datasets.generate_weights import generate_weights
import hyperspy.api as hs
from espm.datasets.generate_EDXS_phases import unique_elts

def generate_spim(phases, weights, densities, N, seed=0,continuous = False):
    """


    Inputs :
        N : (integer) average number of events
        seed : (integer) the seed for reproducible result. Default: 0.
    """
    # Set the seed
    np.random.seed(seed)

    shape_2d = weights.shape[:2]
    phases = phases / np.sum(phases, axis=1, keepdims=True)
    
    # n D W A
    continuous_spim = N * (
        weights.reshape(-1, weights.shape[-1])
        @ (phases * np.expand_dims(densities, axis=1))
    ).reshape(*shape_2d, -1)

    if continuous:
        return continuous_spim

    else :
        return np.random.poisson(continuous_spim)
    
        # # This is probably a very inefficient way to generate the data...
        # stochastic_spim = np.zeros([*shape_2d, spectral_len])
        # for k, w in enumerate(densities):
        #     # generating the spectroscopic events
        #     for i in range(shape_2d[0]):
        #         for j in range(shape_2d[1]):
        #             # Draw a local_N based on the local density
        #             local_N = np.random.poisson(N * w * weights[i, j, k])
        #             # draw local_N events from the ideal spectrum
        #             counts = np.random.choice(
        #                 spectral_len, local_N, p=phases[k]
        #             )
        #             # Generate the spectrum based on the drawn events
        #             hist = np.bincount(counts, minlength=spectral_len)
        #             stochastic_spim[i, j] += hist
        # return stochastic_spim


def save_generated_spim(filename, spim, phases, weights, **kwargs) : 
    # data2spim
    """Convert dataset to a hyperspy signal."""

    s = hs.signals.Signal1D(spim)
    s.set_signal_type("EDS_espm")
    model_params = kwargs["model_parameters"]
    misc_parameters = {
        "N" : kwargs["N"],
        "densities" : kwargs["densities"],
        "model" : kwargs["model"],
        "seed" : kwargs["seed"]
    }
    s.metadata.Truth = {}
    if "weights_params" in kwargs : 
        misc_parameters["weights_params"] = kwargs["weights_params"]
        s.metadata.Truth.Params = misc_parameters
        
    if "phases_parameters" in kwargs : 
        s.add_elements(elements = unique_elts(kwargs['phases_parameters']))
        s.metadata.Truth.phases = kwargs['phases_parameters']
    
    s.axes_manager[-1].offset = model_params["e_offset"]
    s.axes_manager[-1].scale = model_params["e_scale"]

    s.set_microscope_parameters(beam_energy = model_params["E0"])
    s.metadata.Sample = {}
    s.metadata.Sample.thickness = model_params["params_dict"]["Abs"]["thickness"]
    s.metadata.Sample.density = model_params["params_dict"]["Abs"]["density"]
    s.metadata.Acquisition_instrument.TEM.Detector.EDS.type = model_params["params_dict"]["Det"]
    s.metadata.Acquisition_instrument.TEM.Detector.EDS.take_off_angle = model_params["params_dict"]["Abs"]["toa"]
    s.metadata.Acquisition_instrument.TEM.Detector.EDS.width_slope = model_params["width_slope"]
    s.metadata.Acquisition_instrument.TEM.Detector.EDS.width_intercept = model_params["width_intercept"]
    s.metadata.xray_db = model_params["db_name"]
    
    s.metadata.Truth.Data = {}
    s.metadata.Truth.Data.phases = phases
    s.metadata.Truth.Data.weights = weights


    s.save(filename)

# def generate_spim_sample(args):
#     """Generate a spim sample."""
#     # Build a EM model such as EDXS or EELS

#     # Get G

#     # Generate the weights

#     # Generate the phases

#     # Generate the spim

#     # Build the output
#     sample["model_parameters"] = model_parameters # default empty, dict
#     sample["misc_parameters"] = misc_parameters # deault empty, dict
#     sample["shape_2d"] = shape_2d # list of length 2
#     sample["GW"] = phases # np.array
#     sample["H"] = weights # np.array
#     sample["X"] = spim # np.array
#     sample["Xdot"] = spim2
#     sample["G"] = G
#     return sample

def generate_dataset(base_path=DATASETS_PATH, seeds_range = 10, phases = None, weights = None, **kwargs):

    # for seed in range(seeds_range) :
    #     sample = generate_spim_sample(base_path, seed, phases, weights, **kwargs)
    #     spim = data2spim(sample)
    #     spim.save()
    
    # Handle paramters

    data_folder = kwargs["data_folder"]
    model_parameters = kwargs["model_parameters"]
    # Ensure the data folder exist
    folder = base_path / Path(data_folder)
    folder.mkdir(exist_ok=True, parents=True)
    
    # list of densities which will give different total number of events per spectra
    misc_parameters = {
        "N" : kwargs["N"],
        "densities" : kwargs["densities"],
        "model" : kwargs["model"],
        "seed" : kwargs["seed"]
    }
    if "weights_params" in kwargs : 
        misc_parameters["weights_params"] = kwargs["weights_params"]
    
    fixed_seed = misc_parameters["seed"]
    
    
    if "phases_parameters" in kwargs : 
        phases_parameters = kwargs["phases_parameters"]
        n_phases = len(phases_parameters)

    # Generate the phases  
    if phases is None :
        Model = getattr(models, kwargs["model"]) 
        model = Model(**model_parameters)
        model.generate_phases(phases_parameters) 
        g_phases = model.phases
        Ns = kwargs["N"] * np.array(kwargs["densities"])
        g_phases = g_phases*Ns[:,np.newaxis]
    else : 
        g_phases = phases/np.sum(phases, axis=1, keepdims=True)
        Ns = kwargs["N"] * np.array(kwargs["densities"])
        g_phases = g_phases*Ns[:,np.newaxis]
        n_phases = phases.shape[0]

    for seed in tqdm(range(misc_parameters["seed"],seeds_range+misc_parameters["seed"])):
        if weights is None : 
            g_weights = generate_weights(kwargs["weight_type"],kwargs["shape_2d"], **kwargs["weights_params"], n_phases=n_phases, seed=seed)
        else : 
            g_weights = weights
        assert np.allclose(np.sum(g_weights,axis = 2),1.0), "The input weights do not sum to one. Please modify it so that they sum to one along the 0th axis"
        spim = generate_spim(g_phases, g_weights, kwargs["densities"], kwargs["N"], seed=seed)
        filename = str(folder / Path("sample_{}".format(seed - fixed_seed)))
        misc_parameters.update({"seed" : seed})
        save_generated_spim(filename, spim, g_phases, g_weights, **kwargs)
        # save

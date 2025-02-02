r"""
The module :mod:`espm.datasets.base` implements the functions that combines a spatial distribution and associated spectra into a 3D dataset. It also implements the functions to convert the dataset into hyperspy compatible objects.
"""

from espm import models
import numpy as np
from espm.conf import DATASETS_PATH
from pathlib import Path
from tqdm import tqdm
import hyperspy.api as hs


def generate_spim(phases, weights, densities, N, seed=0,continuous = False):
    r"""
    Generate a noiseless spectrum image as tensor product of the phases and weights. Then, if asked for, a noisy spectrum image is generated by drawing from a Poisson distribution.

    The noiseless spectrum image is defined as:

    .. math::

        Y^{nl} = N  D \otimes ( Diag(d) A )

    where :math:`D` is the normalized phases, :math:`A` is the weights, :math:`d` is the density modifier and :math:`N` is the number of counts per pixel.
    
    To obtain the noisy spectrum image, the noiseless spectrum image is drawn from a Poisson distribution.

    Parameters
    ----------
    phases : array_like
        The phases of the model. Shape (n, spectral_len).
    weights : array_like
        The weights of the model. Shape (shape_2d[0], shape_2d[1], n).
    densities : array_like
        Density modifier of the phases. Shape (n,).
    N : int
        The number of counts per pixel.
    seed : int, optional
        Seed for the random number generator. The default is 0.
    continuous : bool, optional
        If True, the function returns a noiseless spectrum image. The default is False.
    
    Returns
    -------
    numpy.ndarray
        The spectrum image. Shape (shape_2d[0], shape_2d[1], spectral_len).

    Notes
    -----
    More details about the spectrum image generation can be found in the contribution: :cite:p:`teurtrie2023espm`.

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


def sample_to_EDSespm(sample,elements = []) : 
    # data2spim
    r"""Convert dataset to a custom hyperspy signal type called EDSespm containing the noisy spectrum image as data, the ground truth as metadata and other useful information.
    
    Parameters
    ----------
    sample : dict
        A dictionary containing the noisy spectrum image as data, the ground truth as metadata and other useful information. See :func:`espm.datasets.base.generate_spim_sample` for more details.
    elements : list, optional
        A list of the elements present in the sample. The default is [].

    Returns
    -------
    EDSespm
        The hyperspy compatible signal object of the :mod:`espm.eds_spim` module.   
    """

    s = hs.signals.Signal1D(sample["X"])
    s.set_signal_type("EDS_espm_Simulated")
    model_params = sample["model_parameters"]
    misc_parameters = sample['misc_parameters']
    s.metadata.Truth = {}
    
    s.axes_manager[-1].offset = model_params["e_offset"]
    s.axes_manager[-1].scale = model_params["e_scale"]
    s.axes_manager[-1].units = "keV"

    s.set_microscope_parameters(beam_energy = model_params["E0"])
    s.metadata.Sample = {}
    s.metadata.Sample.thickness = model_params["params_dict"]["Abs"]["thickness"]
    s.metadata.Sample.density = model_params["params_dict"]["Abs"]["density"]
    s.metadata.Sample.elements = elements
    s.metadata.Acquisition_instrument.TEM.Detector.EDS.type = model_params["params_dict"]["Det"]
    s.metadata.Acquisition_instrument.TEM.Detector.EDS.take_off_angle = model_params["params_dict"]["Abs"]["toa"]
    s.metadata.Acquisition_instrument.TEM.Detector.EDS.width_slope = model_params["width_slope"]
    s.metadata.Acquisition_instrument.TEM.Detector.EDS.width_intercept = model_params["width_intercept"]
    s.metadata.xray_db = model_params["db_name"]
    
    s.metadata.Truth.Data = {}
    s.metadata.Truth.Data.phases = sample["GW"]
    s.metadata.Truth.Data.weights = sample["H"]
    s.metadata.Truth.Data.misc_parameters = misc_parameters

    return s

def sample_to_Signal1D(sample) : 
    r"""
    Same as :func:`espm.datasets.base.sample_to_EDSespm` but for non-EDS data such as the toy dataset.
    """
    s = hs.signals.Signal1D(sample["X"])
    s.metadata.Truth = {}
    s.metadata.Truth.Data = {}
    s.metadata.Truth.Parameter = {}
    s.metadata.Truth.Data.phases = sample["GW"]
    s.metadata.Truth.Data.weights = sample["H"]
    s.metadata.Truth.Data.shape_2d = sample["shape_2d"]
    s.metadata.Truth.Data.G = sample["G"]
    s.metadata.Truth.Data.H_flat = sample["H_flat"]
    s.metadata.Truth.Parameters.misc_parameters = sample["misc_parameters"]
    s.metadata.Truth.Parameters.model_parameters = sample["model_parameters"]

    return s


def generate_spim_sample(phases, weights, model_params,misc_params, seed = 0,g_params = {}):
    r"""
    Generate a dictionary containing: the spectrum image (made with the weights and phases), the ground truth, the model parameters and the misc parameters.

    Parameters
    ----------
    phases : array_like
        The phases of the model. Shape (n, spectral_len).
    weights : array_like
        The weights of the model. Shape (shape_2d[0], shape_2d[1], n). The weights should sum to one along axis 2.
    model_params : dict
        The parameters of the model. For examples see the default parameters in espm.conf.
    misc_params : dict
        The misc parameters of the model. For examples see the default parameters in espm.conf.
    seed : int, optional
        The seed for the random number generator. The default is 0.
    g_params : dict, optional
        The parameters for the g matrix. The default is {}. Note that for EDXS data the g matrix is not used during the creation of the data.
    
    Returns
    -------
    sample : dict
        A dictionary containing the spectrum image, the ground truth, the model parameters and the misc parameters.
    """

    assert np.allclose(np.sum(weights,axis = 2),1.0), "The input weights do not sum to one. Please modify it so that they sum to one along axis 2"
    Xdot = generate_spim(phases, weights, misc_params["densities"], misc_params["N"], seed = seed,continuous=True)
    X = generate_spim(phases, weights, misc_params["densities"], misc_params["N"],seed= seed,continuous=False)
    shape_2d = weights.shape[:2]
    
    if misc_params["model"] == "EDXS" : 
        G = None
    else : 
        model_class = getattr(models, misc_params["model"])
        model = model_class(**model_params)
        model.generate_g_matr(**g_params)
        G = model.G

    normed_phases = phases/np.sum(phases, axis=1, keepdims=True)
    Ns = misc_params["N"] * np.array(misc_params["densities"])
    scaled_phases = normed_phases*Ns[:,np.newaxis]
    
    sample = {}
    sample["model_parameters"] = model_params
    sample["misc_parameters"] = misc_params
    sample["misc_parameters"]["seed"] = seed
    sample["shape_2d"] = shape_2d
    sample["GW"] = scaled_phases
    sample["H"] = weights
    sample["H_flat"] = weights.reshape(-1, weights.shape[-1])
    sample["X"] = X
    sample["Xdot"] = Xdot
    sample["G"] = G
    return sample

def generate_dataset(*args, base_path = DATASETS_PATH, sample_number = 10, base_seed = 0, elements = [], **kwargs): 
    r"""
    Generate a set of spectrum images files and save them in the generated dataset folder. Each spectrum image is saved in a separate file and was generated using a different seed.

    Parameters
    ----------
    base_path : str, optional
        The path to the folder where the samples will be saved. The default is DATASETS_PATH.
    sample_number : int, optional
        The number of samples to generate. The default is 10.
    base_seed : int, optional
        The seed used to generate the samples. The default is 0.

    Returns
    -------
    None.
    """
    for i in tqdm(range(sample_number)) : 
        sample = generate_spim_sample(*args, **kwargs, seed = base_seed + i)
        if sample["misc_parameters"]["model"] == "EDXS" : 
            hs_sig = sample_to_EDSespm(sample,elements = elements) 
        elif sample["misc_parameters"]["model"] == "Toy" : 
            hs_sig = sample_to_Signal1D(sample)  
            # ajouter save  
        else :
            raise ValueError("Unknown model. The implemented models are 'EDXS' and 'Toy") 
        hs_sig.save(base_path / Path(sample["misc_parameters"]["data_folder"]) / Path(f"sample_{i}.hspy"))


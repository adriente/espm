r"""
Absorption functions
--------------------

The :mod:`espm.models.absorption_edxs` module implements the functions to calculate the contribution of absorption in edx spectra.

"""

from scipy.interpolate import interp1d
from espm.utils import number_to_symbol_dict, atomic_to_weight_dict, approx_density
from espm.conf import HSPY_MAC
import numpy as np
from pathlib import Path
from espm.conf import DB_PATH

@number_to_symbol_dict
def absorption_coefficient (x,atomic_fraction = False,*,elements_dict = {"Si" : 1.0}) : 
    r"""
    Calculate the mass-absorption coefficent of a given composition at a certain energy or over a certain energy range.

    Parameters
    ----------
    x :
        :np.array 1D: Energy value or energy scale over which the mass-absorption coefficient is calculated.
    atomic_fraction :
        :boolean: If True, the atomic concentrations of elements_dict are converted in atomic weight fractions.
    elements_dict : 
        :dict: Dictionnary with chemical elements as keys and atomic weight fractions (or atomic concentrations) as values.

    Returns
    -------
    mass-absorption coefficient
        :np.array 1D: Calculated value of the mass-absorption coefficient on all the energy range.

    Notes
    -----
    The mass-absorption coefficients are calculated using the database of hyperspy
    """
    mu = np.zeros_like(x)
    if atomic_fraction : 
        elements_dict = atomic_to_weight_dict(elements_dict = elements_dict)

    sum_elts = sum(elements_dict.values())
    
    for key in elements_dict.keys() : 
        x_db = HSPY_MAC[key]["energies (keV)"]
        y_db = HSPY_MAC[key]["mass_absorption_coefficient (cm2/g)"]
        interp_func = interp1d(x_db,y_db,kind="cubic")
        mu += elements_dict[key]*interp_func(x)/sum_elts

    if len(elements_dict.keys()) == 0 :
        mu = 1 / np.power(x,3)

    return mu

def absorption_correction (x,thickness = 100e-7,toa = 90,density = None,atomic_fraction = False,*,elements_dict = {"Si" : 1.0},**kwargs) : 
    r"""
    Calculate the contribution of the absorption on the EDX spectrum for a thin slab of material with a given composition at a certain energy or over a certain energy range.

    Parameters
    ----------
    x :
        :np.array 1D: Energy value or energy scale over which the mass-absorption coefficient is calculated.
    thickness : 
        :float: Thickness of the material slab in meter. If the thickness is set to 0.0, the function will return 1.0.
    toa :
        :float: Take-off angle in degrees of the x-rays travelling from the sample to the x-ray detectors.
    density : 
        :float: Density of the material in g/m3 (to be checked). If None, an approximation of the density based on the atomic numbers of the elements of the materials is calculated.
    atomic_fraction :
        :boolean: If True, the atomic concentrations of elements_dict are converted in atomic weight fractions.
    elements_dict : 
        :dict: Dictionnary with chemical elements as keys and atomic weight fractions (or atomic concentrations) as values.

    Returns
    -------
    absorption correction
        :np.array 1D: Calculated value of the absorption correction on all the energy range.
    """
    mu = absorption_coefficient(x,atomic_fraction,elements_dict = elements_dict)
    rad_toa = np.deg2rad(toa)
    if density is None : 
        density = approx_density(atomic_fraction,elements_dict = elements_dict)
    
    if thickness == 0 : 
        return 1.0
    else : 
        chi = mu*density*thickness/np.sin(rad_toa)
        return (1 - np.exp(-chi))/chi

def absorption_mass_thickness(x,mass_thickness, toa = 90, atomic_fraction = True, *, elements_dict = {"Si" : 1.0}) : 
    r"""
    Calculate the contribution of the absorption of a mass-thickness map.

    Parameters
    ----------
    x :
        :np.array 1D: Energy value or energy scale over which the mass-absorption coefficient is calculated.
    mass_thickness: 
        :np.array 2D: Mass-thickness map. See the notes for details.
    toa :
        :float: Take-off angle in degrees of the x-rays travelling from the sample to the x-ray detectors.
    atomic_fraction :
        :boolean: If True, the atomic concentrations of elements_dict are converted in atomic weight fractions.
    elements_dict : 
        :dict: Dictionnary with chemical elements as keys and atomic weight fractions (or atomic concentrations) as values.

    Returns
    -------
    absorption correction
        :np.array 3D: Calculated value of the absorption correction on all the energy range for all pixels of the mass-thickness map.

    Notes
    -----
    The mass-thickness map can be extracted from an EDX spectrum image by calculating the concentration of an element for both K and L lines. Since there should be only one concentration for both lines, it is possible to determine the contribution of the absorption for each pixel and thus obtain the mass-thickness map.
    """
    mu = absorption_coefficient(x,atomic_fraction,elements_dict = elements_dict)
    rad_toa = np.deg2rad(toa)
    if type(mu) == float : 
        chi = mu*mass_thickness/np.sin(rad_toa)
    else : 
        chi = mu[:,np.newaxis]@((mass_thickness.reshape(-1)[:,np.newaxis]).T)/np.sin(rad_toa)
    return (1-np.exp(-chi))/chi

def det_efficiency_from_curve (x,filename,kind = "cubic") :
    r"""
    Interpolate a detection efficiency curve that is stored in ~/espm/tables.

    Parameters
    ----------
    x :
        :np.array 1D: Energy value or energy scale over which the mass-absorption coefficient is calculated.
    filename: 
        :string: Name of the file of the detection efficiency.
    kind :
        :string: Polynomial order of the interpolation.

    Returns
    -------
    Detection efficiency
        :np.array 1D: Interpolated detection efficiency.
    """
    array = np.loadtxt(DB_PATH / Path(filename))
    x_curve,y_curve = array[:,0], array[:,1]
    interp_func = interp1d(x_curve,y_curve,kind = kind)
    return interp_func(x)

def det_efficiency_layer (x, thickness = 100e-7, density = None, atomic_fraction = False, *, elements_dict = {"Si" : 1.0}) : 
    r"""
    Calculate the proportion of absorbed X-rays for one virtual layer of the detector.

    Parameters
    ----------
    x :
        :np.array 1D: Energy value or energy scale over which the mass-absorption coefficient is calculated.
    thickness : 
        :float: Thickness of the material slab in meter. If the thickness is set to 0.0, the function will return 1.0.
    density : 
        :float: Density of the material in g/m3 (to be checked). If None, an approximation of the density based on the atomic numbers of the elements of the materials is calculated.
    atomic_fraction :
        :boolean: If True, the atomic concentrations of elements_dict are converted in atomic weight fractions.
    elements_dict : 
        :dict: Dictionnary with chemical elements as keys and atomic weight fractions (or atomic concentrations) as values.

    Returns
    -------
    absorption
        :np.array 1D: Calculated absorption of one layer of the detector.

    Notes
    -----
    We assume that the X-rays arrive perpendicular to the detector.
    """
    mu = absorption_coefficient(x,atomic_fraction,elements_dict=elements_dict)
    if density is None : 
        density = approx_density(atomic_fraction,elements_dict = elements_dict)
    return mu*thickness*density

def det_efficiency (x,det_dict) :
    r"""
    Simulate the detection efficiency of an EDX detector.

    Parameters
    ----------
    x :
        :np.array 1D: Energy value or energy scale over which the mass-absorption coefficient is calculated.
    det_dict :
        :dict: See Examples for an example of the structure of the dict. One of the layer of the input dictionnary should be named "detection" to model a active layer of the detector.

    Returns
    -------
    efficiency
        :np.array 1D: Calculated detection efficiency based on the input layers model.

    Examples
    --------

    >>> from espm.models.absorption_edxs import det_efficiency
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> dict = {"detection" : {"thickness" : 10e-3, "density" : 2.3, "atomic_fraction" : False, "elements_dict" : {"Si" : 1.0}},
    >>> "dead_layer" : {"thickness" : 50e-7, "density" : 2.7, "atomic_fraction" : False, "elements_dict" : {"Al" : 1.0}}}
    >>> x = np.linspace(0.5,20,num = 1000)
    >>> efficiency = det_efficiency(x,dict)
    >>> plt.plot(x,efficiency)

    """
    efficiency = np.ones_like(x)
    for layer in det_dict : 
        if layer == "detection" : 
            efficiency *= 1 - np.exp(-det_efficiency_layer(x,**det_dict[layer]))
        else : 
            efficiency *= np.exp(-det_efficiency_layer(x,**det_dict[layer]))
    return efficiency

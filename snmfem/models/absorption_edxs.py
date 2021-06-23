from scipy.interpolate import interp1d
from snmfem.utils import number_to_symbol_dict, atomic_to_weight_dict, approx_density
from snmfem.conf import HSPY_MAC
import numpy as np
from pathlib import Path
from snmfem.conf import DB_PATH

@number_to_symbol_dict
def absorption_coefficient (x,atomic_fraction = False,*,elements_dict = {"Si" : 1.0}) : 
    mu = np.zeros_like(x)
    if atomic_fraction : 
        elements_dict = atomic_to_weight_dict(elements_dict = elements_dict)
    
    for key in elements_dict.keys() : 
        x_db = HSPY_MAC[key]["energies (keV)"]
        y_db = HSPY_MAC[key]["mass_absorption_coefficient (cm2/g)"]
        interp_func = interp1d(x_db,y_db,kind="cubic")
        mu += elements_dict[key]*interp_func(x)

    if len(elements_dict.keys()) == 0 :
        mu = 1 / np.power(x,3)

    return mu

def absorption_correction (x,thickness = 100e-7,toa = 90,density = None,atomic_fraction = False,*,elements_dict = {"Si" : 1.0}) : 
    mu = absorption_coefficient(x,atomic_fraction,elements_dict = elements_dict)
    rad_toa = np.deg2rad(toa)
    if density is None : 
        density = approx_density(atomic_fraction,elements_dict = elements_dict)
    
    if thickness == 0 : 
        return 1.0
    else : 
        chi = mu*density*thickness/np.sin(rad_toa)
        return (1 - np.exp(-chi))/chi

def det_efficiency_from_curve (x,filename,kind = "cubic") :
    array = np.loadtxt(DB_PATH / Path(filename))
    x_curve,y_curve = array[:,0], array[:,1]
    interp_func = interp1d(x_curve,y_curve,kind = kind)
    return interp_func(x)

def det_efficiency_layer (x, thickness = 100e-7, density = None, atomic_fraction = False, *, elements_dict = {"Si" : 1.0}) : 
    mu = absorption_coefficient(x,atomic_fraction,elements_dict=elements_dict)
    if density is None : 
        density = approx_density(atomic_fraction,elements_dict = elements_dict)
    return mu*thickness*density

def det_efficiency (x,det_dict) :
    efficiency = np.ones_like(x)
    for layer in det_dict : 
        if layer == "detection" : 
            efficiency *= 1 - np.exp(-det_efficiency_layer(x,**det_dict[layer]))
        else : 
            efficiency *= np.exp(-det_efficiency_layer(x,**det_dict[layer]))
    return efficiency

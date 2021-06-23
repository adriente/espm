import numpy as np
from scipy.special import erfc
from snmfem.models.absorption_edxs import det_efficiency_from_curve,det_efficiency,absorption_correction
    
def gaussian(x, mu, sigma):
    """
    docstring
    """
    return (
        1
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-np.power(x - mu, 2) / (2 * np.power(sigma, 2)))
    )

def read_lines_db (elt,db_dict) :
    energies = []
    cs = []
    for line in db_dict[str(elt)] : 
        energies.append(db_dict[str(elt)][line]["energy"])
        cs.append(db_dict[str(elt)][line]["cs"])
    return energies, cs

def read_compact_db (elt,db_dict) :
    energies = db_dict[str(elt)]["energies"]
    cs = db_dict[str(elt)]["cs"] 
    return energies, cs

def bremsstrahlung(x, b0, b1, b2):
    return b0 / x + b1 + b2 * x

def shelf(x, height, length):
    return height * erfc(x - length)

def continuum_xrays(x,params_dict,thickness = 10.0e-7, toa = 90.0, density = None,atomic_fraction = False,*,elements_dict = {}):
    """
    Computes the continuum X-ray based on the brstlg_pars set during init.
    The function is built in a way so that even an incomplete brstlg_pars dict is not a problem.
    """
    B = bremsstrahlung(
            x,
            params_dict["b0"],
            params_dict["b1"],
            params_dict["b2"],
        )
    
    A = absorption_correction(x,thickness,toa, density,atomic_fraction,elements_dict= elements_dict)
    
    if type(params_dict["Det"]) == str : 
        D = det_efficiency_from_curve(x,params_dict["Det"])
    else : 
        D = det_efficiency(x,params_dict["Det"])
    
    S = shelf(x, params_dict["height"], params_dict["length"])
    
    return B * A * D + S


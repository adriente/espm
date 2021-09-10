import numpy as np
from scipy.special import erfc
from snmfem.models.absorption_edxs import det_efficiency_from_curve,det_efficiency,absorption_correction
from snmfem.models import edxs as e
from collections import Counter
    
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

def chapman_bremsstrahlung(x, b0, b1, b2):
    return b0 / x + b1 + b2 * x
    
def lifshin_bremsstrahlung(x, b0, b1, E0 = 200):
    lbb0 = lifshin_bremsstrahlung_b0(x,b0,E0)
    lbb1 = lifshin_bremsstrahlung_b1(x,b1,E0)
    return lbb0 + lbb1

def lifshin_bremsstrahlung_b0(x, b0, E0 = 200):
    assert np.inf not in 1/x, "You have 0.0 in your energy scale. Retry with a cropped energy scale"
    return b0*(E0 - x)/x

def lifshin_bremsstrahlung_b1(x, b1, E0 = 200):
    assert np.inf not in 1/x, "You have 0.0 in your energy scale. Retry with a cropped energy scale"
    return b1*np.power((E0 - x),2)/x

def shelf(x, height, length):
    return height * erfc(x - length)

def continuum_xrays(x,params_dict={},b0= 0, b1 = 0, E0 = 200,*,elements_dict = {"Si" : 1.0} ):
    """
    Computes the continuum X-ray based on the brstlg_pars set during init.
    The function is built in a way so that even an incomplete brstlg_pars dict is not a problem.
    """
    if len(params_dict) == 0 : 
        return 0*x
    
    B = lifshin_bremsstrahlung(
            x,
            b0 = b0,
            b1 = b1,
            E0 = E0
        )
    
    A = absorption_correction(x,**params_dict["Abs"],elements_dict=elements_dict)
    
    if type(params_dict["Det"]) == str : 
        D = det_efficiency_from_curve(x,params_dict["Det"])
    else : 
        D = det_efficiency(x,params_dict["Det"])
    
    return B * A * D 

def G_bremsstrahlung(x,E0,params_dict,*,elements_dict = {}):
    """
    Computes the continuum X-ray based on the brstlg_pars set during init.
    The function is built in a way so that even an incomplete brstlg_pars dict is not a problem.
    """
    
    A = absorption_correction(x,**params_dict["Abs"],elements_dict= elements_dict)
    
    if type(params_dict["Det"]) == str : 
        D = det_efficiency_from_curve(x,params_dict["Det"])
    else : 
        D = det_efficiency(x,params_dict["Det"])

    B0 = A*D*lifshin_bremsstrahlung_b0(
            x,
            b0 = 1,
            E0 = E0
        )

    B1 = A*D*lifshin_bremsstrahlung_b1(
        x,
        b1=1,
        E0 = E0
    )

    B = np.vstack((B0,B1)).T
    
    return B
    
def elts_dict_from_P (part_P,elements_list) : 
    flat_P = np.sum(part_P,axis = 1)/np.sum(part_P)
    elements_dict = {}
    for i,elt in enumerate(elements_list) :
        elements_dict[elt] = flat_P[i] 
    return elements_dict

def elts_dict_from_dict_list (dict_list) : 
    unique_elts_dict = sum((Counter(x) for x in dict_list),Counter())
    sum_elts = sum(unique_elts_dict.values())
    for e in unique_elts_dict : 
        unique_elts_dict[e] /= sum_elts
    return unique_elts_dict

def update_bremsstrahlung (G,part_P,model_parameters,elements_list, norm = True) : 
    elts = elts_dict_from_P(part_P,elements_list)
    model = e.EDXS(**model_parameters)
    B = G_bremsstrahlung(model.x,model.E0,model.params_dict,elements_dict=elts)
    new_G = G.copy()
    new_G[:,-2:] = B
    if norm : 
        new_G = new_G / new_G.sum(axis = 0)
    return new_G




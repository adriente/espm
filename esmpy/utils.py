import numpy as np
from scipy.optimize import nnls
from esmpy.conf import SYMBOLS_PERIODIC_TABLE, NUMBER_PERIODIC_TABLE
import json
from hyperspy.misc.material import atomic_to_weight, density_of_mixture

def rescaled_DH(D,H) :
    """Rescale the matrices D and H such that the columns of H sums approximately to one."""
    _, p = H.shape
    o = np.ones((p,))
    s = np.linalg.lstsq(H.T, o, rcond=None)[0]
    if (s<=0).any():
        s = np.maximum(nnls(H.T, o)[0], 1e-10)
    D_rescale = D@np.diag(1/s)
    H_rescale = np.diag(s)@H
    return D_rescale, H_rescale

def bin_spim(data,n,m):
    """
    
    Take a 3D array of size (x,y,k) [px, py, e]
    Returns a 3D array of size (n,m,k) [new_px, new_py, e]
    """
    # return a matrix of shape (n,m,k)
    bs = data.shape[0]//n,data.shape[1]//m  # blocksize averaged over
    k = data.shape[2]
    return np.reshape(np.array([np.sum(data[k1*bs[0]:(k1+1)*bs[0],k2*bs[1]:(k2+1)*bs[1]],axis=(0,1)) for k1 in range(n) for k2 in range(m)]),(n,m,k))


def number_to_symbol_dict (func) : 
    """
    Decorator
    Takes a dict of elements (a.k.a chemical composition) with atomic numbers as keys (e.g. 26 for Fe)
    returns a dict of elements with symbols as keys (e.g. Fe for iron)
    """
    def inner(*args,**kwargs) : 
        elts_dict = kwargs["elements_dict"]
        new_dict = {}
        with open(NUMBER_PERIODIC_TABLE,"r") as f : 
            NPT = json.load(f)["table"]
        
        for key in elts_dict.keys() : 
            
            if is_symbol(key) : 
                new_dict[key] = elts_dict[key]
            
            elif is_number(key) : 
                new_dict[NPT[str(key)]["symbol"]] = elts_dict[key]
            
            else : 
                raise ValueError("Input has to be either atomic number, either chemical symbols")
        
        kwargs["elements_dict"] = new_dict
        return func(*args,**kwargs)

    return inner

def symbol_to_number_dict (func) : 
    """
    Decorator
    Takes a dict of elements (a.k.a chemical composition) with symbols as keys (e.g. Fe for iron)
    returns a dict of elements with atomic numbers as keys (e.g. 26 for iron)
    """
    def inner(*args,**kwargs) : 
        elts_dict = kwargs["elements_dict"]
        new_dict = {}
        with open(SYMBOLS_PERIODIC_TABLE,"r") as f : 
            SPT = json.load(f)["table"]
        for key in elts_dict.keys() : 
            
            if is_number(key) : 
                new_dict[int(key)] = elts_dict[key]
            
            elif is_symbol(key) : 
                new_dict[SPT[key]["number"]] = elts_dict[key]
            
            else : 
                raise ValueError("Input has to be either atomic number, either chemical symbols")
        
        kwargs["elements_dict"] = new_dict
        return func(*args,**kwargs)
    return inner

def symbol_to_number_list (func) : 
    """
    Decorator
    Takes a dict of elements (a.k.a chemical composition) with symbols as keys (e.g. Fe for iron)
    returns a dict of elements with atomic numbers as keys (e.g. 26 for iron)
    """
    def inner(*args,**kwargs) : 
        elts_list = kwargs["elements"]
        new_list = []
        with open(SYMBOLS_PERIODIC_TABLE,"r") as f : 
            SPT = json.load(f)["table"]
        for key in elts_list : 
            if is_number(key) : 
                new_list.append(int(key))
            elif is_symbol(key) : 
                new_list.append(SPT[key]["number"])
            else : 
                raise ValueError("Input has to be either atomic number, either chemical symbols")
        
        kwargs["elements"] = new_list
        return func(*args,**kwargs)
            
    return inner

def number_to_symbol_list (func) : 
    """
    Decorator
    Takes a dict of elements (a.k.a chemical composition) with symbols as keys (e.g. Fe for iron)
    returns a dict of elements with atomic numbers as keys (e.g. 26 for iron)
    """
    def inner(*args,**kwargs) : 
        elts_list = kwargs["elements"]
        new_list = []
        with open(NUMBER_PERIODIC_TABLE,"r") as f : 
            NPT = json.load(f)["table"]
        for key in elts_list : 
            if is_number(key) : 
                new_list.append(NPT[str(key)]["symbol"])
            elif is_symbol(key) : 
                new_list.append(key)
            else : 
                raise ValueError("Input has to be either atomic number, either chemical symbols")
        
        kwargs["elements"] = new_list
        return func(*args,**kwargs)
            
    return inner

@number_to_symbol_dict
def atomic_to_weight_dict (*,elements_dict = {}) :
    """
    Wrapper to the atomic_to_weight function of hyperspy. Takes a dict of chemical composition expressed in atomic fractions.
    Returns a dict of chemical composition expressed in atomic weight fratiom.
    """ 
    if len(elements_dict.keys()) == 0 : 
        return elements_dict
    else : 
        list_elts = []
        list_at = []
        for elt in elements_dict.keys() : 
            list_elts.append(elt)
            list_at.append(elements_dict[elt])
        list_wt = atomic_to_weight(list_at,list_elts)/100
        
        new_dict = {}
        for i, elt in enumerate(list_elts) : 
            new_dict[elt] = list_wt[i]
        
        return new_dict

@number_to_symbol_dict
def approx_density(atomic_fraction = False,*,elements_dict = {}) :
    """
    Wrapper to the density_of_mixture function of hyperspy. Takes a dict of chemical composition expressed in atomic weight fractions.
    Returns an approximated density.
    """  
    if len(elements_dict.keys()) == 0 : 
        return 1.0
    else : 
        list_elts = []
        list_wt = []
        if atomic_fraction : 
            elements_dict = atomic_to_weight_dict(elements_dict = elements_dict)
        
        for elt in elements_dict.keys() : 
            list_elts.append(elt)
            list_wt.append(elements_dict[elt])
        
        return density_of_mixture(list_wt,list_elts)

def arg_helper(params, d_params):
    """Check if all parameter of d_params are in params. If not, they are added to params."""
    for key in d_params.keys():
        params[key] = params.get(key, d_params[key])
        if isdict(params[key])  and isdict(d_params[key]):
            params[key] = arg_helper(params[key], d_params[key])
    check_keys(params, d_params)
    return params

def check_keys(params, d_params, upperkeys = '',toprint = True):
    """Check recursively if params and d_params if all the keys of params are in d_params."""
    keys = set(d_params.keys())
    for key in params.keys():
        if key not in keys:
            if toprint : 
                print('Warning! Optional argument: {}[\'{}\'] specified by user but not used'.format(upperkeys,key))
        else:
            if isdict(params[key]):
#                 if not(isdict(d_params[key])):
#                     print('Warning! Optional argument: {}{} is not supposed to be a dictionary'.format(upperkeys,key))
#                 else:
#                     check_keys(params[key],d_params[key],upperkeys=upperkeys+'[\'{}\']'.format(key))
                if isdict(d_params[key]):
                    if toprint :
                        check_keys(params[key],d_params[key],upperkeys=upperkeys+'[\'{}\']'.format(key))
    return True

def isdict(p):
    """Return True if the variable a dictionary."""
    return type(p) is dict

def is_symbol (i) :
    symb_list = symbol_list()
    if i in symb_list : 
        return True
    else : 
        return False

def is_number (i) :
    try : 
        int(i)
        return True
    except ValueError : 
        return False
    
def symbol_list () : 
    symbol_list = []
    with open(NUMBER_PERIODIC_TABLE,"r") as f : 
            NPT = json.load(f)["table"]
    for num in NPT.keys() : 
        symbol_list.append(NPT[num]["symbol"])
    return symbol_list

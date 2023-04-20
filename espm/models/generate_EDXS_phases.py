import numpy as np
from espm.models import EDXS
from copy import deepcopy
from espm.conf import DEFAULT_EDXS_PARAMS

# DEFAULT_ELTS = [{"b0" : 4.3298e-04 , "b1" : 6.7732e-02, "elements_dict" :  {"8": 1.0, "12": 0.51, "14": 0.61, "13": 0.07, "20": 0.04, "62": 0.02,
#                         "26": 0.028, "60": 0.002, "71": 0.003, "72": 0.003, "29": 0.02}},
#                     {"b0" : 1.3298e-04 , "b1" : 7.7732e-02, "elements_dict" : {"8": 0.54, "26": 0.15, "12": 1.0, "29": 0.038,
#                         "92": 0.0052, "60": 0.004, "31": 0.03, "71": 0.003}},
#                     {"b0" : 5.3298e-04 , "b1" : 3.7732e-02, "elements_dict" : {"8": 1.0, "14": 0.12, "13": 0.18, "20": 0.47,
#                         "62": 0.04, "26": 0.004, "60": 0.008, "72": 0.004, "29": 0.01}}]

def generate_brem_params (seed) : 
    r"""
    Generate random parameters for the Bremsstrahlung model with a scaling factor that somewhat resembles the real data.

    Parameters
    ----------
    seed : int
        Seed for the random number generator.

    Returns
    -------
    dict
        Dictionary containing the parameters for the Bremsstrahlung model.

    """
    np.random.seed(seed)
    b0 = float(np.random.rand(1)*1e-2)
    b1 = float(np.random.rand(1)*1e-1)
    return {"b0" : b0,"b1" : b1}

def generate_elts_dict (seed, nb_elements = 3) : 
    r"""
    Generate a random dictionary of elements and their relative abundance. The elements are limited to the range 6-82.

    Parameters
    ----------
    seed : int
        Seed for the random number generator.
    nb_elements : int, optional
        Number of elements to generate. The default is 3.

    Returns
    -------
    dict
        Dictionary containing the elements and their relative abundance.
    """
    np.random.seed(seed)
    elts = np.random.choice(np.arange(6,82,dtype = int),nb_elements,replace=False)
    elts = [str(elt) for elt in elts]
    frac = np.random.rand(nb_elements)
    elt_dict = dict(zip(elts,frac))
    return elt_dict

def unique_elts (dict_list) : 
    r"""
    Generate a list of unique chemical elements from a list of chemical elements dictionnaries.

    Parameters
    ----------
    dict_list : list
        List of dictionnaries containing the elements and their relative abundance.

    Returns
    -------
    list
        List of unique chemical elements.

    Examples
    --------
    >>> unique_elts([{"elements_dict" : {"18" : 0.2, "8" : 0.5, "12" : 0.3}}, {"elements_dict" : {"8" : 0.5, "26" : 0.3, "14" : 0.2}}])
    ['18', '8', '12', '26', '14']
    
    """
    full_elts_list = []
    for dict in dict_list : 
        for elt in dict["elements_dict"].keys() : 
            full_elts_list.append(elt)
    return list(set(full_elts_list))

def generate_random_phases(n_phases = 3, seed = 0):
    r"""
    Generate a list of phases of the EDXS model with random elemental compositions and Bremsstrahlung parameters. The model parameters are set to the default values.

    Parameters
    ----------
    n_phases : int, optional
        Number of phases to generate. The default is 3.
    seed : int, optional
        Seed for the random number generator. The default is 0.

    Returns
    -------
    np.array
        Array of phases of the EDXS model.
    """
    dict_list = []
    def_pars = deepcopy(DEFAULT_EDXS_PARAMS)
    model = EDXS(**def_pars)
    np.random.seed(seed)
    seed_list = np.random.choice(10000,size = n_phases,replace = False)
    for s in seed_list : 
        temp = generate_brem_params(s)
        # temp["seed"] = s
        elt_dict = generate_elts_dict(s)
        temp["elements_dict"] = elt_dict
        temp["scale"] = 1.0
        dict_list.append(temp.copy())     

    model.generate_phases(dict_list)
            
    return model.phases

def generate_modular_phases (elts_dicts = 3, brstlg_pars = None, scales = None, model_params = None, seed = 0) :
    r"""
    Generate an array of phases of the EDXS model with set model parameters, elemental compositions and bremsstrahlung parameters. 

    Parameters
    ----------
    elts_dicts : int or list, optional
        Number of phases to generate or list of dictionnaries containing the chemical compositions. The default is 3.
    brstlg_pars : list, optional
        List of dictionnaries containing the bremsstrahlung parameters. If the parameters are not set, they are generated randomly.
    scales : list, optional
        List of scaling factors for the phases. If the scales are not set, they are set to 1.0.
    model_params : dict, optional
        Dictionary containing the model parameters. If the parameters are not set, they are set to the default values. See the config file for the default values.
    seed : int, optional
        Seed for the random number generator. The default is 0.
    
    Returns
    -------
    np.array
        Array of phases of the EDXS model.
    
    Examples
    --------

    .. plot::
        :context: close-figs
        
        >>> from espm.models.generate_EDXS_phases import generate_modular_phases
        >>> import matplotlib.pyplot as plt
        >>> phases = generate_modular_phases(elts_dicts = [{"8": 0.5, "14": 0.2, "26": 0.3}, {"8" : 0.5, "23" : 0.3, "7" : 0.2}],
        ... brstlg_pars = [{"b0" : 0.03, "b1" : 0.01}, {"b0" : 0.002, "b1" : 0.0025}])
        >>> plt.plot(phases[0])
        >>> plt.plot(phases[1])

    """
    dict_list = []
    if type(elts_dicts) == int :
        n_phases = elts_dicts
    else : 
        n_phases = len(elts_dicts)
    if model_params is None :
        model_params = deepcopy(DEFAULT_EDXS_PARAMS)

    model = EDXS(**model_params)
    np.random.seed(seed)
    seed_list = np.random.choice(10000,size = n_phases,replace = False)
    for i,s in enumerate(seed_list) :
        if brstlg_pars is None :  
            temp = generate_brem_params(s)
        else : 
            temp = brstlg_pars[i]
        if type(elts_dicts) == int : 
            elt_dict = generate_elts_dict(s)
            temp["elements_dict"] = elt_dict
        else : 
            temp["elements_dict"] = elts_dicts[i]
        if scales is None : 
            temp["scale"] = 1.0
        else : 
            temp["scale"] = scales[i]
        dict_list.append(temp.copy())     

    model.generate_phases(dict_list)

    return model.phases

# def G_from_phases (dict_list) : 
#     def_pars = deepcopy(DEFAULT_SYNTHETIC_DATA_DICT["model_parameters"])
#     model = EDXS(**def_pars)
#     elts_list = unique_elts(dict_list)
#     model.generate_g_matr(brstlg = True,**def_pars["Abs"],elements_list=elts_list)
#     return model.G

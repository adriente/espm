import numpy as np
from snmfem.models import EDXS
from copy import deepcopy

DEFAULT_ELTS = [{"b0" : 4.3298e-09 , "b1" : 6.7732e-07, "elements_dict" :  {"8": 1.0, "12": 0.51, "14": 0.61, "13": 0.07, "20": 0.04, "62": 0.02,
                        "26": 0.028, "60": 0.002, "71": 0.003, "72": 0.003, "29": 0.02}},
                    {"b0" : 1.3298e-09 , "b1" : 7.7732e-07, "elements_dict" : {"8": 0.54, "26": 0.15, "12": 1.0, "29": 0.038,
                        "92": 0.0052, "60": 0.004, "31": 0.03, "71": 0.003}},
                    {"b0" : 5.3298e-09 , "b1" : 3.7732e-07, "elements_dict" : {"8": 1.0, "14": 0.12, "13": 0.18, "20": 0.47,
                        "62": 0.04, "26": 0.004, "60": 0.008, "72": 0.004, "29": 0.01}}]

DEFAULT_PARAMS = {
    "e_offset" : 0.200,
    "e_size" : 1980,
    "e_scale" : 0.01,
    "E0" : 200,
    "db_name" : "default_xrays.json",
    "params_dict": {
"Abs" : {
            "thickness" : 100e-7,
            "density" : None,
            "toa" : 22,
            "atomic_fraction" : True 
    }
    }
    
    
}

def generate_brem_params (seed) : 
    np.random.seed(seed)
    b0 = float(np.random.rand(1)*1e-7)
    b1 = float(np.random.rand(1)*1e-7)
    return {"b0" : b0,"b1" : b1}

def generate_elts_dict (seed, nb_elements = 3) : 
    np.random.seed(seed)
    elts = np.random.choice(np.arange(6,82,dtype = int),nb_elements,replace=False)
    elts = [str(elt) for elt in elts]
    frac = np.random.rand(nb_elements)
    elt_dict = dict(zip(elts,frac))
    return elt_dict

def unique_elts (dict_list) : 
    full_elts_list = []
    for dict in dict_list : 
        for elt in dict["elements_dict"].keys() : 
            full_elts_list.append(elt)
    return list(set(full_elts_list))

def generate_random_phases(n_phases = 3, seed = 0):
    dict_list = []
    def_pars = deepcopy(DEFAULT_PARAMS)
    model = EDXS(**def_pars)
    if seed == 0 and n_phases==3 :
        for i in range(3) : 
            # temp.update(def_pars["params_dict"])
            temp = DEFAULT_ELTS[i]
            temp["scale"] = 0.1
            dict_list.append(temp.copy())
    
    else:
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
            
    return model.phases, dict_list

def G_from_phases (dict_list) : 
    def_pars = deepcopy(DEFAULT_PARAMS)
    model = EDXS(**def_pars)
    elts_list = unique_elts(dict_list)
    model.generate_g_matr(brstlg = True,**def_pars["Abs"],elements_list=elts_list)
    return model.G

from espm import models
import numpy as np
import espm.models.EDXS_function as ef
from espm.conf import DEFAULT_EDXS_PARAMS
from espm.utils import arg_helper
import lmfit as lm
import re

def make_partial_xy (list_energies,spectrum,x) : 
    # This function is a bit messy but it works. There is probably a better way to do it though. 
    # Create partial x and y based on 
    part_x=np.linspace(list_energies[0][0],list_energies[0][1],num=round((list_energies[0][1]-list_energies[0][0])/spectrum.axes_manager[0].scale))
    part_y=spectrum.isig[list_energies[0][0]:list_energies[0][1]].data
    for elt in list_energies[1:] :
        part_x=np.append(part_x,np.linspace(elt[0],elt[1],num=round((elt[1]-elt[0])/spectrum.axes_manager[0].scale)))
        part_y=np.append(part_y,spectrum.isig[elt[0]:elt[1]].data)

    #Construction of a boolean array for display purposes
    list_boola=[]
    for i in range(len(list_energies)) :
        list_boola.append(np.empty_like(x))

    for i in range(len(list_energies)) :
        for j in range(x.shape[0]) :
            if np.logical_and(x[j]<list_energies[i][1],x[j]>list_energies[i][0]) :
                list_boola[i][j]=True
            else :
                list_boola[i][j] = False

    sum_boola=np.zeros_like(x,dtype=bool)
    for elt in list_boola :
        sum_boola+=elt.astype(bool)
    
    
    return part_x, part_y, sum_boola

def residual(pars,x,data = None) : 
    kwargs = params_to_ndict(pars)
    kwargs["params_dict"] = arg_helper(kwargs["params_dict"],DEFAULT_EDXS_PARAMS)
    
    model = ef.continuum_xrays(x,**kwargs)    
    
    if data is None : 
        return model
    
    return model - data

def params_to_ndict(params) :
    out_dict = {}
    for key, value in params.valuesdict().items() : 
        key_list = key.split("__")
        temp = value
        for key in key_list[::-1] : 
            temp = {key : temp}
        out_dict = custom_update(out_dict,temp)
    return out_dict

def custom_update(d, u):
    for k, v in u.items():
        if type(v) == type({}):
            d[k] = custom_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def ndict_to_params (input_dict) :
    params = lm.Parameters()
    un_nested = nested_dict_iter(input_dict)
    for par in un_nested :
        elt_match = re.match(r"(.*__)([A-Z][a-z]?)$",par[0])
        if type(par[1]) == str : 
            pass
        else :
            # The elements are not learned (they represent too many variables)
            #All positive expect for b1 which is negative
            if elt_match : 
                params.add(par[0],value = par[1], vary=False)
            elif par[0] == "b0" : 
                params.add(par[0],value = par[1])
            else : 
                params.add(par[0],value = par[1], min = 0)
    return params


def nested_dict_iter(nested, prefix = ""):
    for key, value in nested.items():
        if type(value) == type({}):
            new_prefix = prefix + key + "__"
            yield from nested_dict_iter(value,new_prefix)
        else:
            new_key = prefix + key
            yield new_key, value


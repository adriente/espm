import hyperspy.api as hs
from snmfem import estimators
from snmfem import models
import json
import snmfem.conf as conf
import numpy as np
from pathlib import Path

def load_filename (json_file) : 
    with open(json_file,"r") as f : 
        file = conf.DATASETS_PATH / Path(json.load(f)["data_file"])
    return file

def load_hs_data(filename) : 
    spim = hs.load(filename)
    try : 
        offset = spim.axes_manager[2].offset
        scale = spim.axes_manager[2].scale
        size = spim.axes_manager[2].size
        X = spim.data.astype("float32")
        nx, ny, ns = X.shape
        Xflat = X.transpose([2,0,1]).reshape(ns, nx*ny)
        shape_2d = nx,ny
        return Xflat, shape_2d, offset, scale, size
    except AttributeError : 
        print("You need to define the offset, scale or size of the energy axis.")
    except IndexError : 
        print("Your data need to be a spectrum image with 2 navigation axes and 1 signal axis")

def build_analysis(pos_dict,est_dict) : 
    d = {}
    d["name"] = pos_dict["method"]
    d["method"] = pos_dict["method"]
    estimator_dict = {}
    for key in est_dict.keys() : 
        if conf.ESTIMATOR_ARGS[key][3] is None : 
            estimator_dict[key] = est_dict[key]
        elif pos_dict["method"] in conf.ESTIMATOR_ARGS[key][3] : 
            estimator_dict[key] = est_dict[key]
        else : 
            pass
    estimator_dict["n_components"] = pos_dict["k"]
    d["params"] = {**estimator_dict}
    return d

def build_model(json_file,offset,scale,size) :
    with open(json_file,"r") as f : 
        d = json.load(f)
    Model = getattr(models,d["model"])
    model = Model(offset,size,scale,**d["model_parameters"])
    model.generate_g_matr(**d["g_parameters"])
    
    return model.G

def run_analysis(Xflat, G, experiment, shape_2d = None) : 
    
    Estimator = getattr(estimators, experiment["method"]) 

    estimator = Estimator(**experiment["params"])
    
    estimator.fit(Xflat,G=G,shape_2d = shape_2d)
    
    G = estimator.G_
    P = estimator.P_
    A = estimator.A_

    losses = estimator.get_losses()
    return G, P, A, losses

def save_results (filename,G,P,A, losses) :
    d = {}
    d["D"] = G@P
    d["P"] = P
    d["A"] = A
    d["losses"] = losses
    np.savez(filename, **d)

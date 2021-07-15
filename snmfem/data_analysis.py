import hyperspy.api as hs
from snmfem import estimators
from snmfem import models
import json
import snmfem.conf as conf
import numpy as np
from pathlib import Path
from hyperspy.misc.eds.utils import take_off_angle
from snmfem.models.edxs import G_EDXS

def load_filename (json_file) : 
    with open(json_file,"r") as f : 
        file = conf.DATASETS_PATH / Path(json.load(f)["data_file"])
    return file

def set_analysis_parameters (s,beam_energy = 200, azimuth_angle = 0.0, elevation_angle = 22.0, tilt_stage = 0.0, elements = [], thickness = 200e-7, density = 3.5, detector_type = "SDD_efficiency.txt") :
    s.set_signal_type("EDS_TEM")
    s.set_microscope_parameters(beam_energy = beam_energy, azimuth_angle = azimuth_angle, elevation_angle = elevation_angle,tilt_stage = tilt_stage)
    s.add_elements(elements)
    s.metadata.Sample.thickness = thickness
    s.metadata.Sample.density = density
    s.metadata.Acquisition_instrument.TEM.Detector.type = detector_type

    s.metadata.Acquisition_instrument.TEM.Detector.take_off_angle = take_off_angle(tilt_stage,azimuth_angle,elevation_angle)

def get_analysis_parameters (spim) : 
    if spim is str() : 
        s = hs.load(data)
    else : 
        s = spim
    mod_pars = {}
    mod_pars["E0"] = s.metadata.Acquisition_instrument.TEM.beam_energy
    try :
        mod_pars["e_offset"] = s.axes_manager[-1].offset
        assert mod_pars["e_offset"] > 0.1, "The energy scale can't include 0, it will produce errors elsewhere. Please crop your data."
        mod_pars["e_scale"] = s.axes_manager[-1].scale
        mod_pars["e_size"] = s.axes_manager[-1].size
        mod_pars["db_name"] = "default_xrays.json"
        mod_pars["width_slope"] = 0.01
        mod_pars["width_intercept"] = 0.065
    
        pars_dict = {}
        pars_dict["Abs"] = {
            "thickness" : s.metadata.Sample.thickness,
            "toa" : s.metadata.Acquisition_instrument.TEM.Detector.take_off_angle,
            "density" : s.metadata.Sample.density
        }
        try : 
            pars_dict["Det"] = s.metadata.Acquisition_instrument.TEM.Detector.type.as_dictionary()
        except AttributeError : 
            pars_dict["Det"] = s.metadata.Acquisition_instrument.TEM.Detector.type

        mod_pars["params_dict"] = pars_dict
        
        g_pars = {}
        g_pars["brstlg"] = True
        g_pars["elements_list"] = s.metadata.Sample.elements

    except AttributeError : 
        print("You need to define the relevant parameters for the analysis. Use the set_analysis function.")

    return mod_pars, g_pars

def get_shape_2d (s) :   
    shape_2d = s.axes_manager[0].size, s.axes_manager[1].size
    return shape_2d

def get_data (s) : 
    try : 
        X = s.data.astype("float64")
        Xflat = X.transpose([2,0,1]).reshape(X.shape[2], X.shape[0]*X.shape[1])
    except IndexError : 
        print("Your data need to be a spectrum image with 2 navigation axes and 1 signal axis")
    return Xflat

def load_hs_data(data) : 
    if data is str() : 
        spim = hs.load(data)
    else : 
        spim = data
    mod_pars, g_pars = get_analysis_parameters(spim)
    Xflat = get_data(spim)
    shape_2d = get_shape_2d(spim)
    return Xflat, shape_2d, mod_pars, g_pars

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

def build_model(mod_pars, g_pars, G_func = True, model = "EDXS") :
    if G_func : 
        return G_EDXS
    else : 
        Model = getattr(models,model)
        model = Model(**mod_pars)
        model.generate_g_matr(**g_pars)
        
        return model.G

def run_analysis(Xflat, G, experiment, shape_2d = None,mod_pars = None,g_pars = None, ind_list = None) : 
    
    Estimator = getattr(estimators, experiment["method"]) 

    estimator = Estimator(**experiment["params"])
    
    estimator.fit(Xflat,G=G,shape_2d = shape_2d,model_params = mod_pars, g_params = g_pars, fixed_A_inds = ind_list)
    
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

def get_ROI_ranges(spim,ROI) : 
    scale_h, offset_h = spim.axes_manager[0].scale, spim.axes_manager[0].offset
    scale_v, offset_v = spim.axes_manager[1].scale, spim.axes_manager[1].offset
    left = int((ROI.left - offset_h)/scale_h)
    right = int((ROI.right - offset_h)/scale_h)
    top = int((ROI.top - offset_v)/scale_v)
    bottom = int((ROI.bottom - offset_v)/scale_v)
    return left, right, top, bottom

def num_indices (ij, size_h) : 
    return ij[0] + size_h*ij[1]

def build_index_list (left,right,top,bottom, size_h) : 
    ind_list = []
    for i in range(left,right) : 
        for j in range(top,bottom) : 
            ind_list.append(num_indices((i,j),size_h))
    ind_list.sort()
    return ind_list

from  hyperspy._signals.signal1d import Signal1D
from numpy.core.fromnumeric import reshape
from snmfem.models import EDXS
from snmfem import models
from snmfem.models.edxs import G_EDXS
from snmfem.datasets.generate_weights import generate_weights
from hyperspy.misc.eds.utils import take_off_angle
from snmfem.utils import number_to_symbol_list
import numpy as np
# import hyperspy.extensions as e

# # Temporary fix

# e.ALL_EXTENSIONS["signals"]["EDXSsnmfem"] = {'signal_type': 'EDXSsnmfem',
#    'signal_dimension': 1,
#    'dtype': 'real',
#    'lazy': False,
#    'module': 'snmfem.datasets.spim'}

class EDXSsnmfem (Signal1D) : 
        # self.shape_2d = self.axes_manager[0].size, self.axes_manager[1].size
        # self.model_parameters, self.g_parameters = self.get_metadata()
        # self.phases_parameters, self.misc_parameters = self.get_truth()
        # self.phases, self.weights = self.build_truth()
    def __init__ (self,*args,**kwargs) : 
        super().__init__(*args,**kwargs)
        self.shape_2d_ = None
        self.phases_ = None
        self.weights_ = None
        self.X_ = None
        self.Xdot_ = None

    @property
    def shape_2d (self) : 
        if self.shape_2d_ is None : 
            self.shape_2d_ = self.axes_manager[1].size, self.axes_manager[0].size
        return self.shape_2d_

    @property
    def X (self) :
        if self.X_ is None :  
            shape = self.axes_manager[1].size, self.axes_manager[0].size, self.axes_manager[2].size
            self.X_ = self.data.reshape((shape[0]*shape[1], shape[2])).T
        return self.X_

    @property
    def Xdot (self) : 
        if self.Xdot_ is None : 
            try : 
                self.Xdot_ = self.metadata.Truth.Params.N * self.phases @ np.diag(self.metadata.Truth.Params.densities) @ self.weights
            except AttributeError : 
                print("This dataset contains no ground truth. Nothing was done.")
        return self.Xdot_


    @property
    def weights (self) : 
        if self.weights_ is None : 
            self.weights_ = self.build_ground_truth()[1]
        return self.weights_

    @property
    def phases (self) : 
        if self.phases_ is None : 
            self.phases_ = self.build_ground_truth()[0]
        return self.phases_

    @property
    def phases_2d (self) : 
        if self.phases_2d_ is None : 
            self.phases_2d_ = self.build_ground_truth(reshape = False)[0]
        return self.phases_2d_

    @property
    def weights_2d (self) : 
        if self.phases_2d_ is None : 
            self.phases_2d_ = self.build_ground_truth(reshape = False)[1]
        return self.phases_2d_

    def build_ground_truth(self,reshape = True) : 
        mod_pars = get_metadata(self)
        phases_pars, misc_pars = get_truth(self)
        phases, weights = build_truth(self, mod_pars, phases_pars, misc_pars)
        if reshape : 
            phases = phases.T
            weights = weights.reshape((weights.shape[0]*weights.shape[1], weights.shape[2])).T
        return phases, weights

    def build_G(self, problem_type = "bremsstrahlung") :
        self.problem_type = problem_type
        g_pars = {"g_type" : problem_type, "elements" : self.metadata.Sample.elements}
        mod_pars = get_metadata(self)
        if problem_type == "bremsstrahlung" : 
            G = self.update_G
        else : 
            G = build_G(mod_pars,g_pars)
        
        self.G = G
        
        return self.G

    def get_P(self) : 
        D = self.get_decomposition_factors().data.T
        if self.problem_type == "bremsstrahlung" : 
            G = self.G()
        else : 
            G = self.G
        P = np.abs(np.linalg.lstsq(G, D,rcond=None)[0])
        return P

    def update_G(self, part_P=None, G=None):
        model_params = get_metadata(self)
        g_params = {"g_type" : self.problem_type, "elements" : self.metadata.Sample.elements}
        G = G_EDXS(model_params, g_params, part_P=part_P, G=G)
        return G

    def set_analysis_parameters (self,beam_energy = 200, azimuth_angle = 0.0, elevation_angle = 22.0, tilt_stage = 0.0, elements = [], thickness = 200e-7, density = 3.5, detector_type = "SDD_efficiency.txt", width_slope = 0.01, width_intercept = 0.065, xray_db = "default_xrays.json") :
        self.set_microscope_parameters(beam_energy = beam_energy, azimuth_angle = azimuth_angle, elevation_angle = elevation_angle,tilt_stage = tilt_stage)
        self.add_elements(elements = elements)
        self.metadata.Sample.thickness = thickness
        self.metadata.Sample.density = density
        try : 
            del self.metadata.Acquisition_instrument.TEM.Detector.EDS.type
        except AttributeError : 
            pass
        self.metadata.Acquisition_instrument.TEM.Detector.EDS.type = detector_type
        self.metadata.Acquisition_instrument.TEM.Detector.EDS.width_slope = width_slope
        self.metadata.Acquisition_instrument.TEM.Detector.EDS.width_intercept = width_intercept
        self.metadata.xray_db = xray_db

        self.metadata.Acquisition_instrument.TEM.Detector.EDS.take_off_angle = take_off_angle(tilt_stage,azimuth_angle,elevation_angle)
    
    @number_to_symbol_list
    def add_elements(self, *, elements = []) :
        try : 
            self.metadata.Sample.elements = elements
        except AttributeError :
            self.metadata.Sample = {}
            self.metadata.Sample.elements = elements

    def set_microscope_parameters(self, beam_energy = 200, azimuth_angle = 0.0, elevation_angle = 22.0,tilt_stage = 0.0) : 
        self.metadata.Acquisition_instrument = {}
        self.metadata.Acquisition_instrument.TEM = {}
        self.metadata.Acquisition_instrument.TEM.Stage = {}
        self.metadata.Acquisition_instrument.TEM.Detector = {}
        self.metadata.Acquisition_instrument.TEM.Detector.EDS = {}
        self.metadata.Acquisition_instrument.TEM.beam_energy = beam_energy
        self.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha = tilt_stage
        self.metadata.Acquisition_instrument.TEM.Detector.EDS.azimuth_angle = azimuth_angle
        self.metadata.Acquisition_instrument.TEM.Detector.EDS.elevation_angle = elevation_angle
        self.metadata.Acquisition_instrument.TEM.Detector.EDS.energy_resolution_MnKa = 130.0

######################
# Axiliary functions #
######################

def get_metadata(spim) : 
    mod_pars = {}
    try :
        mod_pars["E0"] = spim.metadata.Acquisition_instrument.TEM.beam_energy
        mod_pars["e_offset"] = spim.axes_manager[-1].offset
        assert mod_pars["e_offset"] > 0.01, "The energy scale can't include 0, it will produce errors elsewhere. Please crop your data."
        mod_pars["e_scale"] = spim.axes_manager[-1].scale
        mod_pars["e_size"] = spim.axes_manager[-1].size
        mod_pars["db_name"] = spim.metadata.xray_db
        mod_pars["width_slope"] = spim.metadata.Acquisition_instrument.TEM.Detector.width_slope
        mod_pars["width_intercept"] = spim.metadata.Acquisition_instrument.TEM.Detector.width_intercept
    
        pars_dict = {}
        pars_dict["Abs"] = {
            "thickness" : spim.metadata.Sample.thickness,
            "toa" : spim.metadata.Acquisition_instrument.TEM.Detector.take_off_angle,
            "density" : spim.metadata.Sample.density
        }
        try : 
            pars_dict["Det"] = spim.metadata.Acquisition_instrument.TEM.Detector.type.as_dictionary()
        except AttributeError : 
            pars_dict["Det"] = spim.metadata.Acquisition_instrument.TEM.Detector.type

        mod_pars["params_dict"] = pars_dict

    except AttributeError : 
        print("You need to define the relevant parameters for the analysis. Use the set_analysis_parameters function.")

    return mod_pars

def build_truth(spim, model_params, phases_params, misc_params ) : 
    # axes manager does not respect the original order of the input data shape
    shape_2d = [spim.axes_manager[1].size , spim.axes_manager[0].size]
    if (not(phases_params is None)) and (not(misc_params is None)) :
        Model = getattr(models,misc_params["model"])
        model = Model(**model_params)
        model.generate_phases(phases_params)
        phases = model.phases
        phases = phases / np.sum(phases, axis=1, keepdims=True)
        weights = generate_weights(misc_params["weight_type"],shape_2d, len(phases_params),misc_params["seed"], **misc_params["weights_params"])
        return phases, weights
    else : 
        print("This dataset contains no ground truth. Nothing was done.")
        return None, None

    
def get_truth(spim) : 
    try : 
        phases_pars = spim.metadata.Truth.phases
        misc_pars = spim.metadata.Truth.Params.as_dictionary()
    except AttributeError : 
        print("This dataset contain no ground truth.")
        return None, None

    return phases_pars, misc_pars

def build_G(model_params, g_params) : 
    model = EDXS(**model_params)
    model.generate_g_matr(**g_params)
    return model.G



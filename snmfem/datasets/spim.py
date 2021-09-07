from  hyperspy._signals.signal1d import Signal1D
from snmfem.models import EDXS
from snmfem import models
from snmfem.models.edxs import G_EDXS
from snmfem.datasets.generate_weights import generate_weights
from hyperspy.misc.eds.utils import take_off_angle
from snmfem.utils import number_to_symbol_list
import numpy as np

class EDXSsnmfem (Signal1D) : 
        # self.shape_2d = self.axes_manager[0].size, self.axes_manager[1].size
        # self.model_parameters, self.g_parameters = self.get_metadata()
        # self.phases_parameters, self.misc_parameters = self.get_truth()
        # self.phases, self.weights = self.build_truth()

    def extract_truth(self,reshape = True) : 
        mod_pars = get_metadata(self)
        phases_pars, misc_pars = get_truth(self)
        phases, weights = build_truth(self, mod_pars, phases_pars, misc_pars)
        self.phases, self.weights = phases, weights
        if reshape : 
            phases = phases.T
            weights = weights.reshape((weights.shape[0]*weights.shape[1], weights.shape[2])).T
        return phases, weights

    def extract_params(self, g_type = "bremsstrahlung") :
        self.g_type = g_type
        g_pars = {"g_type" : g_type, "elements" : self.metadata.Sample.elements}
        mod_pars = get_metadata(self)
        if g_type == "bremsstrahlung" : 
            G = self.update_G
        else : 
            G = build_G(mod_pars,g_pars)
        
        self.G = G
        
        return self.G, (self.axes_manager[1].size, self.axes_manager[0].size)

    def get_P(self) : 
        D = self.get_decomposition_factors().data.T
        if self.g_type == "bremsstrahlung" : 
            G = self.G()
        else : 
            G = self.G
        P = np.abs(np.linalg.lstsq(G, D,rcond=None)[0])
        return P

    def update_G(self, part_P=None, G=None):
        model_params = get_metadata(self)
        g_params = {"g_type" : self.g_type, "elements" : self.metadata.Sample.elements}
        G = G_EDXS(model_params, g_params, part_P=part_P, G=G)
        return G
    
    # def plot() : 
    #     pass

    def get_Xflat(self) : 
        shape = self.axes_manager[1].size, self.axes_manager[0].size, self.axes_manager[2].size
        return self.data.reshape((shape[0]*shape[1], shape[2])).T

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
        print("You need to define the relevant parameters for the analysis. Use the set_analysis function.")

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



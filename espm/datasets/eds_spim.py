r"""
The module :mod:`espm.eds_spim` implements the :class:`EDS_espm` class, which is a subclass of the :class:`hyperspy.signals.Signal1D` class.
The main purpose of this class is to provide an easy and clean interface between the hyperspy framework and the espm package: 
- The metadata are organised to correspond as much as possible to the typical metadata that can be found in hyperspy EDS_TEM object.
- The machine learning algorithms of espm can be easily applied to the :class:`EDS_espm` object using the standard hyperspy decomposition method. See the notebooks for examples.
- The :class:`EDS_espm` provides a convinient way to:
    - get the results of :class:`espm.estimators.NMFEstimator`
    - access ground truth in case of simulated data
    - estimate best binning thanks to the method developed by G. Obozinski, N. Perraudin and M. Martinez Ruts.
    - set fixed W for the :class:`espm.estimators.NMFEstimator` decomposition
"""

from  hyperspy.signals import Signal1D
from espm.models import EDXS
from exspy.misc.eds.utils import take_off_angle
from espm.utils import number_to_symbol_list, get_explained_intensity_W, arg_helper
import numpy as np
from espm.estimators import NMFEstimator
import re
import warnings
from prettytable import PrettyTable, MSWORD_FRIENDLY
from tqdm import tqdm

class EDS_espm(Signal1D) : 

    def __init__ (self,*args,**kwargs) : 
        super().__init__(*args,**kwargs)
        self.shape_2d_ = None
        self._phases = None
        self._maps = None
        self._X = None
        self._Xdot = None
        self._maps_2d = None
        self.G_ = None
        self.model_ = None
        self.custom_init_ = None

    ##############
    # Properties #
    ##############

    @property
    def custom_init (self) :
        r"""
        Boolean setting whether using the custom_init (see espm.models.EDXS) or not.
        If True, the custom_init will be used to initialise the decomposition.
        If False, the default initialisation will be used.
        If None, the custom_init will be set to False.
        """
        return self.custom_init_
    
    @custom_init.setter
    def custom_init (self, value) :
        self.custom_init_ = value

    @property
    def shape_2d (self) : 
        r"""
        Shape of the data in the spatial dimension.
        """
        if self.shape_2d_ is None : 
            self.shape_2d_ = self.axes_manager[1].size, self.axes_manager[0].size
        return self.shape_2d_

    @property
    def X (self) :
        r"""
        The data in the form of a 2D array of shape (n_samples, n_features).
        """
        if self._X is None :  
            shape = self.axes_manager[1].size, self.axes_manager[0].size, self.axes_manager[2].size
            self._X = self.data.reshape((shape[0]*shape[1], shape[2])).T
        return self._X

    @property
    def Xdot (self) : 
        r"""
        The ground truth in the form of a 3D array of shape (shape_2d[0],shape_2d[1],n_features), if available.
        """
        if self._Xdot is None : 
            try : 
                self._Xdot = self.phases @ self.maps
            except AttributeError : 
                print("This dataset contains no ground truth. Nothing was done.")
        return self._Xdot


    @property
    def maps (self) :
        r"""
        Ground truth of the spatial distribution of the phases in the form of a 3D array of shape (shape_2d[0],shape_2d[1],n_phases), if available.
        """ 
        if self._maps is None : 
            self._maps = self.build_ground_truth()[1]
        return self._maps

    @property
    def phases (self) : 
        r"""
        Ground truth of the spectra of the phases in the form of a 2D array of shape (n_phases,n_features), if available.
        """
        if self._phases is None : 
            self._phases = self.build_ground_truth()[0]
        return self._phases

    @property
    def maps_2d (self) : 
        r"""
        Ground truth of the spatial distribution of the phases in the form of a 2D array of shape (shape_2d[0]*shape_2d[1],n_phases), if available.
        """
        if self._maps_2d is None : 
            self._maps_2d = self.build_ground_truth(reshape = False)[1]
        return self._maps_2d
    
    @property
    def model(self) :
        r"""
        The :class:`espm.models.EDXS` model corresponding to the metadata of the :class:`EDS_espm` object.
        """ 
        if self.model_ is None : 
            mod_pars = get_metadata(self)
            self.model_ = EDXS(**mod_pars, custom_init=self.custom_init_)
        return self.model_
    
    @property
    def G(self) :
        r"""
        The G matrix of the :class:`espm.models.EDXS` model corresponding to the metadata of the :class:`EDS_espm` object.
        """
        if self.G_ is None : 
            try : 
                if self.problem_type == "identity" :
                    return None
            except AttributeError :
                warnings.warn("You did not used the build_G method to build the G matrix. In ESpM-NMF, an idenity matrix will be used for decomposition")
                return None
        return self.G_
    
    ######################################
    # Modelling and simulation functions #
    ######################################

    def build_ground_truth(self,reshape = True) : 
        r"""
        Get the ground truth stored in the metadata of the :class:`EDS_espm` object, if available. The reshape arguments can be used to get the ground truth in a form easier to use for machine learning algorithms.

        Parameters
        ----------
        reshape : bool, optional
            If False, the ground truth is returned in the form of a 3D array of shape (shape_2d[0],shape_2d[1],n_phases) and a 2D array of shape (n_phases,n_features).
        
        Returns
        -------
        phases : numpy.ndarray
            The ground truth of the spectra of the phases.
        weights : numpy.ndarray
            The ground truth of the spatial distribution of the phases.
        """
        if "phases" in self.metadata.Truth.Data : 
            phases = self.metadata.Truth.Data.phases
            weights = self.metadata.Truth.Data.weights
            if reshape : 
                phases = phases.T
                weights = weights.reshape((weights.shape[0]*weights.shape[1], weights.shape[2])).T
        else : 
            raise AttributeError("There is no ground truth contained in this dataset")
        return phases, weights

    def build_G(self, problem_type = "bremsstrahlung",*, elements_dict = {}) :
        r"""
        Build the G matrix of the :class:`espm.models.EDXS` model corresponding to the metadata of the :class:`EDS_espm` object and stores it as an attribute.

        Parameters
        ----------
        problem_type : str, optional
            Determines the type of the G matrix to build. It can be "bremsstrahlung", "no_brstlg" or "identity". The parameters correspond to:
                - "bremsstrahlung" : the G matrix is a callable with both characteristic X-rays and a bremsstrahlung model.
                - "no_brstlg" : the G matrix is a matrix with only characteristic X-rays.
                - "identity" : the G matrix is None which is equivalent to an identity matrix for espm functions.
        elements_dict : dict, optional
            Dictionary containing atomic numbers and a corresponding cut-off energies. It is used to separate the characteristic X-rays of the given elements into two energies ranges and assign them each a column in the G matrix instead of having one column per element.
            For example elements_dict = {"26",3.0} will separate the characteristic X-rays of the element Fe into two energies ranges and assign them each a column in the G matrix. This is useful to circumvent issues with the absorption.
        Returns
        -------
        G : None or numpy.ndarray or callable
            The G matrix of the :class:`espm.models.EDXS` model corresponding to the metadata of the :class:`EDS_espm` object.
        """
        self.problem_type = problem_type
        self.separated_lines = elements_dict
        g_pars = {"g_type" : problem_type, "elements" : self.metadata.Sample.elements, "elements_dict" : elements_dict}

        self.model.generate_g_matr(**g_pars)
        self.G_ = self.model.G
                
        # Storing the model parameters in the metadata so that the decomposition does not erase them
        # Indeed the decomposition re-creates a new object of the same class when it is called
        self.metadata.EDS_model= {}
        self.metadata.EDS_model.problem_type = problem_type
        self.metadata.EDS_model.separated_lines = elements_dict
        self.metadata.EDS_model.elements = self.model.model_elts
        self.metadata.EDS_model.norm = self.model.norm

    ############################
    # Metadata and model setup #
    ############################

    # So far we don't use exspy methods of setting metadata. 
    # 1) I don't really remember why but we can't inherit from exspy signals (maybe it was a decomposition thing)
    # TODO : Check the inheritance from exspy signals
    # 2) exspy sets only the metadata that are explicitly filled in the argument of the metadata setting functions.
    # It does not really work for us, since all the arguments are required to build the G matrix and the model.

    def set_analysis_parameters(self,
                                beam_energy = 200,
                                azimuth_angle = 0.0,
                                elevation_angle = 22.0,
                                tilt_stage = 0.0,
                                elements = [],
                                thickness = 200e-7,
                                density = 3.5,
                                detector_type = "SDD_efficiency.txt",
                                width_slope = 0.01,
                                width_intercept = 0.065,
                                xray_db = "default_xrays.json") :
        r"""
        Helper function to set the metadata of the :class:`EDS_espm` object. Be careful, it will overwrite the metadata of the object.

        Parameters
        ----------
        beam_energy : float, optional
            The energy of the electron beam in keV.
        azimuth_angle : float, optional
            The azimuth angle of the EDS detector in degrees.
        elevation_angle : float, optional
            The elevation angle of the EDS detector in degrees.
        tilt_stage : float, optional
            The tilt angle of the sample stage in degrees (usually it correspond to alpha on FEI instruments).
        elements : list, optional
            List of the elements to be used in the analysis.
        thickness : float, optional
            The thickness of the sample in centimeters.
        density : float, optional
            The density of the sample in g/cm^3.
        detector_type : str, optional
            The type of the detector. It is either the name of a text file containing the efficiency or a modeling dictionary.
            (see the espm.models.absorption_edxs.det_efficiency function)
        width_slope : float, optional
            The slope of the linear fit of the detector width as a function of the energy.
        width_intercept : float, optional
            The intercept of the linear fit of the detector width as a function of the energy.
        xray_db : str, optional
            The name of the X-ray emission cross-section database to be used. The default tables are avalaible in the espm/tables folder. Additional tables can be generated by emtables.
        """
        self.set_microscope_parameters(beam_energy = beam_energy,
                                       azimuth_angle = azimuth_angle,
                                       elevation_angle = elevation_angle,
                                       tilt_stage = tilt_stage,
                                       detector_type = detector_type)
        self.set_elements(elements = elements)
        self.set_additional_parameters(thickness = thickness,
                                       density = density,
                                       width_slope = width_slope,
                                       width_intercept = width_intercept,
                                       xray_db = xray_db)
        
    def add_analysis_parameters(self,
                                beam_energy = 200,
                                azimuth_angle = 0.0,
                                elevation_angle = 22.0,
                                tilt_stage = 0.0,
                                elements = [],
                                thickness = 200e-7,
                                density = 3.5,
                                detector_type = "SDD_efficiency.txt",
                                width_slope = 0.01,
                                width_intercept = 0.065,
                                xray_db = "default_xrays.json") :
        r"""
        Helper function to set the metadata of the :class:`EDS_espm` object without overwriting the existing metadata.

        Parameters
        ----------
        beam_energy : float, optional
            The energy of the electron beam in keV.
        azimuth_angle : float, optional
            The azimuth angle of the EDS detector in degrees.
        elevation_angle : float, optional
            The elevation angle of the EDS detector in degrees.
        tilt_stage : float, optional
            The tilt angle of the sample stage in degrees (usually it correspond to alpha on FEI instruments).
        elements : list, optional
            List of the elements to be used in the analysis.
        thickness : float, optional
            The thickness of the sample in centimeters.
        density : float, optional
            The density of the sample in g/cm^3.
        detector_type : str, optional
            The type of the detector. It is either the name of a text file containing the efficiency or a modeling dictionary.
            (see the espm.models.absorption_edxs.det_efficiency function)
        width_slope : float, optional
            The slope of the linear fit of the detector width as a function of the energy.
        width_intercept : float, optional
            The intercept of the linear fit of the detector width as a function of the energy.
        xray_db : str, optional
            The name of the X-ray emission cross-section database to be used. The default tables are avalaible in the espm/tables folder. Additional tables can be generated by emtables.
        """
        self.add_microscope_parameters(beam_energy = beam_energy,
                                       azimuth_angle = azimuth_angle,
                                       elevation_angle = elevation_angle,
                                       tilt_stage = tilt_stage,
                                       detector_type = detector_type)
        self.add_elements(elements = elements)
        self.add_additional_parameters(thickness = thickness,
                                        density = density,
                                        width_slope = width_slope,
                                        width_intercept = width_intercept,
                                        xray_db = xray_db)
        
    def set_additional_parameters(self,thickness = 200e-7, density = 3.5, width_slope = 0.01, width_intercept = 0.065, xray_db = "default_xrays.json") : 
        r"""
        Helper function to set the metadata that are specific to the :mod:`espm` package so that it does not overwrite experimental metadata.
        See the documentation of the :func:`set_analysis_parameters` function for the meaning of the parameters.
        """
        try : 
            self.metadata.Sample.thickness = thickness
            self.metadata.Sample.density = density
        except AttributeError : 
            self.metadata.Sample = {}
            self.metadata.Sample.thickness = thickness
            self.metadata.Sample.density = density

        self.metadata.xray_db = xray_db
    
        try : 
            tilt_stage = self.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha
            azimuth_angle = self.metadata.Acquisition_instrument.TEM.Detector.EDS.azimuth_angle
            elevation_angle = self.metadata.Acquisition_instrument.TEM.Detector.EDS.elevation_angle
            current_dict = self.metadata.Acquisition_instrument.TEM.as_dictionary()
            input_dict = {'Detector' : {'EDS' :
                                            {
                                            'width_slope' : width_slope,
                                            'width_intercept' : width_intercept,
                                            'take_off_angle' : take_off_angle(tilt_stage,azimuth_angle,elevation_angle)
                                            }
                                       }
                         }            
            new_dict = arg_helper(input_dict,current_dict, replace = True)
            self.metadata.Acquisition_instrument.TEM = new_dict
        except AttributeError :
            print('The microscope parameters have not been set yet. Please use the set_microscope_parameters method first.') 
        
    def add_additional_parameters(self,thickness = 200e-7, density = 3.5, width_slope = 0.01, width_intercept = 0.065, xray_db = "default_xrays.json") : 
        r"""
        Helper function to set the metadata that are specific to the :mod:`espm` package so that it does not overwrite experimental metadata.
        See the documentation of the :func:`set_analysis_parameters` function for the meaning of the parameters.
        """
        try : 
            self.metadata.Sample
        except AttributeError : 
            self.metadata.Sample = {}
            self.metadata.Sample.thickness = thickness
            self.metadata.Sample.density = density

        try :
            self.metadata.Sample.thickness
        except AttributeError :
            self.metadata.Sample.thickness = thickness

        try :
            self.metadata.Sample.density
        except AttributeError :
            self.metadata.Sample.density = density

        try : 
            self.metadata.xray_db
        except AttributeError :
            self.metadata.xray_db = xray_db

        try : 
            tilt_stage = self.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha
            azimuth_angle = self.metadata.Acquisition_instrument.TEM.Detector.EDS.azimuth_angle
            elevation_angle = self.metadata.Acquisition_instrument.TEM.Detector.EDS.elevation_angle
            current_dict = self.metadata.Acquisition_instrument.TEM.as_dictionary()
            input_dict = {'Detector' : {'EDS' :
                                            {
                                            'width_slope' : width_slope,
                                            'width_intercept' : width_intercept,
                                            'take_off_angle' : take_off_angle(tilt_stage,azimuth_angle,elevation_angle)
                                            }
                                       }
                         }            
            new_dict = arg_helper(input_dict,current_dict, replace = False)
            self.metadata.Acquisition_instrument.TEM = new_dict
        except AttributeError :
            print('The microscope parameters have not been set yet. Please use the set_microscope_parameters method first.')    


    @number_to_symbol_list
    def add_elements(self, *, elements = []) :
        r"""
        Add elements to the existing list of elements in the metadata.

        Parameters
        ----------
        elements : list, optional
            List of the elements to be added to the existing list of elements in the metadata.
        
        Note :     
        The input can be either atomic numbers or chemical symbols.
        For example elements = ['Si', '26', 22] will result in ['Si', 'Fe', 'Ti'].
        """
        try : 
            previous_elements = list(self.metadata.Sample.elements)
            elements = list(set(elements + previous_elements))
            self.metadata.Sample.elements = elements
        except AttributeError :
            self.metadata.Sample = {}
            self.metadata.Sample.elements = elements

    @number_to_symbol_list
    def set_elements(self, *, elements = []) :
        r"""
        Replace elements with the input list in the metadata.

        Parameters
        ----------
        elements : list, optional
            List of the elements to be added in the metadata. 

        Note :     
        The input can be either atomic numbers or chemical symbols.
        For example elements = ['Si', '26', 22] will result in ['Si', 'Fe', 'Ti'].
        """
        try : 
            self.metadata.Sample.elements = elements
        except AttributeError :
            self.metadata.Sample = {}
            self.metadata.Sample.elements = elements

    def set_microscope_parameters(self, beam_energy = 200, azimuth_angle = 0.0, elevation_angle = 22.0,tilt_stage = 0.0, detector_type = "SDD_efficiency.txt") : 
        r"""
        Helper function to set the microscope parameters of the :class:`EDS_espm` object. Be careful, it will overwrite the microscope parameters of the object.
        See the documentation of the :func:`set_analysis_parameters` function for the meaning of the parameters.
        """
        try : 
            current_dict = self.metadata.Acquisition_instrument.as_dictionary()
            input_dict = {'TEM' :
                          {'Stage' : {'tilt_alpha' : tilt_stage},
                           'Detector' : {'EDS' :
                                         {'azimuth_angle' : azimuth_angle,
                                          'elevation_angle' : elevation_angle,
                                          'energy_resolution_MnKa' : 130.0}
                                        },
                          'beam_energy' : beam_energy
                          }
                         }
            new_dict = arg_helper(input_dict,current_dict)
            self.metadata.Acquisition_instrument = new_dict
            self.metadata.Acquisition_instrument.TEM.Detector.EDS.type = detector_type
        except AttributeError :
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
            self.metadata.Acquisition_instrument.TEM.Detector.EDS.type = detector_type

    def add_microscope_parameters(self, beam_energy = 200, azimuth_angle = 0.0, elevation_angle = 22.0,tilt_stage = 0.0, detector_type = "SDD_efficiency.txt") : 
        r"""
        Helper function to set the microscope parameters of the :class:`EDS_espm` object. Be careful, it will overwrite the microscope parameters of the object.
        See the documentation of the :func:`set_analysis_parameters` function for the meaning of the parameters.
        """
        try : 
            current_dict = self.metadata.Acquisition_instrument.as_dictionary()
            input_dict = {'TEM' :
                          {'Stage' : {'tilt_alpha' : tilt_stage},
                           'Detector' : {'EDS' :
                                         {'azimuth_angle' : azimuth_angle,
                                          'elevation_angle' : elevation_angle,
                                          'energy_resolution_MnKa' : 130.0}
                                        },
                          'beam_energy' : beam_energy
                          }
                         }
            new_dict = arg_helper(input_dict,current_dict, replace = False)
            self.metadata.Acquisition_instrument = new_dict
            self.metadata.Acquisition_instrument.TEM.Detector.EDS.type = detector_type
        except AttributeError :
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
            self.metadata.Acquisition_instrument.TEM.Detector.EDS.type = detector_type

    ############################
    # Helper functions for NMF #
    ############################

    def carto_fixed_W(self, brstlg_comps = 1) : 
        r"""
        Helper function to create a fixed_W matrix for chemical mapping. It will output a matrix 
        It can be used to make a decomposition with as many components as they are  chemical elements and then allow each component to have only one of each element.
        The spectral components are then the characteristic peaks of each element and the spatial components are the associated chemical maps.
        The bremsstrahlung is calculated separately and added to other components.

        Parameters
        ----------
        brstlg_comps : int, optional
            Number of bremsstrahlung components to add to the decomposition.

        Returns
        -------
        W : numpy.ndarray
        """
        if self.G_ is None :
            raise ValueError("The G matrix has not been built yet. Please use the build_G method.")
        elements = self.metadata.EDS_model.elements
        if self.problem_type == "no_brstlg" : 
            W = np.diag(-1* np.ones((len(elements), )))
        elif self.problem_type == "bremsstrahlung" : 
            W1 = np.diag(-1* np.ones((len(elements), )))
            W2 = np.zeros((2, len(elements)))
            W_elts = np.vstack((W1,W2))
            W3 = np.zeros((len(elements),brstlg_comps))
            W4 = -1*np.ones((2,brstlg_comps))
            W_brstlg = np.vstack((W3,W4))
            W = np.hstack((W_elts,W_brstlg))

        return W

    def set_fixed_W (self,phases_dict) : 
        r"""
        Helper function to create a fixed_W matrix. The output matrix will have -1 entries except for the elements (and bremsstrahlung parameters) that are present in the phases_dict dictionary.
        In the output (fixed_W) matrix, the -1 entries will be ignored during the decomposition using :class:`espm.estimator.NMFEstimator` are normally learned while the non-negative entries will be fixed to the values given in the phases_dict dictionary.
        Usually, the easiest is to fix some elements to 0.0 in some phases if you want to improve unmixing results. For example, if you have a phase with only Si and O, you can fix the Fe element to 0.0 in this phase.

        Parameters
        ----------
        phases_dict : dict
            Determines which elements of fixed_W are going to be non-negative. The dictionnary has typically the following structure : phases_dict = {"phase1_name" : {"Fe" : 0.0, "O" : 1.25e23}, "phase2_name" : {"Si" : 0.0, "b0" : 0.05}}.
        Returns
        -------
        W : numpy.ndarray
        """
        if self.G_ is None :
            raise ValueError("The G matrix has not been built yet. Please use the build_G method.")
        elements = self.metadata.EDS_model.elements

        # convert elements to symbols but also omitting splitted lines
        @number_to_symbol_list
        def convert_to_symbols(elements = []) : 
            return elements
        
        indices = []
        conv_elts = []
        for i, elt in enumerate(elements) :
            # We always omit low energy lines when they are split (see generate_gmatre of espm.models.EDXS to build G with split lines)
            if re.match(r'.*_lo',elt) :
                pass
            elif re.match(r'.*_hi',elt) :
                indices.append(i)
                conv_elts.append(convert_to_symbols(elements=[elt[:-3]])[0])
            else :
                indices.append(i)
                conv_elts.append(convert_to_symbols(elements=[elt])[0])

        if self.problem_type == "no_brstlg" : 
            W = -1* np.ones((len(elements), len(phases_dict.keys())))
        elif self.problem_type == "bremsstrahlung" : 
            W = -1* np.ones((len(elements)+2, len(phases_dict.keys())))
        else : 
            raise ValueError("problem type should be either no_brstlg or bremsstrahlung")
        for p, phase in enumerate(phases_dict) : 
            for key in phases_dict[phase] : 
                if key == "b0" : 
                    if self.problem_type == "bremsstrahlung" : 
                        W[-2,p] = phases_dict[phase][key]
                    else : 
                        warnings.warn("The chosen EDXS modelling does not incorporate the bremsstrahlung. Input bremsstrahlung parameters will be ignored.")
                if key == "b1" :
                    if self.problem_type == "bremsstrahlung" :
                        W[-1,p] = phases_dict[phase][key]
                    else :
                        warnings.warn("The chosen EDXS modelling does not incorporate the bremsstrahlung. Input bremsstrahlung parameters will be ignored.")
                if key in conv_elts : 
                    W[indices[conv_elts.index(key)],p] = phases_dict[phase][key]
        return W
    
    def concentration_report(self, selected_elts = [], W_input = None, fit_error = True) : 
        if W_input is None :
            if not(isinstance(self.learning_results.decomposition_algorithm,NMFEstimator)) :
                raise ValueError("No espm learning results available, please run a decomposition with an espm algorithm first")
            
            W = self.learning_results.decomposition_algorithm.W_
            G = self.learning_results.decomposition_algorithm.G_
            H = self.learning_results.decomposition_algorithm.H_
            N = get_explained_intensity_W(G,W,H)
            sqN = np.sqrt(N)
            percentages = sqN / N *100

        else :
            W = W_input
            fit_error = False

        elts = self.metadata.EDS_model.elements
        norm = self.metadata.EDS_model.norm

        @number_to_symbol_list
        def convert_elts(elements = []) :
            return elements
        
        # We systematically omit low energy lines when they are split(see generate_gmatre of espm.models.EDXS to build G with split lines)
        elts_only = []
        indices = []
        if selected_elts :
            for i, elt in enumerate(elts) :
                m_hi = re.match(r'([0-9]*)(_hi)',elt)
                m_lo = re.match(r'([0-9]*)(_lo)',elt)
                if m_hi :
                    if convert_elts(elements=[m_hi.group(1)])[0] in selected_elts :
                        elts_only.append(m_hi.group(1))
                        indices.append(i)
                else : 
                    if m_lo : 
                        pass
                    else :
                        if convert_elts(elements=[elt])[0] in selected_elts :
                            elts_only.append(elt)
                            indices.append(i)
        else :
            for i, elt in enumerate(elts) :
                m_hi = re.match(r'([0-9]*)(_hi)',elt)
                m_lo = re.match(r'([0-9]*)(_lo)',elt)
                if m_hi :
                    elts_only.append(m_hi.group(1))
                    indices.append(i)
                else : 
                    if m_lo : 
                        pass
                    else :
                        elts_only.append(elt)
                        indices.append(i)

        conv_elts = convert_elts(elements = elts_only)
        norm = self.metadata.EDS_model.norm[:,indices]
        W = W[indices,:]/W[indices,:].sum(axis = 0)*100
        if fit_error :
            errors = percentages[indices, :]
        else : 
            errors = np.zeros_like(W)
        
        return conv_elts, W, errors
    
    def print_concentration_report (self, selected_elts = [], W_input = None, fit_error = True, disclaimer = True) : 
        r"""
        Print a report of the chemical concentrations from a fitted W.

        Parameters
        ----------
        selected_elts : list, optional
            List of the elements to be printed. If empty, all the elements will be printed.

        W_input : numpy.ndarray, optional
            If not None, the concentrations will be computed from this W matrix instead of the one fitted during the decomposition.

        fit_error : bool, optional
            If True, the statistical errors on the concentrations will be printed.

        disclaimer : bool, optional
            If True, a disclaimer will be printed at the end of the report.

        Returns
        -------
        None

        Notes
        -----
        - This function is only available if the learning results contain a decomposition algorithm that has been fitted.
        """
        conv_elts, W, errors = self.concentration_report(selected_elts = selected_elts, W_input = W_input, fit_error = fit_error)
        
        table = PrettyTable()
        field_list = ["Elements"]
        for i in range(W.shape[1]) :
            field_list.append("p" + str(i) + " (at.%)")
            if fit_error : 
                field_list.append("p" + str(i) + " std (%)")
        table.field_names = field_list
        for i,j in enumerate(conv_elts) :
            row = [j]
            for k in range(W.shape[1]) :
                row.append(W[i,k])
                if fit_error : 
                    row.append(errors[i,k])
            table.add_row(row)

        table.float_format="0.3"
        table.align = "r"
        table.align["Elements"] = "l"
        # table.set_style(MSWORD_FRIENDLY)

        print(table)
        if disclaimer and fit_error: 
            print("\nDisclaimer : The presented errors correspond to the statistical error on the fitted intensity of the peaks.\nIn other words it corresponds to the precision of the measurment.\nThe accuracy of the measurment strongly depends on other factors such as absorption, cross-sections, etc...\nPlease consider these parameters when interpreting the results.")

    def estimate_best_binning(self, inspect = False) :
        r"""
        Estimate the best binning for the dataset based on the method developed by G. Obozinski, N. Perraudin and M. Martinez Ruts.
        M. Martinez Ruts has designed an estimator that compares the binned and unbinned data and its minimum gives the best binning factor. 

        Parameters
        ----------
        bin_sampling : int, optional
            Number of binning factors to sample for the estimation.
        inspect : bool, optional
            If True, the function will return the values of the estimator for each binning factor and the estimated best binning factor.
            If False, it will return only the estimated best binning factor.

        Returns
        -------
        estimated_binning : tuple
            The estimated binning for the dataset.
        """
        # TODO : Write a document explaining the method
        L = self.axes_manager[2].size
        K = self.axes_manager[0].size * self.axes_manager[1].size
        
        facx = np.arange(1, self.axes_manager[0].size//2+1)
        facy = np.arange(1, self.axes_manager[1].size//2+1)

        binx = [self.axes_manager[0].size/i for i in facx]
        biny = [self.axes_manager[1].size/i for i in facy]
        vars_est = np.array([])
        biases_est = np.array([])
        bprod = []
        for i in zip(binx,biny):
            bprod.append((i[0],i[1]))

        for i in tqdm(bprod):
            # Bin the measurement dataset and upsample to bring it back to its original dimensionality
                B = i[0]*i[1]
                binned = self.rebin(scale = (i[0], i[1], 1))
                upsampled = binned.rebin(new_shape = (self.axes_manager[0].size, self.axes_manager[1].size, self.axes_manager[2].size))
                upsampled_data = upsampled.data
                data = self.data

                # Estimator of variance (Lemma 4.3) - \widehat{Var} (\hat{y}_i) = \alpha ^2 y_{i}+ (1-\alpha)^2 \sum_{k \in \mathcal{K}} (w_k^2 n_{i,k})
                vars_est = np.append(vars_est, np.mean(upsampled_data*1/B))

                # Estimator of squared bias (Lemma 4.4) - \widehat{Bias^2}(\hat{y}_i) = (1-\alpha)^2\left((y_{n_i} - y_{i})^2 - \sum_{k\in \mathcal{K}} w_k^2y_{i,k} - y_{i} \right)
                biases_est = np.append(biases_est, np.mean((data-upsampled_data)**2 - 1/B*upsampled_data - (1-2/B)*data))
                
        mprimes_est = vars_est*K/L + biases_est
        estimated_binning = (bprod[np.argmin(mprimes_est)][0], bprod[np.argmin(mprimes_est)][1],1)
        if inspect :
            return mprimes_est, estimated_binning  
        else :
            return estimated_binning


#######################
# Auxiliary functions #
#######################

def get_metadata(spim) : 
    r"""
    Get the metadata of the :class:`EDS_espm` object and format it as a model parameters dictionary.
    """
    mod_pars = {}
    try :
        mod_pars["E0"] = spim.metadata.Acquisition_instrument.TEM.beam_energy
        mod_pars["e_offset"] = spim.axes_manager[-1].offset
        assert mod_pars["e_offset"] > 0.01, "The energy scale can't include 0, it will produce errors elsewhere. Please crop your data."
        mod_pars["e_scale"] = spim.axes_manager[-1].scale
        mod_pars["e_size"] = spim.axes_manager[-1].size
        mod_pars["db_name"] = spim.metadata.xray_db
        mod_pars["width_slope"] = spim.metadata.Acquisition_instrument.TEM.Detector.EDS.width_slope
        mod_pars["width_intercept"] = spim.metadata.Acquisition_instrument.TEM.Detector.EDS.width_intercept
    
        pars_dict = {}
        pars_dict["Abs"] = {
            "thickness" : spim.metadata.Sample.thickness,
            "toa" : spim.metadata.Acquisition_instrument.TEM.Detector.EDS.take_off_angle,
            "density" : spim.metadata.Sample.density
        }
        try : 
            pars_dict["Det"] = spim.metadata.Acquisition_instrument.TEM.Detector.EDS.type.as_dictionary()
        except AttributeError : 
            pars_dict["Det"] = spim.metadata.Acquisition_instrument.TEM.Detector.EDS.type

        mod_pars["params_dict"] = pars_dict

    except AttributeError : 
        print("You need to define the relevant parameters for the analysis. Use the set_analysis_parameters function.")

    return mod_pars



r"""
The module :mod:`espm.eds_spim` implements the :class:`EDS_espm` class, which is a subclass of the :class:`hyperspy.signals.Signal1D` class.
The main purpose of this class is to provide an easy and clean interface between the hyperspy framework and the espm package: 
- The metadata are organised to correspond as much as possible to the typical metadata that can be found in hyperspy EDS_TEM object.
- The machine learning algorithms of espm can be easily applied to the :class:`EDS_espm` object using the standard hyperspy decomposition method. See the notebooks for examples.
- The :class:`EDS_espm` provides a convinient way to:
    - get the results of :class:`espm.estimators.NMFEstimator`
    - access ground truth in case of simulated data
    - set fixed W and H for the :class:`espm.estimators.NMFEstimator` decomposition
"""

from hyperspy.signals import Signal1D
from hyperspy.roi import RectangularROI
from espm.models import EDXS
from espm.models.edxs import G_EDXS
from hyperspy.misc.eds.utils import take_off_angle
from espm.utils import number_to_symbol_list
import numpy as np
from espm.estimators import NMFEstimator
import re


class EDS_espm(Signal1D) : 

    def __init__ (self,*args,**kwargs) : 
        super().__init__(*args,**kwargs)
        self.shape_2d_ = None
        self._phases = None
        self._maps = None
        self._X = None
        self._Xdot = None
        self._maps_2d = None
        self.G = None
        self.model_ = None

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
    def model (self) :
        r"""
        The :class:`espm.models.EDXS` model corresponding to the metadata of the :class:`EDS_espm` object.
        """ 
        if self.model_ is None : 
            mod_pars = get_metadata(self)
            self.model_ = EDXS(**mod_pars)
        return self.model_

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

    def build_G(self, problem_type = "bremsstrahlung", reference_elt = {},stoichiometries = []) :
        r"""
        Build the G matrix of the :class:`espm.models.EDXS` model corresponding to the metadata of the :class:`EDS_espm` object and stores it as an attribute.

        Parameters
        ----------
        problem_type : str, optional
            Determines the type of the G matrix to build. It can be "bremsstrahlung", "no_brstlg" or "identity". The parameters correspond to:
                - "bremsstrahlung" : the G matrix is a callable with both characteristic X-rays and a bremsstrahlung model.
                - "no_brstlg" : the G matrix is a matrix with only characteristic X-rays.
                - "identity" : the G matrix is None which is equivalent to an identity matrix for espm functions.
        reference_elt : dict, optional
            Dictionary containing atomic numbers and a corresponding cut-off energies. It is used to separate the characteristic X-rays of the given elements into two energies ranges and assign them each a column in the G matrix instead of having one column per element.
            For example reference_elt = {"26",3.0} will separate the characteristic X-rays of the element Fe into two energies ranges and assign them each a column in the G matrix. This is useful to circumvent issues with the absorption.
        stoichiometries : list, optional
            List of the stoichiometries of the phases in the sample. In the case the stoichiometry of one of the phase is known, it can be used to improve the accuracy of the decomposition by fixing the ratio between certain elements.
            Each composition of the list should be given a string such as "Fe2O3" or "FeO" for example. A corresponding model element will be added in the metadata. 
            For a clever use of this feature it is best to use it in combination with a fixed W matrix, see the :func:`EDS_espm.set_fixed_W` method.

        Returns
        -------
        G : None or numpy.ndarray or callable
            The G matrix of the :class:`espm.models.EDXS` model corresponding to the metadata of the :class:`EDS_espm` object.
        """
        self.problem_type = problem_type
        self.reference_elt = reference_elt
        self.stoichiometries = stoichiometries
        g_pars = {"g_type" : problem_type, "elements" : self.metadata.Sample.elements, "reference_elt" : reference_elt, "stoichiometries" : stoichiometries}
        
        if problem_type == "bremsstrahlung" : 
            self.G = self.update_G
            # to init the model.model_elts and model.norm
            self.G()
        else : 
            self.G = build_G(self.model,g_pars)

        # Storing the model parameters in the metadata so that the decomposition does not erase them
        # Indeed the decomposition re-creates a new object of the same class when it is called
        self.metadata.EDS_model= {}
        self.metadata.EDS_model.problem_type = problem_type
        self.metadata.EDS_model.reference_elt = reference_elt
        self.metadata.EDS_model.elements = self.model.model_elts
        self.metadata.EDS_model.norm = self.model.norm

    def update_G(self, part_W=None, G=None):
        r"""
        Update the absortion part of the bremsstrahlung of the G matrix.
        """
        try : 
            nelts = (self.metadata.EDS_model.elements).copy()

            for i in self.stoichiometries :
                nelts.remove(i)

            for i, elt in enumerate(nelts) : 
                r = re.match(r'.*_lo',elt)
                t = re.match(r'.*_hi',elt)
                if r : 
                    nelts[i] = nelts[i][:-3]
                if t : 
                    nelts[i] = nelts[i][:-3]
            nelts = list(dict.fromkeys(nelts))

            g_params = {"g_type" : self.problem_type, "elements" : nelts, "reference_elt" : self.reference_elt, "stoichiometries" : self.stoichiometries}
        except AttributeError : 
            g_params = {"g_type" : self.problem_type, "elements" : self.metadata.Sample.elements, "reference_elt" : self.reference_elt, "stoichiometries" : self.stoichiometries}
        G = G_EDXS(self.model, g_params, part_W=part_W, G=G)
        return G

    def set_analysis_parameters (self,  thickness = 200e-7, density = 3.5, detector_type = "SDD_efficiency.txt", width_slope = 0.01, width_intercept = 0.065, xray_db = "default_xrays.json") :
        r"""
        Helper function to set the metadata of the :class:`EDS_espm` object. Be careful, it will overwrite the metadata of the object.

        Parameters
        ----------
        thickness : float, optional
            The thickness of the sample in centimeters.
        density : float, optional
            The density of the sample in g/cm^3.
        detector_type : str, optional
            The type of the detector. It is either  the name of a text file containing the efficiency of 
        width_slope : float, optional
            The slope of the linear fit of the detector width as a function of the energy.
        width_intercept : float, optional
            The intercept of the linear fit of the detector width as a function of the energy.
        xray_db : str, optional
            The name of the X-ray emission cross-section database to be used. The default tables are avalaible in the espm/tables folder. Additional tables can be generated by emtables.
        """

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



        self.metadata.Acquisition_instrument.TEM.Detector.EDS.take_off_angle = take_off_angle(self.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha,
            self.metadata.Acquisition_instrument.TEM.Detector.EDS.azimuth_angle,
            self.metadata.Acquisition_instrument.TEM.Detector.EDS.elevation_angle)


    def set_additional_parameters(self,thickness = 200e-7, density = 3.5,  detector_type = "SDD_efficiency.txt", width_slope = 0.01, width_intercept = 0.065, xray_db = "default_xrays.json") : 
        r"""
        Helper function to set the metadata that are specific to the :mod:`espm` package so that it does not overwrite experimental metadata.
        See the documentation of the :func:`set_analysis_parameters` function for the meaning of the parameters.
        """
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

        tilt_stage = self.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha
        azimuth_angle = self.metadata.Acquisition_instrument.TEM.Detector.EDS.azimuth_angle
        elevation_angle = self.metadata.Acquisition_instrument.TEM.Detector.EDS.elevation_angle
        self.metadata.Acquisition_instrument.TEM.Detector.EDS.take_off_angle = take_off_angle(tilt_stage,azimuth_angle,elevation_angle)

    @number_to_symbol_list
    def add_elements(self, *, elements = []) :
        r"""
        Add elements to the existing list of elements in the metadata.

        Parameters
        ----------
        elements : list, optional
            List of the elements to be added to the existing list of elements in the metadata. They have to be chemical symbols (e.g. ['Si','Fe', 'O']).
        """
        try : 
            self.metadata.Sample.elements = elements
        except AttributeError :
            self.metadata.Sample = {}
            self.metadata.Sample.elements = elements

    def set_microscope_parameters(self, beam_energy = 200, azimuth_angle = 0.0, elevation_angle = 22.0,tilt_stage = 0.0) : 
        r"""
        Helper function to set the microscope parameters of the :class:`EDS_espm` object. Be careful, it will overwrite the microscope parameters of the object.
        See the documentation of the :func:`set_analysis_parameters` function for the meaning of the parameters.
        """
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
        elements = self.metadata.EDS_model.elements

        # convert elements to symbols but also omitting splitted lines or custom stoichiometries
        @number_to_symbol_list
        def convert_to_symbols(elements = []) : 
            return elements
        
        for i, elt in enumerate(elements) :
            try : 
                elements[i] = convert_to_symbols(elements=[elt])[0]
            except ValueError : 
                elements[i] = elt

        if self.problem_type == "no_brstlg" : 
            W = -1* np.ones((len(elements), len(phases_dict.keys())))
        elif self.problem_type == "bremsstrahlung" : 
            W = -1* np.ones((len(elements)+2, len(phases_dict.keys())))
        else : 
            raise ValueError("problem type should be either no_brstlg or bremsstrahlung")
        for p, phase in enumerate(phases_dict) : 
            for e, elt in enumerate(elements) :
                for key in phases_dict[phase] : 
                    if key == elt : 
                        W[e,p] = phases_dict[phase][key]
            for key in phases_dict[phase] : 
                if key == "b0" : 
                    W[-2,p] = phases_dict[phase][key]
                if key == "b1" : 
                    W[-1,p] = phases_dict[phase][key]
        return W
    
    def define_ROI(self):
        r"""
        A function to define a rectangular ROI on an HyperSpy EDXS signal.
        
        Parameters
        ----------
        data : hs.signals.EDSTEMSpectrum
            Input EDXS datacube.
            
        Returns
        -------
        roi : hs.roi.RectangularROI
            A rectangular ROI defined by the user.
        """
        
        centre_x = self.data.shape[1] // 2
        centre_y = self.data.shape[0] // 2
        dx = self.data.shape[1] // 10
        dy = self.data.shape[0] // 10
        
        roi = RectangularROI(left = centre_x - dx, top = centre_y - dy, right = centre_x + dx, bottom = centre_y + dy)
        self.plot()
        imr = roi.interactive(self, color = 'r')
        
        return roi
    
    class MatplotlibBackendError(ValueError):
        pass
    
    def generate_manual_mask(self, n_classes = 3, mask_alpha = 0.75):
        r"""
        Function used to manually draw a mask with multiple classes on the summed spatial map of the data.
        [Warning] This function requires the 'mpl_interactions', 'ipywidgets' and, optionally, 'ipympl' modules to be installed.
        [Warning] Using this function requires restarting the IPython kernel since it changes the Matplotlib backend.
        
        Parameters
        ----------
        n_classes : int
            The number of classes to be used for the mask.
        mask_alpha : float
            The transparency of the mask.
        
        Returns
        -------
        None
        
        Note: The generated mask is saved as a NumPy array in the current working directory.
        """
        import matplotlib.pyplot as plt
        
        try:
            from mpl_interactions import image_segmenter
        except:
            raise ImportError("Module 'mpl_interactions' is not installed.")
        
        try:
            import ipywidgets as widgets
        except:
            raise ImportError("Module 'ipywidgets' is not installed.")
        
        if plt.get_backend() != 'module://ipympl.backend_nbagg':
            raise MatplotlibBackendError("Please switch to the 'ipympl' or 'widget' Matplotlib backend.")
        
        image = self.sum(axis = 2).data

        class_selector = widgets.Dropdown(options=list(range(1, n_classes + 1)), description = "class")


        def update(change):
            multi_class_segmenter.current_class = class_selector.value
            multi_class_segmenter.erasing = erasing_button.value
            
        def on_done_clicked(b):
            np.save(f"mask_{n_classes}_classes.npy", multi_class_segmenter.mask)
            print("Mask saved!")


        erasing_button = widgets.Checkbox(value = False, description = "Erasing")
        erasing_button.observe(update, names = "value")

        class_selector.observe(update, names = "value")
        multi_class_segmenter = image_segmenter(image, nclasses = n_classes, mask_alpha = mask_alpha)
        
        done_button = widgets.Button(description = "Done", disabled = False, button_style = "success", tooltip = "Save mask", icon = "check")
        done_button.on_click(on_done_clicked)
        
        display(widgets.HBox([erasing_button, class_selector, done_button]))
        display(multi_class_segmenter)
    
    def generate_part_fixed_H_matrix(self, type = None, mask = None, ROIs = None, value = 1):
        r"""
        A function to generate a component of the fixed H matrix for one phase.
        
        Parameters
        ----------
        type : str
            Type of the fixed H matrix component. Can be 'mask', 'ROI' or 'not_fixed'.
        mask : np.ndarray
            A binary mask given by the user.
        ROIs : list
            A list of rectangular ROIs given by the user.
        value : float
            Value of the non-negative entries in the partial H matrix. Must be between 0 and 1.
            
        Returns
        -------
        part_f_H : np.ndarray
            A fixed H matrix for one phase.
        """
        part_f_H = (-1) * np.ones(shape = (self.data.shape[0], self.data.shape[1]), dtype = float)
        
        if value > 1 or value < 0:
            raise ValueError("Value must be between 0 and 1.")
        
        if type is None:
            raise ValueError("Type is not defined.")
        
        if type == 'not_fixed':
            return part_f_H
        
        if type == 'mask':
            if mask is None:
                raise ValueError("Mask is not defined.")
            else:
                if mask.shape != (self.data.shape[0], self.data.shape[1]):
                    raise ValueError("Mask shape does not match data shape.")
                part_f_H[mask != 0] = value
        
        if type == 'ROI':
            if ROIs is None:
                raise ValueError("ROIs are not defined.")
            else:
                for i in range(len(ROIs)):
                    region_parameters = ROIs[i].parameters
                    scale_i = self.axes_manager[0].scale
                    scale_j = self.axes_manager[1].scale
                    j_min = int(region_parameters['left'] // scale_j)
                    i_min = int(region_parameters['top'] // scale_i)
                    j_max = int(region_parameters['right'] // scale_j)
                    i_max = int(region_parameters['bottom'] // scale_i)
                    part_f_H[i_min:i_max, j_min:j_max] = value
        
        return part_f_H
    
    def set_fixed_H(self, areas_dict):
        r"""
        Helper function to generate a fixed H matrix for the SmoothNMF decomposition algorithm. The output matrix will have -1 entries except for the
        areas that are specified in the input dictionary. The -1 entries will be ignored during the decomposition and learned normally, while the
        non-negative entries will be kept fixed.
        
        Parameters
        ----------
        areas_dict : dict
            Determines which areas are going to be non-negative. The dictionary has the following structure:
            areas_dict = {"p0" : part_f_H_0, "p1" : part_f_H_1 ...}
            where part_f_H_0, part_f_H_1, ... are NumPy arrays with the same dimensions as the input data's spatial dimensions. They are generated using the generate_part_fixed_H_matrix() function.
            
        Returns
        -------
        H : numpy.ndarray
            A fixed H matrix for the SmoothNMF decomposition algorithm.
        """
        
        H = (-1) * np.ones(shape = (len(areas_dict), self.data.shape[0], self.data.shape[1]), dtype = float)
        
        for i, p in enumerate(areas_dict):
            H[i, :, :] = areas_dict[p]
            
        return H.reshape((len(areas_dict), self.data.shape[0] * self.data.shape[1]))
    
    def print_concentration_report (self,abs = False, selected_elts = [], W_input = None) : 
        r"""
        Print a report of the chemical concentrations from a fitted W.

        Parameters
        ----------
        abs : bool
            If True, print the absolute concentrations, if False, print the relative concentrations.

        selected_elts : list, optional
            List of the elements to be printed. If empty, all the elements will be printed.

        W_input : numpy.ndarray, optional
            If not None, the concentrations will be computed from this W matrix instead of the one fitted during the decomposition.

        Returns
        -------
        None

        Notes
        -----
        - This function is only available if the learning results contain a decomposition algorithm that has been fitted.
        - The "absolute" concentrations correspond to some physical number. To retrieve the number of atoms per unit volume, you need to multiply by the correct pre-factors such as beam current, detector solid angle, etc...
        """
        if W_input is None :
            if not(isinstance(self.learning_results.decomposition_algorithm,NMFEstimator)) :
                raise ValueError("No espm learning results available, please run a decomposition with an espm algorithm first")
            
            W = self.learning_results.decomposition_algorithm.W_

        else :
            W = W_input

        elts = self.metadata.EDS_model.elements
        norm = self.metadata.EDS_model.norm

        @number_to_symbol_list
        def convert_elts(elements = []) :
            return elements
        
        single_elts = [e for e in elts if len(e) < 3]
        elements = convert_elts(elements = single_elts)
        elements += [e for e in elts if len(e) > 2]

        if selected_elts : 
            indices = [index for index,elt in enumerate(elements) if elt in selected_elts]
            W = W[indices,:]
            elements = [elt for elt in elements if elt in selected_elts]
            norm = self.metadata.EDS_model.norm[:,indices]

        mask_zeros = W[:len(elements),:].sum(axis=0) !=0
        norm_W = W[:len(elements),mask_zeros] / W[:len(elements),mask_zeros].sum(axis = 0)
        abs_W =   W / norm[0][:,np.newaxis] 
        longest_name = len(max(elements, key=len))
        
        if abs : 
            print("Abs. quantif. report")
            title_string = " "*longest_name

            for i in range(norm_W.shape[1]) : 
                title_string += "{:>10}".format("p" + str(i))
            print(title_string)
            
            for i,j in enumerate(elements) : 
                main_string = ""
                main_string += "{:{}}".format(j,longest_name) + " : "
                for k in range(norm_W.shape[1]) :
                    main_string += "{:.3e} ".format(abs_W[i,k])
                print(main_string)

        else : 
            print("Concentrations report")
            title_string = " "* longest_name

            for i in range(norm_W.shape[1]) : 
                title_string += "{:>7}".format("p" + str(i))
            print(title_string)
            
            for i,j in enumerate(elements) : 
                main_string = ""
                main_string += "{:{}}".format(j,longest_name) + " : "
                for k in range(norm_W.shape[1]) :
                    main_string += "{:05.4f} ".format(norm_W[i,k])
                print(main_string)

    def quantify(self):
        
        r"""
        Descript 

        Parameters
        ----------
        mask : str
            The path to the file containing the mask.
        
        Returns
        -------
        masks : dict
            A dictionary containing the binary masks for each class.
        """

        self.build_G()
        els = self.metadata.EDS_model.elements
        est = self.learning_results.decomposition_algorithm
        
        # can't believe this isn't more readily available in espm.utils:
        
        els_names = [num_to_symbol(el) for el in els]
        
        WH = (est.W_@est.H_).reshape([len(els)+2]+list(self.data.shape[:-1]))[:-2]
        WH/=WH.sum(0)[np.newaxis,...]/100


        if self.axes_manager.navigation_dimension == 2:        
            Signal = hs.signals.Signal2D
        elif self.axes_manager.navigation_dimension ==1:
            Signal = hs.signals.Signal1D

        qs = [Signal(WH[i],
                    metadata = {"General":{"name":el,
                    "title":el+" Quantification"}},colorbar_label="A") for i,el in enumerate(els_names)]
        
        for q in qs:
            for i in range(self.axes_manager.navigation_dimension):
                q.axes_manager[i].update_from(self.axes_manager[i],["units","scale","name","offset"])

            wh = Signal(WH)

        wh.metadata.EDS_model = self.metadata.EDS_model
        for i in range(self.axes_manager.navigation_dimension):
            wh.axes_manager[1+i].update_from(self.axes_manager[i],["units","scale","name","offset"])
            
        self.quantification_list = qs
        self.quantification_signal =wh
        return

    def plot_comp_model(self,comp_index):
        r"""
        Descript 

        Parameters
        ----------
        mask : str
            The path to the file containing the mask.
        
        Returns
        -------
        masks : dict
            A dictionary containing the binary masks for each class.
        """

        idx = comp_index
        els = self.metadata.EDS_model.elements
        els = [num_to_symbol(el) for el in els]+["Background 1","Background 2"]
        self.GW = self.learning_results.decomposition_algorithm.G_@self.learning_results.decomposition_algorithm.W_
        gs,cs = self.learning_results.decomposition_algorithm.W_.shape
        G_idx = self.learning_results.decomposition_algorithm.G_*self.learning_results.decomposition_algorithm.W_[:,idx]
        
        x = self.axes_manager[-1].axis
        plt.figure()
        plt.plot(x,self.GW[:,idx],"k--",label="Component")
        for i in range(gs):
            color = list(mpl.colors.TABLEAU_COLORS.values())[i%10]
            plt.plot(x,G_idx[:,i],color = color,label = els[i])
            plt.fill_between(x,G_idx[:,i],alpha = 0.4,color = color)
        plt.legend()
        ax = plt.gca()
        plt.title("Model of component {}".format(str(idx)))
        
        return plt.gcf()


    def plot_data_model(self):
        r"""
        Descript 

        Parameters
        ----------
        mask : str
            The path to the file containing the mask.
        
        Returns
        -------
        masks : dict
            A dictionary containing the binary masks for each class.
        """


        W = self.learning_results.decomposition_algorithm.W_
        G = self.learning_results.decomposition_algorithm.G_
        H = self.learning_results.decomposition_algorithm.H_

        WH = W@H

        contributions = [hs.signals.Signal1D((G[:,[i]]@(WH)[[i],:]).T.reshape(self.data.shape))  for i in range(G.shape[1])]
        contributions.append(hs.signals.Signal1D((G@WH).T.reshape(self.data.shape)))
        els = self.metadata.EDS_model.elements 
        titles =[num_to_symbol(el) for el in els]+["Background 1","Bacground 2","Full Model"]
        for i,c in enumerate(contributions):
                for a,b in zip(c.axes_manager._axes,self.axes_manager._axes):
                    a.update_from(b)
                c.metadata.General.title = titles[i]


        if self.axes_manager.navigation_dimension == 1:
            nav = self.sum(-1).as_signal1D(0)
            position = hs.roi.Point1DROI(0)
            nav_kwargs={"color":"blue"}

        elif self.axes_manager.navigation_dimension == 2:
            nav = self.sum(-1).as_signal2D((0,1))
            position = hs.roi.Point2DROI(0,0)
            nav_kwargs={}

        nav.plot(**nav_kwargs)
        position_interactive = position.interactive(self,nav,color="red")
        positions_contribs = [position.interactive(g,None) for g in contributions]
        hs.plot.plot_spectra([position_interactive]+positions_contribs,legend="auto",
                             linestyle=["-"]+["--" for i in positions_contribs],
                             color = ["k"]+list(mpl.colors.TABLEAU_COLORS.values())*10)

        return
        


######################
# Axiliary functions #
######################

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

def build_G(model, g_params) :
    model.generate_g_matr(**g_params)
    return model.G




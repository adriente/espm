r"""
The module :mod:`espm.eds_spim` implements the :class:`EDSespm` class, which is a subclass of the :class:`hyperspy.signals.Signal1D` class.
The main purpose of this class is to provide an easy and clean interface between the hyperspy framework and the espm package: 
- The metadata are organised to correspond as much as possible to the typical metadata that can be found in hyperspy EDS_TEM object.
- The machine learning algorithms of espm can be easily applied to the :class:`EDSespm` object using the standard hyperspy decomposition method. See the notebooks for examples.
- The :class:`EDSespm` provides a convinient way to:
    - get the results of :class:`espm.estimators.NMFEstimator`
    - access ground truth in case of simulated data
    - estimate best binning thanks to the method developed by G. Obozinski, N. Perraudin and M. Martinez Ruts.
    - set fixed W for the :class:`espm.estimators.NMFEstimator` decomposition
"""

from exspy.signals import EDSTEMSpectrum
from espm.models import EDXS
from exspy.utils.eds import take_off_angle
from espm.utils import number_to_symbol_list, get_explained_intensity_W, symbol_to_number_list
import numpy as np
from espm.estimators import NMFEstimator
import re
import warnings
from prettytable import PrettyTable
from tqdm import tqdm
from espm.estimators import SmoothNMF
from espm.conf import NUMBER_PERIODIC_TABLE
import json
from hyperspy.signal_tools import Signal1DRangeSelector
from hyperspy.ui_registry import get_gui
import intervaltree

NPT = json.load(open(NUMBER_PERIODIC_TABLE))

class EDSespm(EDSTEMSpectrum) : 

    _signal_type = "EDS_espm"

    def __init__ (self,*args,**kwargs) : 
        super().__init__(*args,**kwargs)
        self.shape_2d_ = None
        self._X = None
        self.G_ = None
        self.model_ = None
        self.custom_init_ = None
        self.ranges = None
        self._set_default_analysis_params()

    ##############
    # Properties #
    ##############

    def _set_default_analysis_params(self) :
        # TODO : make them fetch preferences from the user 
        md = self.metadata
        md.Signal.signal_type = "EDS_espm"
        
        if "Acquisition_instrument.TEM.Detector.EDS.width_slope" not in md :
            md.set_item("Acquisition_instrument.TEM.Detector.EDS.width_slope", 0.01)
        if "Acquisition_instrument.TEM.Detector.EDS.width_intercept" not in md :
            md.set_item("Acquisition_instrument.TEM.Detector.EDS.width_intercept", 0.065)
        if "xrays_db" not in md :
            md.set_item("xray_db", "200keV_xrays.json")
        if "Acquisition_instrument.TEM.Detector.EDS.type" not in md :
            md.set_item("Acquisition_instrument.TEM.Detector.EDS.type", "SDD_efficiency.txt")
        if "Acquisition_instrument.TEM.Stage.tilt_beta" not in md :
            md.set_item("Acquisition_instrument.TEM.Stage.tilt_beta", 0.0)

    def _check_metadata_G(self) : 
        md = self.metadata

        if "Sample.elements" not in md :
            raise ValueError("The elements of the sample are missing in the metadata. Please use the set_elements method to set the elements.")
        if "Acquisition_instrument.TEM.beam_energy" not in md :
            raise ValueError("The beam energy is missing in the metadata. Please use the set_microscope_parameters method to set the beam energy.")
        if "Sample.density" not in md :
            raise ValueError("The density of the sample is missing in the metadata. Please use the set_analysis_parameters method to set the density.")
        if "Sample.thickness" not in md :
            raise ValueError("The thickness of the sample is missing in the metadata. Please use the set_analysis_parameters method to set the thickness.")
        if "Acquisition_instrument.TEM.Detector.EDS.type" not in md :
            raise ValueError("The detector type is missing in the metadata. Please use the set_analysis_parameters method to set the detector type.")
        if "Acquisition_instrument.TEM.Detector.EDS.take_off_angle" not in md :
            raise ValueError("The take-off angle is missing in the metadata. Please use the set_microscope_parameters method to set the take-off angle.")
        if "Acquisition_instrument.TEM.Detector.EDS.width_slope" not in md :
            raise ValueError("The width slope is missing in the metadata. Please use the set_analysis_parameters method to set the width slope.")
        if "Acquisition_instrument.TEM.Detector.EDS.width_intercept" not in md :
            raise ValueError("The width intercept is missing in the metadata. Please use the set_analysis_parameters method to set the width intercept.")
        if "xray_db" not in md :
            raise ValueError("The xray database is missing in the metadata. Please use the set_analysis_parameters method to set the xray database.")
        
    def _check_metadata_quantification(self) : 
        md = self.metadata

        if "Acquisition_instrument.TEM.Detector.EDS.geometric_efficiency" not in md :
            raise ValueError("The geometric efficiency of the detector is missing in the metadata. Please use the set_analysis_parameters method to set the geometric efficiency.")
        if "Acquisition_instrument.TEM.beam_current" not in md :
            raise ValueError("The beam current is missing in the metadata. Please use the set_microscope_parameters method to set the beam current.")
        if "Acquisition_instrument.TEM.Detector.EDS.real_time" not in md :
            raise ValueError("The acquisition time is missing in the metadata. Please use the set_microscope_parameters method to set the acquisition time.")
        
    @property
    def custom_init (self) :
        r"""
        Boolean setting whether using the custom_init (see espm.models.EDXS) or not.
        If True, the custom_init will be used to initialise the decomposition.
        If False, the default initialisation will be used.
        If None, the  will be set to False.
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
    def model(self) :
        r"""
        The :class:`espm.models.EDXS` model corresponding to the metadata of the :class:`EDSespm` object.
        """ 
        if self.model_ is None : 
            mod_pars = get_metadata(self)
            self.model_ = EDXS(**mod_pars, custom_init=self.custom_init_)
        return self.model_
    
    @property
    def G(self) :
        r"""
        The G matrix of the :class:`espm.models.EDXS` model corresponding to the metadata of the :class:`EDSespm` object.
        """
        if self.G_ is None : 
            try : 
                if self.problem_type == "identity" :
                    return None
            except AttributeError :
                warnings.warn("You did not used the build_G method to build the G matrix. In ESpM-NMF, an idenity matrix will be used for decomposition")
                return None
        return self.G_

    def build_G(self, problem_type = "bremsstrahlung",ignored_elements = ['Cu'],*, elements_dict = {}) :
        r"""
        Build the G matrix of the :class:`espm.models.EDXS` model corresponding to the metadata of the :class:`EDSespm` object and stores it as an attribute.

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
        None 
        """
        self._check_metadata_G()
        self.problem_type = problem_type
        self.separated_lines = elements_dict
        g_pars = {"g_type" : problem_type, 'ignored_elements' : ignored_elements, "elements" : self.metadata.Sample.elements, "elements_dict" : elements_dict}

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
    # Bremsstrahlung functions #
    ############################

    def estimate_mass_thickness(self, ignored_elements = ['Cu'], tol = 1e-8,*, elements_dict = {}) :
        r"""
        Based on the complete metadata of the :class:`EDSespm` object, this function estimates the mass thickness of the sample. This function derives the mass-thickness from the characteristic X-rays. Then the bremsstrahlung parameters are estimated using that mass-thickness. The process is then repeated ten times to ensure convergence. The results are plotted on the spectrum.

        Check the metadata to read the estimated mass-thickness.
        
        Parameters
        ----------
        elements_dict : dict, optional
            Dictionary containing atomic numbers and a corresponding cut-off energies. It is used to separate the characteristic X-rays of the given elements into two energies ranges and assign them each a column in the G matrix instead of having one column per element. This is useful to circumvent issues with the mass-absorption coefficient.

        Returns
        -------
        None

        Notes
        -----
        The mass-thickness :math:`\rho t` in g.cm^-2 is estimated using the following formula:

        .. math::
            \rho t = \frac{H}{I \times 10^{-9} \times \tau \times N_e \times \sigma \times \Omega / (4\pi)}

        where :math:`H` is the intensity of the characteristic X-rays, :math:`I` is the beam current in nA, :math:`\tau` is the acquisition time in seconds, :math:`N_e` is the number of electrons in a Coulomb, :math:`sigma` is the average X-ray emission cross-section, and :math:`\Omega` is the geometric efficiency of the detector in sr.

        We recommend to use the :meth:`select_background_windows` method to select the background windows before running this method.
        """
        # Let's implement for 1D data first. So we sum over dimensions if needed.
        self._check_metadata_G()
        self._check_metadata_quantification()
        if len(self.axes_manager.navigation_axes) > 0 : 
            raise NotImplementedError('For now this function is not fully implemented for spectrum images. Use this on an extracted 1D spectrum.')
        curr_X = self.data

        # First init of fit
        self.build_G(ignored_elements= ignored_elements, elements_dict=elements_dict)
        estimator = SmoothNMF(n_components = 1, G=self.model)
        estimator.fit(curr_X[:,np.newaxis])
        H_init = estimator.H_
        W_init = estimator.W_
        elts = list(self.model.get_elements(include_ignored = False))
        elts_indices = self.model.NMF_simplex()
        new_elts_dict = {elts[i] : W_init[elts_indices[i]] for i in range(len(elts))}
        
        _ = 0
        curr_mt = self.metadata.Sample.thickness * self.metadata.Sample.density
        while _ < 5 :
            # first init of the model
            brstlg_model, mask = self.model.bremsstrahlung_only_tools(mass_thickness=curr_mt,elements_dict = new_elts_dict, ranges = self.ranges)
            masked_X = curr_X[mask]
            brstlg_estimator = SmoothNMF(n_components = 1, G=brstlg_model, fixed_H = H_init, tol = tol )
            brstlg_estimator.fit(masked_X[:,np.newaxis])
            W_brstlg = np.vstack(( -1 * np.ones((W_init.shape[0] - brstlg_estimator.W_.shape[0], brstlg_estimator.W_.shape[1])),brstlg_estimator.W_))

            self.build_G(ignored_elements= ignored_elements, elements_dict=elements_dict)
            # First estimation of the bremsstrahlung + elts
            estimator = SmoothNMF(n_components = 1, G=self.model, fixed_W = W_brstlg, tol = tol)
            estimator.fit(curr_X[:,np.newaxis])
            
            # Get the elements, their concentrations and the mass_thickness value
            W_init = estimator.W_
            H_init = estimator.H_
            
            elts = list(self.model.get_elements(include_ignored = False))
            elts_indices = self.model.NMF_simplex()
            new_elts_dict = {elts[i] : W_init[elts_indices[i]] for i in range(len(elts))}
            total_weight = self._elements_dict_to_weights(new_elts_dict)
            curr_mt = self._extract_mass_thickness(H_init.sum(), total_weight)

            _ += 1

            print("The current estimated mass-thickness is {} g.cm^-2".format(curr_mt),flush = True)

        self.plot(True)
        self._plot.signal_plot.ax.set_title("Estimated mass-thickness : {} g.cm^-2".format(curr_mt))
        
        axis = self.axes_manager.signal_axes[0].axis
        self._plot.signal_plot.ax.plot(axis,
                                       estimator.G_@estimator.W_@estimator.H_,
                                       'b-',
                                       label = 'Full model')
        self._plot.signal_plot.ax.plot(axis[mask],
                                       brstlg_estimator.G@brstlg_estimator.W_@brstlg_estimator.H_,
                                       'g.',
                                       label = 'Bremmstrahlung')
        self._plot.signal_plot.ax.legend()

        self.metadata.Sample.thickness = 1.0
        self.metadata.Sample.density = curr_mt

    def _elements_dict_to_weights(self,elements_dict) :
        """
        Convert a dictionary of elements and their quantities to total weight.

        Parameters
        ----------
        elements_dict : dict
            Dictionary containing atomic numbers as keys and quantities as values.

        Returns
        -------
        total_weight : float
            Total weight of the elements in grams.
        """
        total_weight = sum(
            quantity * NPT['table'][element]['atomic_mass'] * 1.66053906660e-24
            for element, quantity in elements_dict.items()
        )
        return total_weight

    def _extract_mass_thickness(self,H_value, total_weight) : 
        Na = 6.02214179e23 # TODO : Check the usefulness of Na. 
        # If I am correct the concentrations we guess have no unit.
        # Since they are not in mole, no need to normalize using Na
        Ne = 6.25e18 # Number of electrons in a Coulomb
        # real time shound be the whole acquisition time (without dead time but with all pixels)
        return H_value* total_weight/(self.metadata.Acquisition_instrument.TEM.beam_current * 1e-9 *
                  self.metadata.Acquisition_instrument.TEM.Detector.EDS.real_time *
                  Ne  * self.model.norm[0][0] *
                  (self.metadata.Acquisition_instrument.TEM.Detector.EDS.geometric_efficiency/(4*np.pi))
                  )
    
    def select_background_windows(self, num_windows = 4, ranges = None) :
        r"""
        Select the background windows for the bremsstrahlung estimation. The function will open a window with the spectrum and the user will be able to select the background windows by clicking and dragging the mouse. Click then on 'Apply' to validate the selection. A bremmstrahlung model will be estimated and plotted on the spectrum.

        Parameters
        ----------
        num_windows : int, optional
            Number of background windows to select.
        ranges : list, optional
            List of tuples containing the left and right bounds of the background windows. If provided, the function will not open a window and will directly use the provided ranges, bypassing the gui.

        Returns
        -------
        None
        """
        # The code is quite dirty, but it works.
        # To code a proper gui we need to wait for an update of hyperspy
        if self.model is None : 
            raise ValueError("The G matrix has not been built yet. Please use the build_G method.")
        if ranges is not None :
           self.ranges = ranges
           self.model.ranges = self.ranges
        else :  
            if len(self.axes_manager.navigation_axes) > 0 : 
                raise NotImplementedError('For now this function is not fully implemented for spectrum images. Use this on an extracted 1D spectrum.')
            cm = self._register_ranges
            init_ranges = self._generate_ranges(num_windows)
            self.spans = []
            for i in range(num_windows) : 
                self.spans.append(Signal1DRangeSelector(self))
            
            for j, span in enumerate(self.spans) : 
                span.span_selector.extents = init_ranges[j]
                span.on_close.append((cm, self))
                get_gui(span, toolkey = "hyperspy.interactive_range_selector")

    def _register_ranges(self,signal, left, right) : 
        # The unused args are required for the event to properly complete
        coord_list = [[span.ss_left_value, span.ss_right_value] for span in self.spans]
        coord_list.sort(key = lambda coord : coord[0])
        tree = intervaltree.IntervalTree.from_tuples(coord_list)
        self.ranges = []
        for branch in tree : 
            self.ranges.append([branch[0], branch[1]])

        self.model.ranges = self.ranges

        model = self._compute_bremsstrahlung()
        self._plot_background(model)

    def _compute_bremsstrahlung(self) :
        mt = self.metadata.Sample.density * self.metadata.Sample.thickness
        elts_dict = {elt : 1.0 for elt in self.metadata.Sample.elements} 
        brstlg_model, mask = self.model.bremsstrahlung_only_tools(mass_thickness=mt,elements_dict = elts_dict, ranges= self.ranges)
        curr_X = self.data
        masked_X = curr_X[mask]

        # Estimate the bremsstrahlung on the partial data
        brstlg_estimator = SmoothNMF(n_components = 1, G=brstlg_model)
        brstlg_estimator.fit(masked_X[:,np.newaxis])
        # get the fitting results
        fH = brstlg_estimator.H_
        fW = brstlg_estimator.W_

        # Now adapt to the full range
        # It is not super efficient but I think it is not an issue. The model can't be easily continued, it is not a function.
        axis = self.axes_manager.signal_axes[0]
        full_range = [[axis.low_value, axis.high_value]]
        full_brstlg_model, full_mask = self.model.bremsstrahlung_only_tools(mass_thickness=mt,elements_dict = elts_dict, ranges= full_range)

        return full_brstlg_model@fW@fH
    
    def _plot_background(self, model) :
        # The full range from self.compute_background misses both ends
        # The axis needs to be trimmed accordingly
        axis = self.axes_manager.signal_axes[0].axis[1:-1] 
        self._plot.signal_plot.ax.plot(axis,model)
            
    def _generate_ranges(self, num) : 
        axis = self.axes_manager.signal_axes[0]
        bounds = (axis.low_value, axis.high_value)
        values = np.linspace(bounds[0], bounds[1], num = 2*num + 2)
        ranges_list = [(values[2*i-1],values[2*i]) for i in range(1,num+1)]
        return ranges_list

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
        raw_elts = self.metadata.EDS_model.elements
        elements = self.model.get_elements()
        indices = self.model.NMF_simplex()

        # convert elements to symbols but also omitting splitted lines
        @number_to_symbol_list
        def convert_to_symbols(elements = []) : 
            return elements
        
        conv_elts = convert_to_symbols(elements=elements)

        if self.problem_type == "no_brstlg" : 
            W = -1* np.ones((len(raw_elts), len(phases_dict.keys())))
        elif self.problem_type == "bremsstrahlung" : 
            W = -1* np.ones((len(raw_elts)+2, len(phases_dict.keys())))
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
    
    def decomposition(
        self,
        normalize_poissonian_noise=False,
        navigation_mask=None,
        closing=True,
        *args,
        **kwargs,
    ):
        """Apply a decomposition to a dataset with a choice of algorithms.

        The results are stored in ``self.learning_results``.

        Read more in the :ref:`User Guide <mva.decomposition>`.

        Parameters
        ----------
        normalize_poissonian_noise : bool, default True
            If True, scale the signal to normalize Poissonian noise using
            the approach described in [*]_.
        navigation_mask : None or float or boolean numpy array, default 1.0
            The navigation locations marked as True are not used in the
            decomposition. If float is given the vacuum_mask method is used to
            generate a mask with the float value as threshold.
        closing: bool, default True
            If true, applied a morphologic closing to the mask obtained by
            vacuum_mask.
        algorithm : {"SVD", "MLPCA", "sklearn_pca", "NMF", "sparse_pca", "mini_batch_sparse_pca", "RPCA", "ORPCA", "ORNMF", custom object}, default "SVD"
            The decomposition algorithm to use. If algorithm is an object,
            it must implement a ``fit_transform()`` method or ``fit()`` and
            ``transform()`` methods, in the same manner as a scikit-learn estimator.
        output_dimension : None or int
            Number of components to keep/calculate.
            Default is None, i.e. ``min(data.shape)``.
        centre : {None, "navigation", "signal"}, default None
            * If None, the data is not centered prior to decomposition.
            * If "navigation", the data is centered along the navigation axis.
              Only used by the "SVD" algorithm.
            * If "signal", the data is centered along the signal axis.
              Only used by the "SVD" algorithm.
        auto_transpose : bool, default True
            If True, automatically transposes the data to boost performance.
            Only used by the "SVD" algorithm.
        signal_mask : boolean numpy array
            The signal locations marked as True are not used in the
            decomposition.
        var_array : numpy array
            Array of variance for the maximum likelihood PCA algorithm.
            Only used by the "MLPCA" algorithm.
        var_func : None or function or numpy array, default None
            * If None, ignored
            * If function, applies the function to the data to obtain ``var_array``.
              Only used by the "MLPCA" algorithm.
            * If numpy array, creates ``var_array`` by applying a polynomial function
              defined by the array of coefficients to the data. Only used by
              the "MLPCA" algorithm.
        reproject : {None, "signal", "navigation", "both"}, default None
            If not None, the results of the decomposition will be projected in
            the selected masked area.
        return_info: bool, default False
            The result of the decomposition is stored internally. However,
            some algorithms generate some extra information that is not
            stored. If True, return any extra information if available.
            In the case of sklearn.decomposition objects, this includes the
            sklearn Estimator object.
        print_info : bool, default True
            If True, print information about the decomposition being performed.
            In the case of sklearn.decomposition objects, this includes the
            values of all arguments of the chosen sklearn algorithm.
        svd_solver : {"auto", "full", "arpack", "randomized"}, default "auto"
            If auto:
                The solver is selected by a default policy based on `data.shape` and
                `output_dimension`: if the input data is larger than 500x500 and the
                number of components to extract is lower than 80% of the smallest
                dimension of the data, then the more efficient "randomized"
                method is enabled. Otherwise the exact full SVD is computed and
                optionally truncated afterwards.
            If full:
                run exact SVD, calling the standard LAPACK solver via
                :py:func:`scipy.linalg.svd`, and select the components by postprocessing
            If arpack:
                use truncated SVD, calling ARPACK solver via
                :py:func:`scipy.sparse.linalg.svds`. It requires strictly
                `0 < output_dimension < min(data.shape)`
            If randomized:
                use truncated SVD, calling :py:func:`sklearn.utils.extmath.randomized_svd`
                to estimate a limited number of components
        copy : bool, default True
            * If True, stores a copy of the data before any pre-treatments
              such as normalization in ``s._data_before_treatments``. The original
              data can then be restored by calling ``s.undo_treatments()``.
            * If False, no copy is made. This can be beneficial for memory
              usage, but care must be taken since data will be overwritten.
        **kwargs : extra keyword arguments
            Any keyword arguments are passed to the decomposition algorithm.


        Examples
        --------
        >>> s = exspy.data.EDS_TEM_FePt_nanoparticles()
        >>> si = hs.stack([s]*3)
        >>> si.change_dtype(float)
        >>> si.decomposition()

        See also
        --------
        vacuum_mask

        References
        ----------
        .. [*] M. Keenan and P. Kotula, "Accounting for Poisson noise
           in the multivariate analysis of ToF-SIMS spectrum images", Surf.
           Interface Anal 36(3) (2004): 203-212.
        """
        model_ = self.model_
        super().decomposition(
            normalize_poissonian_noise=normalize_poissonian_noise,
            navigation_mask=navigation_mask,
            *args,
            **kwargs,
        )
        self.model_ = model_

    def plot_1D_results(self, elements = []) :
        if not(isinstance(self.learning_results.decomposition_algorithm,NMFEstimator)) :
            raise ValueError("No espm learning results available, please run a decomposition with an espm algorithm first")
        
        W = self.learning_results.decomposition_algorithm.W_
        G = self.learning_results.decomposition_algorithm.G_
        H = self.learning_results.decomposition_algorithm.H_.mean(axis = 1)

        @symbol_to_number_list
        def convert_elts(elements = []) :
            return elements
        
        spectrum_1D = self.mean()
        spectrum_1D.plot(True)
        spectrum_1D._plot.signal_plot.ax.plot(spectrum_1D.axes_manager.signal_axes[0].axis, G@W@H, 'b-', label = 'Full model')
        
        conv_elts = convert_elts(elements = elements)
        conv_elts_dict = {conv_elts[i] : elt for i, elt in enumerate(elements)}
        line_styles = [ '--', '-.', ':']
        colors = ['g', 'r', 'c', 'm', 'y', 'k']
        
        _ = 0
        for elt in conv_elts:
            indices = [i for i, mod_elt in enumerate(self.metadata.EDS_model.elements) if str(elt) == mod_elt[:2]]
            if indices:
                component = sum(G[:,idx][:,np.newaxis] @ W[idx,:][:,np.newaxis] @ H for idx in indices)
                spectrum_1D._plot.signal_plot.ax.plot(self.axes_manager.signal_axes[0].axis, component, label=f'{conv_elts_dict[elt]}', linestyle=line_styles[_%len(line_styles)], color=colors[_%len(colors)])
                _+=1

        spectrum_1D._plot.signal_plot.ax.legend()

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

        
        @number_to_symbol_list
        def convert_elts(elements = []) :
            return elements

        elts = self.model.get_elements(False)
        elts_indices = self.model.NMF_simplex()

        if selected_elts : 
            conv_elts = convert_elts(elements=elts)
            conv_elts_dict = {conv_elts[i] : num for i, num in enumerate(elts_indices)}
            new_elts_indices = []
            for elt in selected_elts :
                if elt in conv_elts_dict.keys() :  
                    new_elts_indices.append(conv_elts_dict[elt])

            W = W[new_elts_indices,:]*100 /W[new_elts_indices,:].sum(axis = 0)
            if fit_error :
                errors = percentages[new_elts_indices, :]
                errors[errors > 10000] = np.inf
            else : 
                errors = np.zeros_like(W)

            return selected_elts, W, errors

        else : 
            conv_elts = convert_elts(elements=elts)

            W = W[elts_indices,:]*100 # /W[indices,:].sum(axis = 0)
            if fit_error :
                errors = percentages[elts_indices, :]
                errors[errors > 10000] = np.inf
            else : 
                errors = np.zeros_like(W)

            return conv_elts, W, errors
        
        # norm = self.metadata.EDS_model.norm

        
    
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
            print("\nDisclaimer : The presented errors correspond to the statistical error on the fitted intensity of the peaks according to a Poisson law.\nIn other words it corresponds to the precision of the measurment.\nThe accuracy of the measurment strongly depends on other factors such as absorption, cross-sections, etc...\nPlease consider these parameters when interpreting the results.")

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
        
    def set_analysis_parameters(
        self,
        thickness=None,
        density=None,
        detector_type=None,
        width_slope=None,
        width_intercept=None,
        geom_eff=None,
        xray_db=None
    ):
        r"""
        Set the relevant parameters for the analysis in the metadata of the :class:`EDSespm` object.

        Parameters
        ----------
        thickness : float
            Thickness of the sample in cm.
        density : float
            Density of the sample in g/cm^3.
        detector_type : str
            Type of the detector. The default is "SDD_efficiency.txt".
        width_slope : float
            Slope of the width of the peaks in the EDS spectrum.
        width_intercept : float
            Intercept of the width of the peaks in the EDS spectrum.
        geom_eff : float
            Geometric efficiency of the detector.
        acq_time : float
            Acquisition time of the spectrum in seconds.
        probe_current : float
            Probe current in A.
        xray_db : str
            Path to the xray database file. The default is "200keV_xrays.json".
        """
        md = self.metadata

        if thickness is not None:
            md.set_item("Sample.thickness", thickness)
        if density is not None:
            md.set_item("Sample.density", density)
        if detector_type is not None:
            md.set_item("Acquisition_instrument.TEM.Detector.EDS.type", detector_type)
        if width_slope is not None:
            md.set_item("Acquisition_instrument.TEM.Detector.EDS.width_slope", width_slope)
        if width_intercept is not None:
            md.set_item("Acquisition_instrument.TEM.Detector.EDS.width_intercept", width_intercept)
        if geom_eff is not None:
            md.set_item("Acquisition_instrument.TEM.Detector.EDS.geometric_efficiency", geom_eff)
        if xray_db is not None:
            md.set_item("xray_db", xray_db)

        try : 
            md.set_item("Acquisition_instrument.TEM.Detector.EDS.take_off_angle",
                        take_off_angle(tilt_stage = md.Acquisition_instrument.TEM.Stage.tilt_alpha,
                                       azimuth_angle = md.Acquisition_instrument.TEM.Detector.EDS.azimuth_angle,
                                       elevation_angle = md.Acquisition_instrument.TEM.Detector.EDS.elevation_angle,
                                       beta_tilt = md.Acquisition_instrument.TEM.Stage.tilt_beta))
        except AttributeError :
            print("You need to define the azimuth and elevation of the detector as well as the alpha and beta tilt of the sample holder. Please, use the set_microscope_parameters function.")


#######################
# Auxiliary functions #
#######################

def get_metadata(spim) : 
    r"""
    Get the metadata of the :class:`EDSespm` object and format it as a model parameters dictionary.
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



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

import json
import warnings

import hyperspy.api as hs
import hyperspy.events
import intervaltree
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# from functools import wraps
from exspy.signals import EDSTEMSpectrum
from exspy.utils.eds import take_off_angle
from hyperspy.roi import BaseROI, RectangularROI
from hyperspy.signal_tools import Signal1DRangeSelector
from hyperspy.ui_registry import get_gui
from prettytable import PrettyTable
from scipy.optimize import curve_fit
from tqdm import tqdm

from espm.conf import NUMBER_PERIODIC_TABLE
from espm.estimators import NMFEstimator, SmoothNMF
from espm.models import EDXS
from espm.utils import (
    get_explained_intensity_W,
    num_to_symbol,
    number_to_symbol_list,
    quant_spectrum,
    symbol_to_number_dict,
    symbol_to_number_list,
)

NPT = json.load(open(NUMBER_PERIODIC_TABLE))


def _check_decomposition(func) -> bool:
    # @wraps(func)
    def inner(instance, *args, **kwargs):
        est = instance.learning_results.decomposition_algorithm
        assert est is not None, (
            f"No decomposition was performed. Please, use the espm decomposition before running {func}"
        )
        assert issubclass(type(est), NMFEstimator), (
            f"To use {func} you need to use an object that inherits NMFEstimator, e.g. SmoothNMF."
        )
        return func(instance, *args, **kwargs)

    return inner


class EDSespm(EDSTEMSpectrum):
    _signal_type = "EDS_espm"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape_2d_ = None
        self._X = None
        self.G_ = None
        self.model_ = None
        self.custom_init_ = None
        self.ranges = None
        self.average_spectrum_ = None
        self.energy_axis_ = None
        self.elements_ = None

        self._set_default_analysis_params()

    ##############
    # Properties #
    ##############

    def _set_default_analysis_params(self) -> None:
        # TODO : make them fetch preferences from the user
        md = self.metadata
        md.Signal.signal_type = "EDS_espm"

        if "Acquisition_instrument.TEM.Detector.EDS.width_slope" not in md:
            md.set_item("Acquisition_instrument.TEM.Detector.EDS.width_slope", 0.01)
        if "Acquisition_instrument.TEM.Detector.EDS.width_intercept" not in md:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.width_intercept", 0.065
            )
        if "xrays_db" not in md:
            md.set_item("xray_db", "200keV_xrays.json")
        if "Acquisition_instrument.TEM.Detector.EDS.type" not in md:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.type", "SDD_efficiency.txt"
            )
        if "Acquisition_instrument.TEM.Stage.tilt_beta" not in md:
            md.set_item("Acquisition_instrument.TEM.Stage.tilt_beta", 0.0)

    def _check_metadata_G(self) -> None:
        md = self.metadata

        if "Sample.elements" not in md:
            raise ValueError(
                "The elements of the sample are missing in the metadata. Please use the set_elements method to set the elements."
            )
        if "Acquisition_instrument.TEM.beam_energy" not in md:
            raise ValueError(
                "The beam energy is missing in the metadata. Please use the set_microscope_parameters method to set the beam energy."
            )
        if "Sample.density" not in md:
            raise ValueError(
                "The density of the sample is missing in the metadata. Please use the set_analysis_parameters method to set the density."
            )
        if "Sample.thickness" not in md:
            raise ValueError(
                "The thickness of the sample is missing in the metadata. Please use the set_analysis_parameters method to set the thickness."
            )
        if "Acquisition_instrument.TEM.Detector.EDS.type" not in md:
            raise ValueError(
                "The detector type is missing in the metadata. Please use the set_analysis_parameters method to set the detector type."
            )
        if "Acquisition_instrument.TEM.Detector.EDS.take_off_angle" not in md:
            raise ValueError(
                "The take-off angle is missing in the metadata. Please use the set_microscope_parameters method to set the take-off angle."
            )
        if "Acquisition_instrument.TEM.Detector.EDS.width_slope" not in md:
            raise ValueError(
                "The width slope is missing in the metadata. Please use the set_analysis_parameters method to set the width slope."
            )
        if "Acquisition_instrument.TEM.Detector.EDS.width_intercept" not in md:
            raise ValueError(
                "The width intercept is missing in the metadata. Please use the set_analysis_parameters method to set the width intercept."
            )
        if "xray_db" not in md:
            raise ValueError(
                "The xray database is missing in the metadata. Please use the set_analysis_parameters method to set the xray database."
            )

    def _check_metadata_quantification(self) -> None:
        md = self.metadata

        if "Acquisition_instrument.TEM.Detector.EDS.geometric_efficiency" not in md:
            raise ValueError(
                "The geometric efficiency of the detector is missing in the metadata. Please use the set_analysis_parameters method to set the geometric efficiency."
            )
        if "Acquisition_instrument.TEM.beam_current" not in md:
            raise ValueError(
                "The beam current is missing in the metadata. Please use the set_microscope_parameters method to set the beam current."
            )
        if "Acquisition_instrument.TEM.Detector.EDS.real_time" not in md:
            raise ValueError(
                "The acquisition time is missing in the metadata. Please use the set_microscope_parameters method to set the acquisition time."
            )

    @property
    def custom_init(self) -> bool:
        r"""
        Boolean setting whether using the custom_init (see espm.models.EDXS) or not.
        If True, the custom_init will be used to initialise the decomposition.
        If False, the default initialisation will be used.
        If None, the  will be set to False.
        """
        return self.custom_init_

    @custom_init.setter
    def custom_init(self, value: bool) -> None:
        self.custom_init_ = value

    @property
    def shape_2d(self) -> tuple[int]:
        r"""
        Shape of the data in the spatial dimension.
        """
        if self.shape_2d_ is None:
            self.shape_2d_ = self.axes_manager[1].size, self.axes_manager[0].size
        return self.shape_2d_

    @property
    def X(self) -> np.ndarray:
        r"""
        The data in the form of a 2D array of shape (n_samples, n_features).
        """
        if self._X is None:
            shape = (
                self.axes_manager[1].size,
                self.axes_manager[0].size,
                self.axes_manager[2].size,
            )
            self._X = self.data.reshape((shape[0] * shape[1], shape[2])).T
        return self._X

    @property
    def G(self) -> np.ndarray:
        r"""
        The G matrix of the :class:`espm.models.EDXS` model corresponding to the metadata of the :class:`EDSespm` object.
        """
        if self.G_ is None:
            try:
                if self.problem_type == "identity":
                    return None
            except AttributeError:
                warnings.warn(
                    "You did not used the build_G method to build the G matrix. In ESpM-NMF, an idenity matrix will be used for decomposition"
                )
                return None
        return self.G_

    @property
    def model(self) -> EDXS:
        r"""
        The :class:`espm.models.EDXS` model corresponding to the metadata of the :class:`EDSespm` object.
        """
        if self.model_ is None:
            mod_pars = get_metadata(self)
            self.model_ = EDXS(**mod_pars, custom_init=self.custom_init_)
        return self.model_

    @property
    def energy_axis(self):
        if self.energy_axis_ is None:
            self.energy_axis_ = self.axes_manager.signal_axes[0].axis
        return self.energy_axis_

    @property
    def average_spectrum(self):
        if self.average_spectrum_ is None:
            nav_axes = self.axes_manager.navigation_axes
            if len(nav_axes) > 0:
                self.average_spectrum_ = self.mean(axis=nav_axes).data
            else:
                self.average_spectrum_ = self.data
        return self.average_spectrum_

    @property
    def elements(self):
        if self.elements_ is None:

            @symbol_to_number_list
            def convert_to_numbers(elements):
                return elements

            elements = convert_to_numbers(elements=self.metadata.Sample.elements)
            self.elements_ = [str(el) for el in elements]
        return self.elements_

    def build_G(
        self,
        problem_type: str = "bremsstrahlung",
        ignored_elements: list[str] = ["Cu"],
        *,
        elements_dict: dict[str, float] = {},
        use_calibration: bool = False,
        **kwargs,
    ) -> None:
        r"""
        Build the G matrix of the :class:`espm.models.EDXS` model corresponding to the metadata of the :class:`EDSespm` object and stores it as an attribute.

        Parameters
        ----------
        problem_type : str, optional
            Determines the type of the G matrix to build. It can be "bremsstrahlung", "no_brstlg" or "identity". The parameters correspond to:
                - "bremsstrahlung" : the G matrix is a callable with both characteristic X-rays and a bremsstrahlung model.
                - "no_brstlg" : the G matrix is a matrix with only characteristic X-rays.
                - "identity" : the G matrix is None which is equivalent to an identity matrix for espm functions.
        ignored_elements : list, optional
            List of chemical elements to ignore when building the G matrix.
        use_calibration : bool, optional
            If True, build G using a calibrated/fitted peak table (from fit_table or poly_fit).
        elements_dict : dict, optional
            Dictionary containing atomic numbers and a corresponding cut-off energies. It is used to separate the characteristic X-rays of the given elements into two energies ranges and assign them each a column in the G matrix instead of having one column per element.
            For example elements_dict = {"26",3.0} will separate the characteristic X-rays of the element Fe into two energies ranges and assign them each a column in the G matrix. This is useful to circumvent issues with the absorption.
        use_calibration: bool, optional
            Whether to use automatic calibration to build G.
            Defaults to `False`.
        **kwargs : dict
            Additional arguments to pass to fit_table or poly_fit (e.g., window_mult, filter_cs).
        Returns
        -------
        None
        """
        self._check_metadata_G()
        self.problem_type = problem_type
        self.separated_lines = elements_dict

        g_pars = {
            "g_type": problem_type,
            "ignored_elements": ignored_elements,
            "elements": self.metadata.Sample.elements,
            "elements_dict": elements_dict,
            "use_calibration": use_calibration,
        }

        self.model.generate_g_matr(**g_pars)
        self.G_ = self.model.G

        # Storing the model parameters in the metadata so that the decomposition does not erase them
        # Indeed the decomposition re-creates a new object of the same class when it is called
        self.metadata.EDS_model = {}
        self.metadata.EDS_model.problem_type = problem_type
        self.metadata.EDS_model.separated_lines = elements_dict
        self.metadata.EDS_model.elements = self.model.model_elts
        self.metadata.EDS_model.norm = self.model.norm

    def set_analysis_parameters(
        self,
        thickness: float = None,
        density: float = None,
        detector_type: str | dict = None,
        width_slope: float = None,
        width_intercept: float = None,
        geom_eff: float = None,
        xray_db: str = None,
    ) -> None:
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
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.width_slope", width_slope
            )
        if width_intercept is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.width_intercept",
                width_intercept,
            )
        if geom_eff is not None:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.geometric_efficiency", geom_eff
            )
        if xray_db is not None:
            md.set_item("xray_db", xray_db)

        try:
            md.set_item(
                "Acquisition_instrument.TEM.Detector.EDS.take_off_angle",
                take_off_angle(
                    tilt_stage=md.Acquisition_instrument.TEM.Stage.tilt_alpha,
                    azimuth_angle=md.Acquisition_instrument.TEM.Detector.EDS.azimuth_angle,
                    elevation_angle=md.Acquisition_instrument.TEM.Detector.EDS.elevation_angle,
                    beta_tilt=md.Acquisition_instrument.TEM.Stage.tilt_beta,
                ),
            )
        except AttributeError:
            print(
                "You need to define the azimuth and elevation of the detector as well as the alpha and beta tilt of the sample holder. Please, use the set_microscope_parameters function."
            )

    ############################
    # Helper functions for NMF #
    ############################

    def carto_fixed_W(self, brstlg_comps: int = 1) -> np.ndarray:
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
        if self.G_ is None:
            raise ValueError(
                "The G matrix has not been built yet. Please use the build_G method."
            )
        elements = self.metadata.EDS_model.elements
        if self.problem_type == "no_brstlg":
            W = np.diag(-1 * np.ones((len(elements),)))
        elif self.problem_type == "bremsstrahlung":
            W1 = np.diag(-1 * np.ones((len(elements),)))
            W2 = np.zeros((2, len(elements)))
            W_elts = np.vstack((W1, W2))
            W3 = np.zeros((len(elements), brstlg_comps))
            W4 = -1 * np.ones((2, brstlg_comps))
            W_brstlg = np.vstack((W3, W4))
            W = np.hstack((W_elts, W_brstlg))

        return W

    def set_fixed_W(self, phases_dict: dict[str, float]) -> np.ndarray:
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
        if self.G_ is None:
            raise ValueError(
                "The G matrix has not been built yet. Please use the build_G method."
            )
        raw_elts = self.metadata.EDS_model.elements
        elements = self.model.get_elements()
        indices = self.model.NMF_simplex()

        # convert elements to symbols but also omitting splitted lines
        @number_to_symbol_list
        def convert_to_symbols(elements=[]):
            return elements

        conv_elts = convert_to_symbols(elements=elements)

        if self.problem_type == "no_brstlg":
            W = -1 * np.ones((len(raw_elts), len(phases_dict.keys())))
        elif self.problem_type == "bremsstrahlung":
            W = -1 * np.ones((len(raw_elts) + 2, len(phases_dict.keys())))
        else:
            raise ValueError(
                "problem type should be either no_brstlg or bremsstrahlung"
            )
        for p, phase in enumerate(phases_dict):
            for key in phases_dict[phase]:
                if key == "b0":
                    if self.problem_type == "bremsstrahlung":
                        W[-2, p] = phases_dict[phase][key]
                    else:
                        warnings.warn(
                            "The chosen EDXS modelling does not incorporate the bremsstrahlung. Input bremsstrahlung parameters will be ignored."
                        )
                if key == "b1":
                    if self.problem_type == "bremsstrahlung":
                        W[-1, p] = phases_dict[phase][key]
                    else:
                        warnings.warn(
                            "The chosen EDXS modelling does not incorporate the bremsstrahlung. Input bremsstrahlung parameters will be ignored."
                        )
                if key in conv_elts:
                    W[indices[conv_elts.index(key)], p] = phases_dict[phase][key]
        return W

    def print_concentration_report(
        self,
        selected_elts: list[str] = [],
        W_input: np.ndarray = None,
        fit_error: bool = True,
        disclaimer: bool = True,
    ) -> None:
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
        conv_elts, W, errors = self.concentration_report(
            selected_elts=selected_elts, W_input=W_input, fit_error=fit_error
        )

        table = PrettyTable()
        field_list = ["Elements"]
        for i in range(W.shape[1]):
            field_list.append("p" + str(i) + " (at.%)")
            if fit_error:
                field_list.append("p" + str(i) + " std (%)")
        table.field_names = field_list
        for i, j in enumerate(conv_elts):
            row = [j]
            for k in range(W.shape[1]):
                row.append(W[i, k])
                if fit_error:
                    row.append(errors[i, k])
            table.add_row(row)

        table.float_format = "0.3"
        table.align = "r"
        table.align["Elements"] = "l"
        # table.set_style(MSWORD_FRIENDLY)

        print(table)
        if disclaimer and fit_error:
            print(
                "\nDisclaimer : The presented errors correspond to the statistical error on the fitted intensity of the peaks according to a Poisson law.\nIn other words it corresponds to the precision of the measurment.\nThe accuracy of the measurment strongly depends on other factors such as absorption, cross-sections, etc...\nPlease consider these parameters when interpreting the results."
            )

    ############################
    # Bremsstrahlung functions #
    ############################

    def estimate_mass_thickness(
        self,
        ignored_elements: list[str] = ["Cu"],
        tol: float = 1e-8,
        *,
        elements_dict: dict = {},
    ) -> None:
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
        if len(self.axes_manager.navigation_axes) > 0:
            raise NotImplementedError(
                "For now this function is not fully implemented for spectrum images. Use this on an extracted 1D spectrum."
            )
        curr_X = self.data

        # First init of fit
        self.build_G(ignored_elements=ignored_elements, elements_dict=elements_dict)
        estimator = SmoothNMF(n_components=1, G=self.model)
        estimator.fit(curr_X[:, np.newaxis])
        H_init = estimator.H_
        W_init = estimator.W_
        elts = list(self.model.get_elements(include_ignored=False))
        elts_indices = self.model.NMF_simplex()
        new_elts_dict = {elts[i]: W_init[elts_indices[i]] for i in range(len(elts))}

        _ = 0
        curr_mt = self.metadata.Sample.thickness * self.metadata.Sample.density
        while _ < 5:
            # first init of the model
            brstlg_model, mask = self.model.bremsstrahlung_only_tools(
                mass_thickness=curr_mt, elements_dict=new_elts_dict, ranges=self.ranges
            )
            masked_X = curr_X[mask]
            brstlg_estimator = SmoothNMF(
                n_components=1, G=brstlg_model, fixed_H=H_init, tol=tol
            )
            brstlg_estimator.fit(masked_X[:, np.newaxis])
            W_brstlg = np.vstack(
                (
                    -1
                    * np.ones(
                        (
                            W_init.shape[0] - brstlg_estimator.W_.shape[0],
                            brstlg_estimator.W_.shape[1],
                        )
                    ),
                    brstlg_estimator.W_,
                )
            )

            self.build_G(ignored_elements=ignored_elements, elements_dict=elements_dict)
            # First estimation of the bremsstrahlung + elts
            estimator = SmoothNMF(
                n_components=1, G=self.model, fixed_W=W_brstlg, tol=tol
            )
            estimator.fit(curr_X[:, np.newaxis])

            # Get the elements, their concentrations and the mass_thickness value
            W_init = estimator.W_
            H_init = estimator.H_

            elts = list(self.model.get_elements(include_ignored=False))
            elts_indices = self.model.NMF_simplex()
            new_elts_dict = {elts[i]: W_init[elts_indices[i]] for i in range(len(elts))}
            total_weight = self._elements_dict_to_weights(new_elts_dict)
            curr_mt = self._extract_mass_thickness(H_init.sum(), total_weight)

            _ += 1

            print(
                "The current estimated mass-thickness is {} g.cm^-2".format(curr_mt),
                flush=True,
            )

        self.plot(True)
        self._plot.signal_plot.ax.set_title(
            "Estimated mass-thickness : {} g.cm^-2".format(curr_mt)
        )

        axis = self.energy_axis
        self._plot.signal_plot.ax.plot(
            axis, estimator.G_ @ estimator.W_ @ estimator.H_, "b-", label="Full model"
        )
        self._plot.signal_plot.ax.plot(
            axis[mask],
            brstlg_estimator.G @ brstlg_estimator.W_ @ brstlg_estimator.H_,
            "g.",
            label="Bremmstrahlung",
        )
        self._plot.signal_plot.ax.legend()

        self.metadata.Sample.thickness = 1.0
        self.metadata.Sample.density = curr_mt

    def _elements_dict_to_weights(self, elements_dict):
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
            quantity * NPT["table"][element]["atomic_mass"] * 1.66053906660e-24
            for element, quantity in elements_dict.items()
        )
        return total_weight

    def _extract_mass_thickness(self, H_value, total_weight):
        Na = 6.02214179e23  # TODO : Check the usefulness of Na.
        # If I am correct the concentrations we guess have no unit.
        # Since they are not in mole, no need to normalize using Na
        Ne = 6.25e18  # Number of electrons in a Coulomb
        # real time shound be the whole acquisition time (without dead time but with all pixels)
        return (
            H_value
            * total_weight
            / (
                self.metadata.Acquisition_instrument.TEM.beam_current
                * 1e-9
                * self.metadata.Acquisition_instrument.TEM.Detector.EDS.real_time
                * Ne
                * self.model.norm[0][0]
                * (
                    self.metadata.Acquisition_instrument.TEM.Detector.EDS.geometric_efficiency
                    / (4 * np.pi)
                )
            )
        )

    def select_background_windows(self, num_windows=4, ranges=None):
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
        if self.model_ is None:
            raise ValueError(
                "The G matrix has not been built yet. Please use the build_G method."
            )
        if ranges is not None:
            self.ranges = ranges
            self.model.ranges = self.ranges
        else:
            if len(self.axes_manager.navigation_axes) > 0:
                raise NotImplementedError(
                    "For now this function is not fully implemented for spectrum images. Use this on an extracted 1D spectrum."
                )
            cm = self._register_ranges
            init_ranges = self._generate_ranges(num_windows)
            self.spans = []
            for i in range(num_windows):
                self.spans.append(Signal1DRangeSelector(self))

            for j, span in enumerate(self.spans):
                span.span_selector.extents = init_ranges[j]
                span.on_close.append((cm, self))
                get_gui(span, toolkey="hyperspy.interactive_range_selector")

    def _register_ranges(self, signal, left, right):
        # The unused args are required for the event to properly complete
        coord_list = [[span.ss_left_value, span.ss_right_value] for span in self.spans]
        for coords in coord_list:
            if np.nan in coords:
                # TODO : needs to be improved in future versions.
                raise ValueError(
                    "You have to click and drag each area at least so that the calculated bremsstrahlung is displayed."
                )
        coord_list.sort(key=lambda coord: coord[0])
        tree = intervaltree.IntervalTree.from_tuples(coord_list)
        self.ranges = []
        for branch in tree:
            self.ranges.append([branch[0], branch[1]])

        self.model.ranges = self.ranges

        model = self._compute_bremsstrahlung()
        self._plot_background(model)

    def _compute_bremsstrahlung(self):
        mt = self.metadata.Sample.density * self.metadata.Sample.thickness
        elts_dict = {elt: 1.0 for elt in self.metadata.Sample.elements}
        brstlg_model, mask = self.model.bremsstrahlung_only_tools(
            mass_thickness=mt, elements_dict=elts_dict, ranges=self.ranges
        )
        curr_X = self.data
        masked_X = curr_X[mask]

        # Estimate the bremsstrahlung on the partial data
        brstlg_estimator = SmoothNMF(n_components=1, G=brstlg_model)
        brstlg_estimator.fit(masked_X[:, np.newaxis])
        # get the fitting results
        fH = brstlg_estimator.H_
        fW = brstlg_estimator.W_

        # Now adapt to the full range
        # It is not super efficient but I think it is not an issue. The model can't be easily continued, it is not a function.
        axis = self.axes_manager.signal_axes[0]
        full_range = [[axis.low_value, axis.high_value]]
        full_brstlg_model, full_mask = self.model.bremsstrahlung_only_tools(
            mass_thickness=mt, elements_dict=elts_dict, ranges=full_range
        )

        return full_brstlg_model @ fW @ fH

    def _plot_background(self, model):
        # The full range from self.compute_background misses both ends
        # The axis needs to be trimmed accordingly
        axis = self.energy_axis[1:-1]
        self._plot.signal_plot.ax.plot(axis, model)

    def _generate_ranges(self, num):
        axis = self.axes_manager.signal_axes[0]
        bounds = (axis.low_value, axis.high_value)
        values = np.linspace(bounds[0], bounds[1], num=2 * num + 2)
        ranges_list = [(values[2 * i - 1], values[2 * i]) for i in range(1, num + 1)]
        return ranges_list

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

    @_check_decomposition
    def plot_1D_results(
        self, xray_lines: bool = True, elements: list[str] = []
    ) -> None:
        r"""
        Plots each spectrum (component) resulting from an espm decompositions.
        It shows the contribution of each selected element and the total model.

        Parameters
        ----------
        elements : list[str]
            list of elements to display

        Results
        -------
        None
        """
        W = self.learning_results.decomposition_algorithm.W_
        G = self.learning_results.decomposition_algorithm.G_
        H = self.learning_results.decomposition_algorithm.H_.mean(axis=1)

        @symbol_to_number_list
        def convert_elts(elements=[]):
            return elements

        spectrum_1D = self.mean()
        spectrum_1D.plot(xray_lines=xray_lines)
        spectrum_1D._plot.signal_plot.ax.plot(
            spectrum_1D.axes_manager.signal_axes[0].axis,
            G @ W @ H,
            "b-",
            label="Full model",
        )

        conv_elts = convert_elts(elements=elements)
        conv_elts_dict = {conv_elts[i]: elt for i, elt in enumerate(elements)}
        line_styles = ["--", "-.", ":"]
        colors = ["g", "r", "c", "m", "y", "k"]

        _ = 0
        for elt in conv_elts:
            indices = [
                [i]
                for i, mod_elt in enumerate(self.metadata.EDS_model.elements)
                if str(elt) == mod_elt[:2]
            ]
            if indices:
                component = sum([G[:, idx] @ W[idx, :] @ H for idx in indices])
                spectrum_1D._plot.signal_plot.ax.plot(
                    self.energy_axis,
                    component,
                    label=f"{conv_elts_dict[elt]}",
                    linestyle=line_styles[_ % len(line_styles)],
                    color=colors[_ % len(colors)],
                )
                _ += 1

        spectrum_1D._plot.signal_plot.ax.legend()

    def concentration_report(self, selected_elts=[], W_input=None, fit_error=True):
        if W_input is None:
            assert isinstance(
                self.learning_results.decomposition_algorithm, NMFEstimator
            ), (
                "No espm learning results available, please run a decomposition with an espm algorithm first"
            )

            W = self.learning_results.decomposition_algorithm.W_
            G = self.learning_results.decomposition_algorithm.G_
            H = self.learning_results.decomposition_algorithm.H_
            N = get_explained_intensity_W(G, W, H)
            sqN = np.sqrt(N)
            percentages = sqN / N * 100

        else:
            W = W_input
            fit_error = False

        @number_to_symbol_list
        def convert_elts(elements=[]):
            return elements

        elts = self.model.get_elements(False)
        elts_indices = self.model.NMF_simplex()

        if selected_elts:
            conv_elts = convert_elts(elements=elts)
            conv_elts_dict = {conv_elts[i]: num for i, num in enumerate(elts_indices)}
            new_elts_indices = []
            for elt in selected_elts:
                if elt in conv_elts_dict.keys():
                    new_elts_indices.append(conv_elts_dict[elt])

            W = W[new_elts_indices, :] * 100 / W[new_elts_indices, :].sum(axis=0)
            if fit_error:
                errors = percentages[new_elts_indices, :]
                errors[errors > 10000] = np.inf
            else:
                errors = np.zeros_like(W)

            return selected_elts, W, errors

        else:
            conv_elts = convert_elts(elements=elts)

            W = W[elts_indices, :] * 100  # /W[indices,:].sum(axis = 0)
            if fit_error:
                errors = percentages[elts_indices, :]
                errors[errors > 10000] = np.inf
            else:
                errors = np.zeros_like(W)

            return conv_elts, W, errors

    def estimate_best_binning(self, inspect=False):
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

        facx = np.arange(1, self.axes_manager[0].size // 2 + 1)
        facy = np.arange(1, self.axes_manager[1].size // 2 + 1)

        binx = [self.axes_manager[0].size / i for i in facx]
        biny = [self.axes_manager[1].size / i for i in facy]
        vars_est = np.array([])
        biases_est = np.array([])
        bprod = []
        for i in zip(binx, biny):
            bprod.append((i[0], i[1]))

        for i in tqdm(bprod):
            # Bin the measurement dataset and upsample to bring it back to its original dimensionality
            B = i[0] * i[1]
            binned = self.rebin(scale=(i[0], i[1], 1))
            upsampled = binned.rebin(
                new_shape=(
                    self.axes_manager[0].size,
                    self.axes_manager[1].size,
                    self.axes_manager[2].size,
                )
            )
            upsampled_data = upsampled.data
            data = self.data

            # Estimator of variance (Lemma 4.3) - \widehat{Var} (\hat{y}_i) = \alpha ^2 y_{i}+ (1-\alpha)^2 \sum_{k \in \mathcal{K}} (w_k^2 n_{i,k})
            vars_est = np.append(vars_est, np.mean(upsampled_data * 1 / B))

            # Estimator of squared bias (Lemma 4.4) - \widehat{Bias^2}(\hat{y}_i) = (1-\alpha)^2\left((y_{n_i} - y_{i})^2 - \sum_{k\in \mathcal{K}} w_k^2y_{i,k} - y_{i} \right)
            biases_est = np.append(
                biases_est,
                np.mean(
                    (data - upsampled_data) ** 2
                    - 1 / B * upsampled_data
                    - (1 - 2 / B) * data
                ),
            )

        mprimes_est = vars_est * K / L + biases_est
        estimated_binning = (
            bprod[np.argmin(mprimes_est)][0],
            bprod[np.argmin(mprimes_est)][1],
            1,
        )
        if inspect:
            return mprimes_est, estimated_binning
        else:
            return estimated_binning

    def define_ROI(self, xray_lines: bool = True) -> RectangularROI:
        r"""
        A function to define a rectangular ROI on an HyperSpy EDXS signal.

        Parameters
        ----------
        xray_lines : bool
            If True, it displays the xray lines markers ok exspy. It might slow down the execution though.

        Returns
        -------
        roi : hs.roi.RectangularROI
            A rectangular ROI defined by the user.
        """
        scale_x = self.axes_manager[0].scale
        scale_y = self.axes_manager[1].scale

        centre_x = self.data.shape[1] * scale_x / 2
        centre_y = self.data.shape[0] * scale_y / 2
        dx = self.data.shape[1] * scale_x / 10
        dy = self.data.shape[0] * scale_y / 10

        roi = RectangularROI(
            left=centre_x - dx,
            top=centre_y - dy,
            right=centre_x + dx,
            bottom=centre_y + dy,
        )
        self.plot()
        imr = roi.interactive(self, color="r")
        selected_spectrum = hs.interactive(imr.mean)
        selected_spectrum.metadata.General.title = "Spectrum of the selected area"
        selected_spectrum.plot(xray_lines=xray_lines)

        return roi

    def generate_part_fixed_H_matrix(
        self, rois: list[RectangularROI] = None, value: float = 1
    ) -> np.ndarray:
        r"""
        A function to generate a component of the fixed H matrix for one phase.

        Parameters
        ----------
        ROIs : list
            A list of rectangular ROIs given by the user.
        value : float
            Value of the non-negative entries in the partial H matrix. Must be between 0 and 1.

        Returns
        -------
        part_f_H : np.ndarray
            A fixed H matrix for one phase.
        """
        part_f_H = (-1) * np.ones(
            shape=(self.data.shape[0], self.data.shape[1]), dtype=float
        )

        if value < 0:
            raise ValueError("Value must be above 0.")
        # if value < 1 :
        #     print("The value of some of the fixed_H matrix is below 1.0.",
        #           "In most cases, it means that you want to use simplex_H = True and simplex_W = False in smoothNMF decomposition.")

        if rois is None or rois == []:
            return part_f_H

        # We don't enter that part of the code if there are no ROIs so it should be fine
        if issubclass(type(rois[0]), BaseROI):
            for roi in rois:
                region_parameters = roi.parameters
                scale_i = self.axes_manager[0].scale
                scale_j = self.axes_manager[1].scale
                j_min = int(region_parameters["left"] // scale_j)
                i_min = int(region_parameters["top"] // scale_i)
                j_max = int(region_parameters["right"] // scale_j)
                i_max = int(region_parameters["bottom"] // scale_i)
                part_f_H[i_min:i_max, j_min:j_max] = value

        return part_f_H

    def set_fixed_H(self, areas_dict: dict[str, np.ndarray]) -> np.ndarray:
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

        H = (-1) * np.ones(
            shape=(len(areas_dict), self.data.shape[0], self.data.shape[1]), dtype=float
        )

        for i, p in enumerate(areas_dict):
            H[i, :, :] = areas_dict[p]

        return H.reshape((len(areas_dict), self.data.shape[0] * self.data.shape[1]))

    @_check_decomposition
    def elemental_mapping(self, *, skipped_elements: list = []) -> None:
        r"""
        Performs pixel-wise elemental quantification using the results of an espm decomposition.
        Results are stored in self.quantification_signal and self.quantification_list.
        Sets self.quantification_list, self.quantification_signal, self.quantification_signal_1d.

        Parameters
        ----------
        elements : list
            List of elements that will not be quantified, therefore renormalizing the remaining ones.
        Returns
        -------
        None
        """

        est = self.learning_results.decomposition_algorithm

        _W = est.W_
        H = est.H_

        @number_to_symbol_list
        def sym_elts(elements=[]):
            return elements

        @symbol_to_number_list
        def num_elts(elements=[]):
            return elements

        _elts = sym_elts(elements=self.model.get_elements(False))
        _elts_indices = self.model.NMF_simplex()

        skipped_elts = sym_elts(elements=skipped_elements)
        elts = []
        elts_indices = []
        for i, elt in enumerate(_elts):
            if elt in skipped_elts:
                pass
            else:
                elts.append(elt)
                elts_indices.append(_elts_indices[i])

        W = _W[elts_indices, :]
        WH = W @ H
        WH = WH.reshape([W.shape[0]] + list(self.data.shape[:-1]))

        WH /= WH.sum(0)[np.newaxis, ...] / 100

        if self.axes_manager.navigation_dimension == 2:
            Signal = hs.signals.Signal2D
        elif self.axes_manager.navigation_dimension == 1:
            Signal = hs.signals.Signal1D

        qs = [
            Signal(
                WH[i],
                metadata={"General": {"name": el, "title": el + " Quantification"}},
                colorbar_label="A",
            )
            for i, el in enumerate(elts)
        ]

        for q in qs:
            for i in range(self.axes_manager.navigation_dimension):
                q.axes_manager[i].update_from(
                    self.axes_manager[i], ["units", "scale", "name", "offset"]
                )
            q.metadata.Signal.quantity = "Atomic %"
            wh = Signal(WH)

        for i in range(self.axes_manager.navigation_dimension):
            wh.axes_manager[1 + i].update_from(
                self.axes_manager[i], ["units", "scale", "name", "offset"]
            )

        self.quantification_list = qs
        self.quantification_signal = wh

        for k, m in self.metadata:
            self.quantification_signal.metadata.set_item(k, m)
        self.quantification_signal.metadata.set_item("Sample.elements", elts)
        self.quantification_signal.metadata.set_item("Signal.quantity", "Atomic %")
        self.quantification_signal.axes_manager[0].name = "Elements"
        if self.quantification_signal.metadata.has_item("Sample.xray_lines"):
            self.quantification_signal.metadata.set_item(
                "Sample.xray_lines",
                [
                    i
                    for i in self.quantification_signal.metadata.Sample.xray_lines
                    if i.split("_")[0] not in skipped_elts
                ],
            )
        self.quantification_signal_1d = self.quantification_signal.as_signal1D(0)

        # hack to label elements. Horrible, I know.
        def label_elements():
            self.quantification_signal._plot.navigator_plot.ax.set_xticks(
                list(range(len(elts))), elts
            )
            return

        self.quantification_signal.axes_manager[0].events.index_changed.connect(
            label_elements, []
        )

        hs.plot.plot_images(qs)

        return

    @_check_decomposition
    def plot_comp_model(self, comp_index: int):
        r"""
        Plots espm model of the component #comp_index, showing contributions of each element and background.

        Parameters
        ----------
        comp_index : int
            Component index for which the model plot is built.

        Returns
        -------
        Figure : matplotlib.pyplot.figure
            The figure of the plot.
        """

        idx = comp_index
        els = self.get_full_el_list() + ["Background 1", "Background 2"]
        self.GW = (
            self.learning_results.decomposition_algorithm.G_
            @ self.learning_results.decomposition_algorithm.W_
        )
        gs, cs = self.learning_results.decomposition_algorithm.W_.shape
        G_idx = (
            self.learning_results.decomposition_algorithm.G_
            * self.learning_results.decomposition_algorithm.W_[:, idx]
        )

        x = self.axes_manager[-1].axis
        plt.figure()
        plt.plot(x, self.GW[:, idx], "k--", label="Component")
        for i in range(gs):
            color = list(mpl.colors.TABLEAU_COLORS.values())[i % 10]
            plt.plot(x, G_idx[:, i], color=color, label=els[i])
            plt.fill_between(x, G_idx[:, i], alpha=0.4, color=color)
        plt.legend()
        ax = plt.gca()
        plt.title("Model of component {}".format(str(idx)))

        return plt.gcf()

    @_check_decomposition
    def plot_data_model(self):
        r"""
        Plots espm decomposition model as a new signal, showing contributions of each element and background
        to each pixel.

        Parameters
        ----------
        WH : np.ndarray
            WH model. by default is taken from the current decomposition.

        Returns
        -------
        None
        """

        W = self.learning_results.decomposition_algorithm.W_
        G = self.learning_results.decomposition_algorithm.G_
        H = self.learning_results.decomposition_algorithm.H_
        if self.learning_results.navigation_mask is not None:
            H = self.fix_masked_H()

        WH = W @ H

        contributions = [
            hs.signals.Signal1D((G[:, [i]] @ (WH)[[i], :]).T.reshape(self.data.shape))
            for i in range(G.shape[1])
        ]
        contributions.append(hs.signals.Signal1D((G @ WH).T.reshape(self.data.shape)))
        els = self.metadata.EDS_model.elements
        titles = self.get_full_el_list() + [
            "Background 1",
            "Background 2",
            "Full Model",
        ]
        for i, c in enumerate(contributions):
            for a, b in zip(c.axes_manager._axes, self.axes_manager._axes):
                a.update_from(b)
            c.metadata.General.title = titles[i]

        if self.axes_manager.navigation_dimension == 1:
            nav = self.sum(-1).as_signal1D(0)
            position = hs.roi.Point1DROI(0)
            nav_kwargs = {"color": "blue"}

        elif self.axes_manager.navigation_dimension == 2:
            nav = self.sum(-1).as_signal2D((0, 1))
            position = hs.roi.Point2DROI(0, 0)
            nav_kwargs = {}

        nav.plot(**nav_kwargs)
        position_interactive = position.interactive(self, nav, color="red")
        positions_contribs = [position.interactive(g, None) for g in contributions]
        hs.plot.plot_spectra(
            [position_interactive] + positions_contribs,
            legend="auto",
            linestyle=["-"] + ["--" for i in positions_contribs],
            color=["k"] + list(mpl.colors.TABLEAU_COLORS.values()) * 10,
        )

        return

    @_check_decomposition
    def plot_data_model_ROI(self):
        r"""
        Plots ESPM EDXS model fit results on the experimental data, summed over the chosen region of interest.

        Parameters
        ----------
        None :
            The data are taken from a previous decomposition.

        Returns
        -------
        None
        """

        W = self.learning_results.decomposition_algorithm.W_
        G = self.learning_results.decomposition_algorithm.G_
        H = self.learning_results.decomposition_algorithm.H_
        if self.learning_results.navigation_mask is not None:
            H = self.fix_masked_H()

        WH = np.matmul(W, H)

        contributions = [
            hs.signals.Signal1D((G[:, [i]] @ (WH)[[i], :]).T.reshape(self.data.shape))
            for i in range(G.shape[1])
        ]
        contributions.append(
            hs.signals.Signal1D((np.matmul(G, WH)).T.reshape(self.data.shape))
        )

        titles = self.get_full_el_list() + [
            "Background 1",
            "Background 2",
            "Full Model",
        ]
        for i, c in enumerate(contributions):
            for a, b in zip(c.axes_manager._axes, self.axes_manager._axes):
                a.update_from(b)
            c.metadata.General.title = titles[i]

        fig, ax = plt.subplots()
        self.plot()

        roi = hs.roi.RectangularROI(
            left=0,
            top=0,
            right=self.axes_manager[1].size,
            bottom=self.axes_manager[0].size,
        )

        imr = roi.interactive(self, color="green").sum(axis=0).sum(axis=0)
        contributions_roi = [
            roi.interactive(g, None).sum(axis=0).sum(axis=0) for g in contributions
        ]

        spectra = [imr] + contributions_roi
        lines = []

        (line,) = ax.plot(imr.data, label=imr.metadata.General.title, linestyle="-")
        lines.append(line)

        for spectrum in contributions_roi:
            (line,) = ax.plot(
                spectrum.data, label=spectrum.metadata.General.title, linestyle="--"
            )
            lines.append(line)

        ax.legend()

        def update_plot(*args, **kwargs):
            imr = roi.interactive(self, color="green").sum(axis=0).sum(axis=0)
            contributions_roi = [
                roi.interactive(g, None).sum(axis=0).sum(axis=0) for g in contributions
            ]

            all_data = [imr] + contributions_roi
            for line, new_data in zip(lines, all_data):
                line.set_ydata(new_data.data)

            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()

        roi.events.changed.connect(update_plot)
        update_plot()
        plt.show()

        return

    @_check_decomposition
    def elemental_profile(self, **kwargs):
        r"""
        Plots quantification profiles of all elements.

        Parameters
        ----------
        **kwargs are passed to Line2DROI

        Returns
        -------
        Quantification profiles
        """
        line = hs.roi.Line2DROI(**kwargs)
        p1 = hs.signals.Signal2D(self.data.sum(-1))
        if self.learning_results.navigation_mask is not None:
            p1.data[
                self.learning_results.navigation_mask.reshape(self.data.shape[:-1])
            ] = np.nan
        for i in range(2):
            p1.axes_manager[i].update_from(
                self.axes_manager[i], ["units", "scale", "name", "offset"]
            )
        p1.plot()
        line.interactive(p1, color="red")
        p_contrib = [
            line.interactive(g, None) for g in self.quantification_list
        ]  # here are the profiles stored
        for p in p_contrib:
            p.axes_manager[0].name = "Profile"
            p.metadata.Signal.quantity = "Atomic %"
        hs.plot.plot_spectra(p_contrib, legend="auto")
        ax = plt.gca()
        ax.set_ylabel("Atomic %")
        return p_contrib

    def calibrate_from_lines(self) -> None:
        r"""
        Selects two Xray lines from a 1D EDXS spectrum for further calibration by the apply_interactive_calibration method.
        """
        print("Instructions : ")
        print(
            "1. Select two X-ray lines (ideally far apart). The energies of the fitted peaks appears nearby."
        )
        print(
            "2. Note down those values and use : apply_interactive_calibration(e1,e2), where e1 and e2 are the energy values in keV."
        )
        self._gauss_means = np.zeros(2)
        a = self.sum((0, 1))
        b = a.deepcopy()
        b.data = np.zeros(b.data.shape)
        a.plot()
        hs.plot.plot_spectra(
            [b, b], fig=plt.gcf(), ax=plt.gca(), color="k", linewidth=2
        )
        eax = self.axes_manager[-1].axis
        ne = self.axes_manager[-1].axis.shape[0]

        roi1 = hs.roi.SpanROI(left=eax[ne // 5], right=eax[2 * ne // 5])
        roi_signal1 = roi1.interactive(a, color="blue")

        roi2 = hs.roi.SpanROI(left=eax[3 * ne // 5], right=eax[4 * ne // 5])
        roi_signal2 = roi2.interactive(a)

        hs.interactive(
            self.fit_plot_gauss,
            event=roi1.events.changed,
            roi_signal=roi_signal1,
            a=a,
            roi=roi1,
            i=1,
        )
        hs.interactive(
            self.fit_plot_gauss,
            event=roi2.events.changed,
            roi_signal=roi_signal2,
            a=a,
            roi=roi2,
            i=2,
        )
        # print("When ready, run self.apply_interactive_calibration(enery_left_peak,energy_right_peak)")

    def apply_interactive_calibration(
        self, energy_left_peak: float, energy_right_peak: float
    ) -> None:
        r"""
        Applies the calibration on two peaks previously selected using calibrate_from_lines.
        It modifies the axes of the object.
        """
        # Dumbass hyperspy keeps events linked and has no "remove events" method
        self.axes_manager.events.any_axis_changed.trigger = (
            hyperspy.events.Event().trigger
        )
        self.axes_manager.events.any_axis_changed._connected_some = {}

        current_e1 = min(self._gauss_means)
        current_e2 = max(self._gauss_means)
        eax = self.axes_manager[-1]
        ch2 = eax.value2index(current_e2)
        old_scale = eax.scale
        old_offset = eax.offset

        new_scale = (
            old_scale
            * (energy_right_peak - energy_left_peak)
            / (current_e2 - current_e1)
        )
        new_offset = (
            energy_right_peak - (current_e2 - old_offset) * new_scale / old_scale
        )
        print(new_scale)
        print(new_offset)
        with self.axes_manager.events.any_axis_changed.suppress():
            self.axes_manager[-1].scale = new_scale
        self.axes_manager[-1].offset = new_offset
        return

    def fit_plot_gauss(self, roi_signal, a, roi, i):

        x = a.axes_manager[-1].axis
        y = np.zeros(a.data.shape)
        eax = a.axes_manager[-1]
        y[eax.value2index(roi.left) : eax.value2index(roi.right)] = roi_signal.data

        mean = (x * y).sum() / y.sum()

        sigma = np.sqrt((y * (x - mean) ** 2).sum() / y.sum())
        popt, _pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])
        self._gauss_means[i - 1] = popt[1]
        fit_mean = popt[1]

        fig = plt.gcf()
        ax = fig.axes[0]
        l = fig.axes[0].lines[i]
        l.set_ydata(Gauss(x, *popt))

        # --- Interactive Text Handling ---
        text_label = f"Peak energy: {fit_mean:.2f} keV"

        # Look for an existing text object belonging to this specific index/plot
        text_obj = None
        for txt in ax.texts:
            if txt.get_gid() == f"gauss_text_{i}":
                text_obj = txt
                break

        if text_obj:
            # Update the existing text and reposition it dynamically if needed
            text_obj.set_text(text_label)
            text_obj.set_position((fit_mean, max(y) * 0.9))
        else:
            # Create a new text object and give it a unique Group ID (gid)
            # transform=ax.transData places it relative to your data coordinates
            ax.text(
                fit_mean,
                max(y) * 0.9,
                text_label,
                color=l.get_color(),
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
                gid=f"gauss_text_{i}",
            )

        fig.canvas.draw()

    #########
    # Utils #
    #########

    def get_full_el_list(self):
        els = self.metadata.EDS_model.elements.copy()
        els_names = [num_to_symbol(el) for el in els]
        return els_names

    ####################
    # Auto Calibration #
    ####################

    def fit_single_peak(self, theoretical_energy, window):
        energy_axis = self.energy_axis

        mask = (energy_axis >= theoretical_energy - window) & (
            energy_axis <= theoretical_energy + window
        )
        if not np.any(mask):
            return None

        xdata = energy_axis[mask]
        ydata = self.average_spectrum[mask]

        x0_guess = xdata[np.argmax(ydata)]
        A_guess = max(np.max(ydata) - np.min(ydata), 1e-6)
        sigma_guess = (
            self.model.width_slope * theoretical_energy + self.model.width_intercept
        ) / 2.3548
        C_guess = np.min(ydata)

        p0 = (A_guess, x0_guess, sigma_guess, C_guess)
        bounds = (
            (0.0, theoretical_energy - window, 0.9 * sigma_guess, 0.0),
            (np.inf, theoretical_energy + window, 1.1 * sigma_guess, np.inf),
        )

        try:
            popt, _ = curve_fit(
                gaussian_with_bg,
                xdata,
                ydata,
                p0,
                # sigma=np.sqrt(np.maximum(ydata, 1.0)),
                # absolute_sigma=True,
                bounds=bounds,
            )
        except Exception:
            return None

        # A, x0, _, _, _ = popt
        # if A <= 1e-9 or abs(x0 - E_theoretical) >= 0.98 * window:
        #     return None

        return popt

    def fit_table(
        self,
        window,
        filter_cs,
        filter_conc,
        # min_snr=0.0,
    ):
        table = self.model.db_dict

        calibrated_table = {}

        concentrations, _ = quant_spectrum(
            self.mean(axis=nav_axes)
            if len(nav_axes := self.axes_manager.navigation_axes) > 0
            else self
        )

        @symbol_to_number_dict
        def symbol_to_number(elements_dict):
            return {str(k): v for k, v in elements_dict.items()}

        concentrations = symbol_to_number(elements_dict=concentrations)

        for elt in self.elements:
            if concentrations[elt] < filter_conc:
                continue

            lines = table[elt]

            max_cs = max([line["cs"] for _, line in lines.items()])

            calibrated_table[elt] = {}

            for line_name, line_data in lines.items():
                e = line_data["energy"]
                cs = line_data["cs"]

                if cs < filter_cs * max_cs:
                    continue

                fit = self.fit_single_peak(e, window)
                if fit is None:
                    continue

                A, x0, sigma, C = fit

                # if C > 0 and (A / C) < min_snr:
                #     continue

                calibrated_table[elt][line_name] = {
                    "energy": x0,
                    "cs": cs,
                    "theoretical": e,
                    "sigma": sigma,
                    "amplitude": A,
                    "bg": C,
                }

        return calibrated_table

    def auto_calibrate(
        self,
        degree=2,
        weighted=True,
        window=0.1,
        filter_cs=0.0,
        filter_conc=0.0,
        # robust=False,
    ):
        calibrated = self.fit_table(window, filter_cs, filter_conc)

        x = []
        y = []
        sigma = []

        for elt in calibrated:
            for _, line in calibrated[elt].items():
                x.append(line["theoretical"])
                y.append(line["energy"])
                sigma.append(line["sigma"])

        # if len(x) == 0:
        #     raise ValueError(
        #         "No valid peaks were found/fitted for polynomial calibration."
        #     )

        x = np.asarray(x)
        y = np.asarray(y)
        sigma = np.asarray(sigma)

        w = (
            None
            if not weighted
            else self.average_spectrum[np.searchsorted(self.energy_axis, x)]
        )

        affine = np.polyfit(y, x, 1, w=w)

        a, b = affine

        axis = self.axes_manager[-1]
        axis.scale *= a
        axis.offset = a * axis.offset + b
        self.energy_axis_ = None
        self.average_spectrum_ = None
        self.model_ = None

        y = np.polyval(affine, y)

        # if robust and len(x) > degree + 1:
        #     x_arr = np.array(x)
        #     y_arr = np.array(y)
        #     inliers = np.ones(len(x), dtype=bool)

        #     for _ in range(3):
        #         if np.sum(inliers) <= degree + 1:
        #             break
        #         w_in = None if w is None else np.array(w)[inliers]
        #         poly_cand = np.polyfit(x_arr[inliers], y_arr[inliers], degree, w=w_in)
        #         res = np.abs(y_arr - np.polyval(poly_cand, x_arr))
        #         std_res = np.std(res[inliers])
        #         if std_res < 1e-6:
        #             break
        #         new_inliers = res < 2.5 * std_res
        #         if np.array_equal(new_inliers, inliers):
        #             break
        #         inliers = new_inliers

        #     x_fit = x_arr[inliers]
        #     y_fit = y_arr[inliers]
        #     w_fit = None if w is None else np.array(w)[inliers]
        #     energy_poly = np.polyfit(x_fit, y_fit, degree, w=w_fit)
        #     sigma_poly = np.polyfit(x_fit, np.array(sigma)[inliers], degree, w=w_fit)
        # else:
        #     energy_poly = np.polyfit(x, y, degree, w=w)
        #     sigma_poly = np.polyfit(x, sigma, degree, w=w)

        energy_poly = np.polyfit(x, y, degree, w=w)
        sigma_poly = np.polyfit(x, sigma, degree, w=w)

        calibrated_db = {
            k: {
                name: {
                    **(
                        calibrated[k][name]
                        if k in calibrated and name in calibrated[k]
                        else {}
                    ),
                    "energy": np.polyval(energy_poly, line["energy"]),
                    "cs": line["cs"],
                    "sigma": np.polyval(sigma_poly, line["energy"]),
                    "theoretical": line["energy"],
                }
                for name, line in v.items()
            }
            for k, v in self.model.db_dict.items()
            if k in self.elements
        }

        self.model.calibrated_db_dict = calibrated_db
        self.model.energy_calibration_poly = energy_poly

        return (
            calibrated_db,
            energy_poly,
            # sigma_poly,
        )


#######################
# Auxiliary functions #
#######################


def get_metadata(spim):
    r"""
    Get the metadata of the :class:`EDSespm` object and format it as a model parameters dictionary.
    """
    mod_pars = {}
    try:
        mod_pars["E0"] = spim.metadata.Acquisition_instrument.TEM.beam_energy
        mod_pars["e_offset"] = spim.axes_manager[-1].offset
        assert mod_pars["e_offset"] > 0.01, (
            "The energy scale can't include 0, it will produce errors elsewhere. Please crop your data."
        )
        mod_pars["e_scale"] = spim.axes_manager[-1].scale
        mod_pars["e_size"] = spim.axes_manager[-1].size
        mod_pars["db_name"] = spim.metadata.xray_db
        mod_pars["width_slope"] = (
            spim.metadata.Acquisition_instrument.TEM.Detector.EDS.width_slope
        )
        mod_pars["width_intercept"] = (
            spim.metadata.Acquisition_instrument.TEM.Detector.EDS.width_intercept
        )

        pars_dict = {}
        pars_dict["Abs"] = {
            "thickness": spim.metadata.Sample.thickness,
            "toa": spim.metadata.Acquisition_instrument.TEM.Detector.EDS.take_off_angle,
            "density": spim.metadata.Sample.density,
        }
        try:
            pars_dict["Det"] = (
                spim.metadata.Acquisition_instrument.TEM.Detector.EDS.type.as_dictionary()
            )
        except AttributeError:
            pars_dict["Det"] = (
                spim.metadata.Acquisition_instrument.TEM.Detector.EDS.type
            )

        mod_pars["params_dict"] = pars_dict

    except AttributeError:
        print(
            "You need to define the relevant parameters for the analysis. Use the set_analysis_parameters function."
        )

    return mod_pars


def build_G(model, g_params):
    model.generate_g_matr(**g_params)
    return model.G


def Gauss(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def gaussian_with_bg(x, A, x0, sigma, C):
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + C  # + m * (x - x0)

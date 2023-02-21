r"""
EDXS model
----------

The :mod:`espm.models.edxs` module implements the creation of the G matrix that contains a modelisation of the characteristic and continuous X-rays.

"""

import numpy as np
from espm.models import PhysicalModel
from espm.models.EDXS_function import G_bremsstrahlung, continuum_xrays, gaussian, read_lines_db, read_compact_db, update_bremsstrahlung, elts_dict_from_dict_list
from espm.conf import DEFAULT_EDXS_PARAMS
from espm.utils import arg_helper, symbol_to_number_dict, symbol_to_number_list
from espm.models.absorption_edxs import absorption_correction, det_efficiency, det_efficiency_from_curve, absorption_mass_thickness
# Class to model the EDXS spectra. This is a temporary version since there are some design issues.


class EDXS(PhysicalModel):
    def __init__(
        self, 
        *args, 
        width_slope=0.01,
        width_intercept=0.065,
        **kwargs
    ):
        """
        :database_path: file path to a database of x-ray data (energy and intensity ratios of the peaks)
        :abs_db_path: file path to a database of attenuation coefficients (Useful for artificial data only for the moment)
        :brstlg_pars: dictionary of parameters for the continuum X-rays. Only a part of the parameters can be used without issues.
        :e_offset: energy offset of the energy axis (float)
        :e_size: number of energy channels (int)
        :e_scale: ev/channel calibration of the energy axis (float)
        :width_slope: The FWHM of the detector increases with energy which is modeled with an affine function. This is the slope of this affine function (float).
        :width_intercept: The FWHM of the detector increases with energy which is modeled with an affine function. This is the intercept of this affine function (float).
        """
        super().__init__(*args,**kwargs)
        self.width_slope = width_slope
        self.width_intercept = width_intercept

        default_params = DEFAULT_EDXS_PARAMS
        self.params_dict = arg_helper(self.params_dict,default_params)
        
        self.lines = self.db_mdata["lines"]
        self.norm = 1.0


    @symbol_to_number_list
    def generate_g_matr(self, g_type="bremsstrahlung", norm = True, reference_elt = {},*,elements=[], **kwargs):
        r"""
        Generate the G matrix. With a complete model the matrix is (e_size,n+2). The first n columns correspond to the sum of X-ray characteristic peaks associated to each shell of the elements. The last 2 columns correspond to a bremsstrahlung model. 
        
        Parameters
        ----------
        g_type : 
            :string: Options of the edxs model to include in the G matrix. The three possibilities are "identity", "no_brstlg" and "bremsstrahlung". G is going to be the identity matrix, only characteristic X-ray peaks or characteristic X-rays plus bremsstrahlung model, respectively.
        norm : 
            :boolean: Norms the columns of G for computation purposes. That feature will be removed, for a better solution.
        reference_elt : 
            :dict: The keys are chemical elements (atomic number) and the values are cut-off energies. This argument is used to split some of the columns of G into 2 columns. The first column corresponds to characteristic X-rays before the cut-off and second one corresponds to characteristic X-rays before the cut-off. This feature is implemented to enable more accurate absorption correction.
        elements : 
            :list: List of modeled chemical elements. The list can be populated either with atomic numbers or chemical symbols, e.g. "Fe" or 26.

        Returns
        -------
        g matrix :
            :np.array 2D: matrix of the edx model. 

        Notes
        -----
        See our paper about the espm package for more information about the equations of this model.
        """
        # Diagonal g_matr
        if g_type == "bremsstrahlung" : 
            self.bkgd_in_G = True

        if len(elements) == 0:
            return None

        elif g_type == "identity" : 
            return None
        # model based on elements_list
        elif (g_type == "bremsstrahlung") or (g_type == "no_brstlg"):
            
            # The number of shells depend on the element, it is then not straightforward to pre-determine the size of g_matr
            self.G = np.zeros((self.x.shape[0], 0))
            # For each element we unpack all shells and then unpack all lines of each shell.
            for elt in elements:
                if self.lines : 
                    energies, cs = read_lines_db(elt,self.db_dict)
                else : 
                    energies, cs = read_compact_db(elt,self.db_dict)
                if str(elt) in reference_elt : 
                    peaks_low = np.zeros((self.x.shape[0], 1))
                    peaks_high = np.zeros((self.x.shape[0], 1))
                    for i, energy in enumerate(energies):
                        if (energy > np.min(self.x)) and (energy < np.max(self.x)):
                                
                            if type(self.params_dict["Det"]) == str : 
                                D = det_efficiency_from_curve(energy,self.params_dict["Det"])
                            else : 
                                D = det_efficiency(energy,self.params_dict["Det"])

                            A = absorption_correction(energy,**self.params_dict["Abs"],elements_dict = {elt : 1.0})
                            
                            width = self.width_slope * energy + self.width_intercept
                            if energy < reference_elt[str(elt)] : 
                                peaks_low += (
                                    cs[i]
                                    * gaussian(self.x, energy, width / 2.3548)[
                                        np.newaxis
                                    ].T
                                )*D*A
                            else : 
                                peaks_high += (
                                    cs[i]
                                    * gaussian(self.x, energy, width / 2.3548)[
                                        np.newaxis
                                    ].T
                                )*D*A
                    peaks = np.hstack((peaks_low,peaks_high))
                else : 
                    peaks = np.zeros((self.x.shape[0], 1))
                    for i, energy in enumerate(energies):
                        
                        # The actual detected width is calculated at each energy
                        if (energy > np.min(self.x)) and (energy < np.max(self.x)):
                            
                            if type(self.params_dict["Det"]) == str : 
                                D = det_efficiency_from_curve(energy,self.params_dict["Det"])
                            else : 
                                D = det_efficiency(energy,self.params_dict["Det"])

                            A = absorption_correction(energy,**self.params_dict["Abs"],elements_dict = {elt : 1.0})
                            
                            width = self.width_slope * energy + self.width_intercept
                            

                            peaks += (
                                cs[i]
                                * gaussian(self.x, energy, width / 2.3548)[
                                    np.newaxis
                                ].T
                            )*D*A
                
                if np.max(peaks) > 0.0:
                    self.G = np.concatenate((self.G, peaks), axis=1)
                else : 
                    print("No peak is present in the energy range for element : {}".format(elt))
            
            # Appends a pure continuum spectrum is needed
            if self.bkgd_in_G:
                approx_elts = {key : 1.0/len(elements) for key in elements}
                brstlg_spectrum = G_bremsstrahlung(self.x,self.E0,self.params_dict,elements_dict=approx_elts)
                if np.max(brstlg_spectrum) > 0.0 : 
                    self.G = np.concatenate((self.G, brstlg_spectrum), axis=1)
                else : 
                    print("Bremsstrahlung parameters were not provided, bkgd not added in G")
                    self.bkgd_in_G = False

            if norm : 
                norms = np.sqrt(np.sum(self.G**2, axis=0, keepdims=True))
                if g_type == "bremsstrahlung" : 
                    norms[0][:-2] = np.mean(norms[0][:-2])
                else : 
                    norms[0] = np.mean(norms[0])
                self.norm = norms
                self.G /= self.norm
        else : 
            print("g_type has to be one of those : \"bremsstrahlung\", \"no_brstlg\" or \"identity\". G will be None, corresponding to \"identity\". ")

            

   

    def generate_phases(self, phases_parameters) : 
        r"""
        Generate a series of spectra from list of phase parameters

        Parameters
        ----------
        phases_parameters : 
            :list: List of dicts containing the phase parameters.
        
        Returns
        -------
        phase_array : 
            :np.array 2D: Array which lines correspond to the modelled phases in the input list.

        Notes
        -----
        The absorption correction is done using the average composition of the phases. The same correction is used for each phase.
        """
        self.phases = []
        unique_elts = dict(elts_dict_from_dict_list([x["elements_dict"] for x in phases_parameters]))
        for p in phases_parameters:
            # p.update({"elements_dict" : unique_elts})
            self.phases.append(self.generate_spectrum(**p,abs_elts_dict = unique_elts))
        self.phases = np.array(self.phases)
        self.phases /= self.phases.sum(axis = 1)[:,np.newaxis]

    @symbol_to_number_dict
    def generate_spectrum(self, b0=0, b1 = 0, scale = 1.0,abs_elts_dict = {},*,elements_dict = {}):
        r"""
        Generate a spectrum from bremsstrahlung parameters and a chemical composition.

        Parameters
        ----------
        b0 : 
            :float: First bremsstrahlung parameter. 
        b1 : 
            :float: Second bremsstrahlung parameter.
        scale : 
            :float: Scale factor to apply to the bremsstrahlung. Feature to be removed, scaling should be done with b0 and b1.
        abs_elts_dict : 
            :dict: Dictionnary of elements and associated concentrations. This dictionnary is used to calculate the contribution of the absorption in the spectrum.
        elements_dict : 
            :dict: Dictionnary of elements and associated concentrations describing the chemical composition of the modeled sample.
        Returns
        -------
        spectrum : 
            :np.array 1D: Output edx spectrum corresponding to the input chemical composition.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from espm.models.edxs import EDXS
        >>> from espm.conf import DEFAULT_SYNTHETIC_DATA_DICT
        >>> b0, b1 = 5.5367e-5, 0.00192181
        >>> elts_dict = {"Si" : 1.0,"Ca" : 1.0,"O" : 3.0,"C" : 0.3}
        >>> model = EDXS(**DEFAULT_SYNTHETIC_DATA_DICT['model_parameters'])
        >>> spectrum = model.generate_spectrum(b0,b1, elements_dict = elts_dict)
        >>> plt.plot(spectrum)

        Notes
        -----
        Check EDXS_function for details about the bremsstrahlung model.
        """
        temp = np.zeros_like(self.x)
        for elt in elements_dict.keys():
            if self.lines : 
                energies, cs = read_lines_db(elt,self.db_dict)
            else : 
                energies, cs = read_compact_db(elt,self.db_dict)
            for i, energy in enumerate(energies):
                if (energy > np.min(self.x)) and (energy < np.max(self.x)):
                    if type(self.params_dict["Det"]) == str : 
                        D = det_efficiency_from_curve(energy,self.params_dict["Det"])
                    else : 
                        D = det_efficiency(energy,self.params_dict["Det"])
                    A = absorption_correction(energy,**self.params_dict["Abs"],elements_dict = {elt : 1.0})
                
                    width = self.width_slope * energy + self.width_intercept
                    temp += (
                        elements_dict[elt]
                        * cs[i]
                        * gaussian(self.x, energy, width / 2.3548)
                    )*A*D
        temp /= temp.sum()
        if abs_elts_dict == {} : 
            temp += continuum_xrays(self.x,self.params_dict,b0,b1,self.E0,elements_dict=elements_dict) * scale
        else : 
            temp += continuum_xrays(self.x,self.params_dict,b0,b1,self.E0,elements_dict=abs_elts_dict) * scale
        
        return temp

    # @symbol_to_number_dict
    # def generate_abs_correction(self,mass_thickness : np.ndarray,*,elements_dict = {}) : 
    #     temp = np.zeros((self.x.shape[0],mass_thickness.shape[0]*mass_thickness.shape[1]))
    #     for elt in elements_dict.keys():
    #         if self.lines : 
    #             energies, cs = read_lines_db(elt,self.db_dict)
    #         else : 
    #             energies, cs = read_compact_db(elt,self.db_dict)
    #         for i, energy in enumerate(energies):
    #             if (energy > np.min(self.x)) and (energy < np.max(self.x)):
    #                 A = absorption_mass_thickness(energy, mass_thickness=mass_thickness,**self.params_dict["Abs"],elements_dict = elements_dict)
                
    #                 width = self.width_slope * energy + self.width_intercept
    #                 temp -= (
    #                     elements_dict[elt]
    #                     * cs[i]
    #                     * gaussian(self.x, energy, width / 2.3548)[:,np.newaxis]
    #                 )*((A.reshape(-1)[:,np.newaxis]).T)
    #     temp += absorption_mass_thickness(self.x, mass_thickness=mass_thickness,**self.params_dict["Abs"],elements_dict = elements_dict)
        

def G_EDXS (model_params, g_params, part_W = None, G = None) : 
    r"""
    Update the bremsstrahlung part of the G matrix. This function is used for the NMF decomposition so that the absorption correction is updated in between each step.

    Parameters
    ----------
    model_params : 
        :dict: Parameters of the edxs model. Check espm.conf.DEFAULT_EDXS_PARAMS for an example. 
    g_params : 
        :dict: Parameters specific to the creation of the G matrix.
    part_W : 
        :np.array 2D: Concentrations determined during the NMF decomposition. It is used to determine the average composition that will be used to determine the absorption correction.
    G : 
        :np.array 2D: Current G matrix. Setting it to None will initialize the G matrix.

    Returns
    -------
    updated G : 
        :np.array 2D: Updated G matrix with a new absorption correction.
    """
    if G is None : 
        model = EDXS(**model_params)
        model.generate_g_matr(**g_params)
        G = model.G

    if part_W is None : 
        return G
    else : 
        new_G = update_bremsstrahlung(G,part_W,model_params,g_params["elements"])
        return new_G

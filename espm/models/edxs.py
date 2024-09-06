r"""
EDXS model
----------

The :mod:`espm.models.edxs` module implements the creation of the G matrix that contains a modelisation of the characteristic and continuous X-rays.

**This module is a key component of the EDXS data simulation and the EDXS data analysis.**

"""

import numpy as np
import re
from espm.models import PhysicalModel
from espm.models.EDXS_function import G_bremsstrahlung, continuum_xrays, gaussian, read_lines_db, read_compact_db, elts_dict_from_dict_list
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
        custom_init = False,
        **kwargs
    ):
        r"""
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
        
        self.lines = self.db_mdata["lines"]
        self.norm = 1.0
        self.model_elts = []
        self.custom_init = custom_init

    def __add_elts_G(self, reference_elt = {}, *, elements=[]):
        for elt in elements:
            if self.lines : 
                energies, cs = read_lines_db(elt,self.db_dict)
            else : 
                energies, cs = read_compact_db(elt,self.db_dict)
            if elt in reference_elt : 
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
                        if energy < reference_elt[elt] : 
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
        
            print(np.max(peaks, axis = 0))
            print(str(elt))
            if np.all((np.max(peaks, axis = 0)) > 0.0):
                self.G = np.concatenate((self.G, peaks), axis=1)
                if elt in reference_elt : 
                    self.model_elts.append(str(elt)+'_lo')
                    self.model_elts.append(str(elt)+'_hi')
                else : 
                    self.model_elts.append(str(elt))
            else : 
                print("The energy split of the element : {} leads to empty G columns. Please remove split or change its energy.".format(elt))
                raise ValueError("Empty G column")           

    @symbol_to_number_list
    @symbol_to_number_dict
    def generate_g_matr(self, g_type="bremsstrahlung",*,elements=[],elements_dict = {},**kwargs):
        r"""
        Generate the G matrix. With a complete model the matrix is (e_size,n+2). The first n columns correspond to the sum of X-ray characteristic peaks associated to each shell of the elements. The last 2 columns correspond to a bremsstrahlung model. 
        
        The first n columns of G (noted :math:`\kappa`), corresponding to the characteristic X-rays, can be calculated using the following formula:
        
        .. math::

            \kappa_{k,Z} = \sum_{ij} x_{ij}(Z) \frac{1}{\Delta(\varepsilon_k) \sqrt{2\pi} } e{^{-\frac{1}{2} {\left( \frac{\varepsilon_k - \varepsilon^{ij}(Z)}{\Delta(\varepsilon_k)} \right)}^2}}
        
        where :math:`\varepsilon_k` is the energy of the kth energy channel, :math:`\varepsilon^{ij}(Z)` is the energy of the ijth line of the Zth element, :math:`\Delta(\varepsilon_k)` is the FWHM of the detector at the energy :math:`\varepsilon_k` and :math:`x_{ij}(Z)` is the intensity ratio of the ijth line of the Zth element. The last term of the equation is a gaussian function centered at :math:`\varepsilon^{ij}(Z)` and with a FWHM of :math:`\Delta(\varepsilon_k)`.

        The last two columns of G (noted :math:`\beta`), corresponding to the bremsstrahlung model, can be calculated using the following formula:

        .. math::

            \beta_{k,n+1} = \frac{\varepsilon_0 - \varepsilon_k}{\varepsilon_0 \varepsilon_k} \times \frac{1 - (\varepsilon_0 - \varepsilon_k)}{\varepsilon_0}

            \beta_{k,n+2} = \frac{(\varepsilon_0 - \varepsilon_k)^2}{\varepsilon^2_0 \varepsilon_k}

        Note that there is no parameter for the bremsstrahlung since it is supposed to be learned with the W matrix (see espm.estimators).
        
        Parameters
        ----------
        g_type : 
            :string: Options of the edxs model to include in the G matrix. The three possibilities are "identity", "no_brstlg" and "bremsstrahlung". G is going to be the identity matrix, only characteristic X-ray peaks or characteristic X-rays plus bremsstrahlung model, respectively.
        elements_dict : 
            :dict: The keys are chemical elements (atomic number) and the values are cut-off energies. This argument is used to split some of the columns of G into 2 columns. The first column corresponds to characteristic X-rays before the cut-off and second one corresponds to characteristic X-rays before the cut-off. This feature is implemented to enable more accurate absorption correction.
        elements : 
            :list: List of modeled chemical elements. The list can be populated either with atomic numbers or chemical symbols, e.g. "Fe" or 26.

        Returns
        -------
        g matrix :
            :np.array 2D: matrix of the edx model. 

        Notes
        -----
        See our paper about the espm package :cite:p:`teurtrie2023espm` for more information about the equations of this model.
        """
        
        # Reset the internally stored elements list
        self.model_elts = []

        valid_elts = self.__check_elts_in_G(elements)
        print('Input elements')
        print(elements)
        print('Valid elements')
        print(valid_elts)
        
        if g_type == "bremsstrahlung" : 
            self.bkgd_in_G = True
        else : 
            self.bkgd_in_G = False

        # None is a default value for the G matrix and thus G will be considered to be the identity matrix in most of espm functions.
        if len(valid_elts) == 0:
            self.G = None

        elif g_type == "identity" : 
            self.G = None
        # model based on elements_list
        elif (g_type == "bremsstrahlung") or (g_type == "no_brstlg"):
            
            # The number of shells depend on the element, it is then not straightforward to pre-determine the size of g_matr
            self.G = np.zeros((self.x.shape[0], 0))
            # For each element we unpack all shells and then unpack all lines of each shell.
            self.__add_elts_G(reference_elt = elements_dict, elements = valid_elts)
            
            # Appends a pure continuum spectrum is needed
            if self.bkgd_in_G:
                approx_elts = {key : 1.0/len(valid_elts) for key in valid_elts}
                brstlg_spectrum = G_bremsstrahlung(self.x,self.E0,self.params_dict,elements_dict=approx_elts)
                if np.max(brstlg_spectrum) > 0.0 : 
                    self.G = np.concatenate((self.G, brstlg_spectrum), axis=1)
                else : 
                    print("Bremsstrahlung parameters were not provided, bkgd not added in G")
                    self.bkgd_in_G = False

            norms = np.sqrt(np.sum(self.G**2, axis=0, keepdims=True))
            if g_type == "bremsstrahlung" : 
                norms[0][:-2] = np.mean(norms[0][:-2])
            else : 
                norms[0] = np.mean(norms[0])
            self.norm = norms
            self.G /= self.norm
        else : 
            print("g_type has to be one of those : \"bremsstrahlung\", \"no_brstlg\" or \"identity\". G will be None, corresponding to \"identity\". ")
    
    def __check_elts_in_G(self, elements):
        """
        Check if the elements of the metadata are in the range of the energy axis.
        """
        valid_elts = []
        for elt in elements:
            if self.lines : 
                energies, cs = read_lines_db(elt,self.db_dict)
            else : 
                energies, cs = read_compact_db(elt,self.db_dict)

            energy_range = [np.min(self.x), np.max(self.x)]
            found = 0
            for energy in energies:
                if (energy > energy_range[0]) and (energy < energy_range[1]):
                    valid_elts.append(elt)
                    found = 1
                    break
            if not found:
                print("No peak is present in the energy range for element : {}".format(elt))
        return valid_elts
    
    def generate_phases(self, phases_parameters) : 
        r"""
        Generate a series of spectra from list of phase parameters. 

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
        Generate a spectrum from bremsstrahlung parameters and a chemical composition. The modelling is done using the same formula as for the generate_g_matr function.

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

        .. plot::
            :context: close-figs

            >>> import matplotlib.pyplot as plt
            >>> from espm.models.edxs import EDXS
            >>> from espm.conf import DEFAULT_EDXS_PARAMS
            >>> b0, b1 = 5.5367e-5, 0.00192181
            >>> elts_dict = {"Si" : 1.0,"Ca" : 1.0,"O" : 3.0,"C" : 0.3}
            >>> model = EDXS(**DEFAULT_EDXS_PARAMS)
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
    
    def get_elements(self) :
        for elt in self.model_elts:
            if re.match(r'[0-9]*(_lo)',elt) : 
                pass
            elif re.match(r'[0-9]*(_hi)',elt) :
                m = re.match(r'([0-9]*)(_hi)',elt) 
                yield m.group(1)
            else : 
                yield elt
        
    def carac_X_span(self) : 
        all_indices = []
        for elt in self.get_elements():
            if self.lines : 
                energies, cs = read_lines_db(elt,self.db_dict)
            else : 
                energies, cs = read_compact_db(elt,self.db_dict)
            for energy in energies:
                width = self.width_slope * energy + self.width_intercept
                span = [energy - 2*width, energy + 2*width]
                indices = np.where((self.x > span[0]) & (self.x < span[1]))[0]
                all_indices.append(indices)
        return np.unique(np.concatenate(all_indices))
    
    def NMF_initialize_W(self, D) :
        if self.G is None :
            raise ValueError('The G matrix is identity, the W matrix cannot be initialized. Please use a np.array for G in the ESpM-NMF instead of the model object')
        if self.bkgd_in_G and self.custom_init:
            idx = self.carac_X_span()
            mask = np.ones(self.G.shape[0], bool)
            mask[idx] = 0
            Wbrem = (np.linalg.lstsq(self.G[mask,-2:],D[mask,:],rcond = None)[0]).clip(min = 0)
            Wcarac = (np.linalg.lstsq(self.G[idx,:-2],D[idx,:] ,rcond = None)[0]).clip(min = 0)
            # filter = np.where(np.mean(self.G[:,:-2],axis=1)<(np.max(np.mean(self.G[:,:-2],axis=1))*0.001))[0]
            W = np.vstack((Wcarac,Wbrem))
        else :
            W = (np.linalg.lstsq(self.G,D,rcond = None)[0]).clip(min = 0)

        return W
        
    def NMF_simplex(self):
        """
        Produce the indices of the rows of W on which to perform the simplex constraint.
        """
        # We don't check here whether G was correctyl initialized since it is tested in the init.
        ind_list = []
        # We skip the low energy lines
        for i, elt in enumerate(self.model_elts):
            if re.match(r'[0-9]*(_lo)',elt) : 
                pass
            else : 
                ind_list.append(i)
        # We skip the bremsstrahlung
        return ind_list
    
    def NMF_update(self, W=None):
        """
        Update the G matrix with the new absorption correction.
        """
        # We don't need to check whether G was correctyl initialized it should work anyway.
        
        if W is None:
            return self.G
        if not(self.bkgd_in_G) :
            return self.G
        else :
            new_brstlg = self.update_bremsstrahlung(W)
            new_G = self.G.copy()
            new_G[:,-2:] = new_brstlg/self.norm[0][-2:]
            self.G = new_G
            return self.G

    def update_bremsstrahlung(self, W) : 
        """
        Update the bremsstrahlung part of the G matrix. This function is used for the NMF decomposition so that the absorption correction is updated in between each step.
        """
        if not(self.bkgd_in_G) :
            raise AttributeError("The bremsstrahlung is not comprised in the model.")
        
        indices = self.NMF_simplex()
        mean_compo = np.mean(W[indices,:],axis=1)
        normed_compo = mean_compo/np.sum(mean_compo)
        elements_dict = {key : normed_compo[i] for i,key in enumerate(self.get_elements())}
        bremsstrahlung = G_bremsstrahlung(self.x,self.E0,self.params_dict,elements_dict=elements_dict)
        return bremsstrahlung

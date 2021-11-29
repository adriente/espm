import numpy as np
from snmfem.models import PhysicalModel
from snmfem.models.EDXS_function import G_bremsstrahlung, continuum_xrays, gaussian, read_lines_db, read_compact_db, update_bremsstrahlung, elts_dict_from_dict_list
from snmfem.conf import DEFAULT_EDXS_PARAMS
from snmfem.utils import arg_helper, symbol_to_number_dict, symbol_to_number_list
from snmfem.models.absorption_edxs import absorption_correction, det_efficiency, det_efficiency_from_curve
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

        # e_offset=0.20805000000000007,
        # e_size=1980,
        # e_scale=0.01,
    @symbol_to_number_list
    def generate_g_matr(self, g_type="bremsstrahlung", norm = True,*,elements=[], **kwargs):
        """
        Generates a matrix (e_size,n). Each column corresponds to the sum of X-ray characteristic gaussian peaks associated to each shell of the elements of elements_lists. n is then len(elements_list)*number of shells per element.
        :elements_list: List of integers. Each integer is an element of the model. If None, the g_matr is diagonal matrix of size e_size.
        :brstlg: Boolean. If true a continuum X-ray spectrum is added to g_matr.
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
                self.G /= np.sqrt(np.sum(self.G**2, axis=0, keepdims=True))
        else : 
            print("g_type has to be one of those : \"bremsstrahlung\", \"no_brstlg\" or \"identity\". G will be None, corresponding to \"identity\". ")

            

   

    def generate_phases(self, phases_parameters) : 
        self.phases = []
        unique_elts = dict(elts_dict_from_dict_list([x["elements_dict"] for x in phases_parameters]))
        for p in phases_parameters:
            # p.update({"elements_dict" : unique_elts})
            self.phases.append(self.generate_spectrum(**p,abs_elts_dict = unique_elts))
        self.phases = np.array(self.phases)
        self.phases /= self.phases.sum(axis = 1)[:,np.newaxis]

    @symbol_to_number_dict
    def generate_spectrum(self, b0=0, b1 = 0, scale = 1.0,abs_elts_dict = {},*,elements_dict = {}):
        """
        Generates an EDXS spectrum with the specified elements. The continuum x-rays are added and scaled to the gaussian peaks.
        :elements_dict: dictionnary of elements with concentration in that way {"integer":concentration}
        :scale: Scale of the continuum x-rays to the characteristic X-rays. It can be set to 0 to eliminate the continuum. (float)
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

def G_EDXS (model_params, g_params, part_P = None, G = None) : 
    if G is None : 
        model = EDXS(**model_params)
        model.generate_g_matr(**g_params)
        G = model.G

    if part_P is None : 
        return G
    else : 
        new_G = update_bremsstrahlung(G,part_P,model_params,g_params["elements"])
        return new_G

 # def generate_abs_coeff(self, elements_dict = None):
    #     """
    #     Function to update self.abs based on the given database. This function is working as intended but it can't be used as it is for SNMF.
    #     The way it was coded is probably over-complicated ...
    #     """
    #     temp = np.zeros_like(self.x)
    #     sum_mass = 0
    #     for elt in elements_dict.keys():
    #         sum_mass += 2 * int(elt)*elements_dict[elt]

    #     for elt in elements_dict.keys():
    #         H = np.ones_like(self.x)
    #         d = np.ones_like(self.x)
    #         k = np.ones_like(self.x)
    #         H_tups = []
    #         dk_tups = []
    #         list_L_shells = []
    #         list_M_shells = []
    #         max_ind_L = 0
    #         max_ind_M = 0
    #         for shell in self.abs_db[elt]:
    #             regex = re.match(r"([K-M])([1-5]?)", shell)
    #             if regex:
    #                 if regex.group(1) == "L":
    #                     list_L_shells.append(int(regex.group(2)))
    #                 if regex.group(1) == "M":
    #                     list_M_shells.append(int(regex.group(2)))
    #             if list_L_shells:
    #                 max_ind_L = max(list_L_shells)
    #             if list_M_shells:
    #                 max_ind_M = max(list_M_shells)
    #         for shell in self.abs_db[elt]:
    #             regex = re.match(r"([K-M])([1-5]?)", shell)
    #             if regex:
    #                 tup = self.abs_db[elt][shell]
    #                 ind_H = np.argmin(np.abs(self.x - tup[0]))
    #                 H_tups.append((ind_H, tup[1]))

    #             if regex:
    #                 if regex.group(1) == "K":
    #                     en = self.abs_db[elt][shell][0]
    #                     ind_dk = np.argmin(np.abs(self.x - en))
    #                     dk_tups.append(
    #                         (
    #                             ind_dk,
    #                             self.abs_db[elt]["d"][0],
    #                             self.abs_db[elt]["exp"][0],
    #                         )
    #                     )

    #                 if regex.group(0) == "L" + str(max_ind_L):
    #                     en = self.abs_db[elt][shell][0]
    #                     ind_dk = np.argmin(np.abs(self.x - en))
    #                     dk_tups.append(
    #                         (
    #                             ind_dk,
    #                             self.abs_db[elt]["d"][1],
    #                             self.abs_db[elt]["exp"][1],
    #                         )
    #                     )

    #                 if regex.group(0) == "M" + str(max_ind_M):
    #                     en = self.abs_db[elt][shell][0]
    #                     ind_dk = np.argmin(np.abs(self.x - en))
    #                     dk_tups.append(
    #                         (
    #                             ind_dk,
    #                             self.abs_db[elt]["d"][2],
    #                             self.abs_db[elt]["exp"][2],
    #                         )
    #                     )

    #         sort_key = lambda x: x[0]
    #         dk_tups.sort(key=sort_key)
    #         H_tups.sort(key=sort_key)
    #         d[0 : dk_tups[0][0]] = self.abs_db[elt]["d"][-1]
    #         k[0 : dk_tups[0][0]] = self.abs_db[elt]["exp"][-1]
    #         for i in range(len(dk_tups) - 1):
    #             d[dk_tups[i][0] : dk_tups[i + 1][0]] = dk_tups[i][1]
    #             k[dk_tups[i][0] : dk_tups[i + 1][0]] = dk_tups[i][2]
    #         d[dk_tups[-1][0] :] = self.abs_db[elt]["d"][0]
    #         k[dk_tups[-1][0] :] = self.abs_db[elt]["exp"][0]
    #         for i in range(len(H_tups) - 1):
    #             H[H_tups[i][0] : H_tups[i + 1][0]] = H_tups[i][1]

    #         temp += (H
    #             * np.exp(d + k * np.log(self.x))*elements_dict[elt]*(2* int(elt)/sum_mass)
    #         )

    #     self.abs = temp
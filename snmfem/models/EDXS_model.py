import numpy as np
import json
import os
import snmfem.utils as u
import re
import snmfem.conf as conf
from pathlib import Path
from snmfem.models.base_model import EMModel
from snmfem.models.EDXS_function import continuum_xrays, gaussian, simple_abs_coeff

# Class to model the EDXS spectra. This is a temporary version since there are some design issues.


class EDXS_Model(EMModel):
    def __init__(
        self, 
        *args,
        abs_db_path=None, 
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
        # In the absence of abs coeff database a basic attenuation coefficient is built
        # If a database is given, the absorption coeff needs to be built using the dedicated function
        if abs_db_path is None:
            self.abs = simple_abs_coeff(self.x)
        else:
            with open(abs_db_path, "r") as abs_file:
                self.abs_db = json.load(abs_file)["table"]
            self.abs = np.zeros_like(self.x)

        # e_offset=0.20805000000000007,
        # e_size=1980,
        # e_scale=0.01,

    def generate_g_matr(self, elements_list=None, brstlg=False):
        """
        Generates a matrix (e_size,n). Each column corresponds to the sum of X-ray characteristic gaussian peaks associated to each shell of the elements of elements_lists. n is then len(elements_list)*number of shells per element.
        :elements_list: List of integers. Each integer is an element of the model. If None, the g_matr is diagonal matrix of size e_size.
        :brstlg: Boolean. If true a continuum X-ray spectrum is added to g_matr.
        """
        # Diagonal g_matr
        if elements_list is None:
            pass
        # model based on elements_list
        else:
            self.bkgd_in_G = brstlg
            # The number of shells depend on the element, it is then not straightforward to pre-determine the size of g_matr
            self.G = np.zeros((self.x.shape[0], 0))
            # For each element we unpack all shells and then unpack all lines of each shell.
            for elt in elements_list:
                for shell in self.db_dict[str(elt)]:
                    peaks = np.zeros((self.x.shape[0], 1))
                    for i, energy in enumerate(self.db_dict[str(elt)][shell]["energies"]):
                        # The actual detected width is calculated at each energy
                        if energy > np.min(self.x):
                            width = self.width_slope * energy + self.width_intercept
                            peaks += (
                                self.db_dict[str(elt)][shell]["ratios"][i]
                                * gaussian(self.x, energy, width / 2.3548)[
                                    np.newaxis
                                ].T
                            )
                    if np.max(peaks) > 0.0:
                        self.G = np.concatenate((self.G, peaks), axis=1)
            # Appends a pure continuum spectrum is needed
            if self.bkgd_in_G:
                brstlg_spectrum = continuum_xrays(self.x,self.params_dict,self.abs)[np.newaxis].T
                self.G = np.concatenate((self.G, brstlg_spectrum), axis=1)

    def generate_spectrum(self, elements_dict, scale):
        """
        Generates an EDXS spectrum with the specified elements. The continuum x-rays are added and scaled to the gaussian peaks.
        :elements_dict: dictionnary of elements with concentration in that way {"integer":concentration}
        :scale: Scale of the continuum x-rays to the characteristic X-rays. It can be set to 0 to eliminate the continuum. (float)
        """
        temp = np.zeros_like(self.x)
        for elt in elements_dict.keys():
            for shell in self.db_dict[elt]:
                for i, energy in enumerate(self.db_dict[elt][shell]["energies"]):
                    width = self.width_slope * energy + self.width_intercept
                    temp += (
                        elements_dict[elt]
                        * self.db_dict[elt][shell]["ratios"][i]
                        * gaussian(self.x, energy, width / 2.3548)
                    )
        temp += continuum_xrays(self.x,self.params_dict,self.abs) * scale
        self.spectrum = temp

    def generate_abs_coeff(self, elements_dict):
        """
        Function to update self.abs based on the given database. This function is working as intended but it can't be used as it is for SNMF.
        The way it was coded is probably over-complicated ...
        """
        temp = np.zeros_like(self.x)
        sum_Z = 0
        sum_conc = 0
        for elt in elements_dict.keys():
            sum_Z += 2 * int(elt)
            sum_conc += elements_dict[elt]

        for elt in elements_dict.keys():
            H = np.ones_like(self.x)
            d = np.ones_like(self.x)
            k = np.ones_like(self.x)
            H_tups = []
            dk_tups = []
            list_L_shells = []
            list_M_shells = []
            max_ind_L = 0
            max_ind_M = 0
            for shell in self.abs_db[elt]:
                regex = re.match(r"([K-M])([1-5]?)", shell)
                if regex:
                    if regex.group(1) == "L":
                        list_L_shells.append(int(regex.group(2)))
                    if regex.group(1) == "M":
                        list_M_shells.append(int(regex.group(2)))
                if list_L_shells:
                    max_ind_L = max(list_L_shells)
                if list_M_shells:
                    max_ind_M = max(list_M_shells)
            for shell in self.abs_db[elt]:
                regex = re.match(r"([K-M])([1-5]?)", shell)
                if regex:
                    tup = self.abs_db[elt][shell]
                    ind_H = np.argmin(np.abs(self.x - tup[0]))
                    H_tups.append((ind_H, tup[1]))

                if regex:
                    if regex.group(1) == "K":
                        en = self.abs_db[elt][shell][0]
                        ind_dk = np.argmin(np.abs(self.x - en))
                        dk_tups.append(
                            (
                                ind_dk,
                                self.abs_db[elt]["d"][0],
                                self.abs_db[elt]["exp"][0],
                            )
                        )

                    if regex.group(0) == "L" + str(max_ind_L):
                        en = self.abs_db[elt][shell][0]
                        ind_dk = np.argmin(np.abs(self.x - en))
                        dk_tups.append(
                            (
                                ind_dk,
                                self.abs_db[elt]["d"][1],
                                self.abs_db[elt]["exp"][1],
                            )
                        )

                    if regex.group(0) == "M" + str(max_ind_M):
                        en = self.abs_db[elt][shell][0]
                        ind_dk = np.argmin(np.abs(self.x - en))
                        dk_tups.append(
                            (
                                ind_dk,
                                self.abs_db[elt]["d"][2],
                                self.abs_db[elt]["exp"][2],
                            )
                        )

            sort_key = lambda x: x[0]
            dk_tups.sort(key=sort_key)
            H_tups.sort(key=sort_key)
            d[0 : dk_tups[0][0]] = self.abs_db[elt]["d"][-1]
            k[0 : dk_tups[0][0]] = self.abs_db[elt]["exp"][-1]
            for i in range(len(dk_tups) - 1):
                d[dk_tups[i][0] : dk_tups[i + 1][0]] = dk_tups[i][1]
                k[dk_tups[i][0] : dk_tups[i + 1][0]] = dk_tups[i][2]
            d[dk_tups[-1][0] :] = self.abs_db[elt]["d"][0]
            k[dk_tups[-1][0] :] = self.abs_db[elt]["exp"][0]
            for i in range(len(H_tups) - 1):
                H[H_tups[i][0] : H_tups[i + 1][0]] = H_tups[i][1]

            temp += (
                elements_dict[elt]
                * 2
                * int(elt)
                * H
                * np.exp(d + k * np.log(self.x))
                / (sum_conc * sum_Z)
            )

        self.abs = temp

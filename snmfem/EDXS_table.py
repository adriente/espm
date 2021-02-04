import xraylib as xr
import numpy as np
import re
import os
import pandas as pd
import json

# Class to generate the tables required for the calculation of EDXS spectra. For now it is able to generate Characteristic X-ray ratios and attenuation coefficient tables.


class EDXS_Table:

    # OV means over-voltage
    # These are the files to compute the ionization cross_sections of each shell
    # These tables were extracted from Bote2009 pdf tables. It was done through Notepad++ because of special characters in the pdf.
    LOW_OV_FILE = os.path.join(os.path.dirname(__file__), "Data", "Bote2009_lowOV.txt")
    HIGH_OV_FILE = os.path.join(
        os.path.dirname(__file__), "Data", "Bote2009_highOV.txt"
    )

    def __init__(self, energy_range=20, beam_energy=200, cs_threshold=1e-8):
        """
        :energy_range: Max energy of the detector in keV (float)
        :beam_energy: Energy of the electron beam in keV (float). Usual values 100 and 200.
        """
        # Full dictionary of the xraylib library
        self.xr_dict = xr.__dict__
        # Regex to generate the internal dicts
        self.re_shell = r"([K-M][0-9]?)(_SHELL)"
        self.re_line = r"([A-Z][0-9]?[A-Z][0-9])(_LINE)"
        # Internal dicts
        self.shell_dict = self.create_shell_dict()
        self.line_dict = self.create_line_dict()
        # list of all the supported elements
        self.elements = list(range(1, 100))
        self.table = {}
        self.energy_range = energy_range
        self.beam_energy = beam_energy
        self.low_OV_table = pd.read_fwf(EDXS_Table.LOW_OV_FILE).fillna(method="ffill")
        self.high_OV_table = pd.read_fwf(EDXS_Table.HIGH_OV_FILE).fillna(method="ffill")
        # Heights of the attenuation coefficients from the paper of Wernisch(1984)
        self.wernisch_H = {
            "L1": 1,
            "L2": 0.862,
            "L3": 0.611,
            "M1": 1,
            "M2": 0.935,
            "M3": 0.842,
            "M4": 0.638,
            "M5": 0.443,
        }
        # cross section threshold for characteristic X-rays
        self.cs_threshold = cs_threshold

    def create_shell_dict(self):
        """
        Creation of the internal dict which stores every SHELL macro from the xraylib.
        The dict has this structure : {"Shell name": int}
        """
        shell_dict = {}
        for key in self.xr_dict.keys():
            shell = re.match(self.re_shell, key)
            if shell:
                shell_dict[shell.group(1)] = self.xr_dict[key]
        return shell_dict

    def create_line_dict(self):
        """
        Creation of the internal dict which stores every LINE macro from the xraylib.
        The dict has this structure : {"Line name": int}
        """
        line_dict = {}
        for key in self.xr_dict.keys():
            line = re.match(self.re_line, key)
            if line:
                line_dict[line.group(1)] = self.xr_dict[key]
        return line_dict

    def transition_dict(self, shell):
        """
        Generates a dictionary of the lines associated with the given shell.
        The dict has this structure : {"line_name":int}
        """
        transition_dict = {}
        regex = shell + r"[H-Z][0-9]"
        for line in self.line_dict.keys():
            if re.match(regex, line):
                transition_dict[line] = self.line_dict[line]
        return transition_dict

    def generate_table(self):
        """
        Generates the table of X-ray line ratios for each element. The table takes the form of a nested dict with the following structure :
        {"table" :{"integer" : {"shell" : {"energies" : list of floats, "ratios" : list of floats}}}}
        """
        element_dict = {}
        for elt in self.elements:
            shells_dict = {}
            for shell in self.shell_dict.keys():
                trans_dict = self.transition_dict(shell)
                energies = []
                ratios = []
                quant_dict = {}
                for line in trans_dict.keys():
                    # Not all the shell and line combination are valid. They need to be checked for validity first
                    try:
                        ratio = (
                            self.CK_ionization(elt, shell)
                            * xr.FluorYield(elt, self.shell_dict[shell])
                            * xr.RadRate(elt, trans_dict[line])
                        )

                        energy = xr.LineEnergy(elt, trans_dict[line])
                        # There is no need to included the undetected lines as well as the minor ones
                        if (energy < self.energy_range) and (ratio > self.cs_threshold):
                            ratios.append(ratio)
                            energies.append(energy)
                    # if a shell line is rejected we go to the next one.
                    except ValueError:
                        pass
                # Some shells are empty, especially for light elements. We need to exclude that case.
                if not (ratios == []):
                    # The ratios are normalized
                    m = max(ratios)
                    ratios[:] = [i / m for i in ratios]
                    quant_dict["energies"] = energies
                    quant_dict["ratios"] = ratios
                # again if empty we exclude it
                if not (quant_dict == {}):
                    shells_dict[shell] = quant_dict
            element_dict[str(elt)] = shells_dict
        self.table["table"] = element_dict

    def first_ionization(self, element, shell):
        """
        Calculates the first ionization cross-section of the given shell of the given element using the Bote 2009 tables. These table are only valid for ionization of atoms by electrons.
        :element: integer
        :shell: Shell string
        """
        # Tries first to get the ionization energy if possible for the given shell and element
        try:
            energy = self.low_OV_table.loc[
                (self.low_OV_table["Z"] == element) & (self.low_OV_table["S"] == shell)
            ].iloc[0]["E"]
            U = self.beam_energy * 1000 / energy
            # There are two analytical
            if U < 16:
                # Full expression of the cross section for low OV : 4*pi*a0**2*(U-1)/(U**2)*(a1 + a2*U + a3/(1+U) + a4/(1+U)**3 + a5/(1+U)**5)
                # Useful expression of X-ray ratios : (a1 + a2*U + a3/(1+U) + a4/(1+U)**3 + a5/(1+U)**5)
                # U is OverVoltage (OV) and a0 is the Bohr radius
                a_values = self.low_OV_table.loc[
                    (self.low_OV_table["Z"] == element)
                    & (self.low_OV_table["S"] == shell)
                ]
                a1 = a_values["a1"].iloc[0]
                a2 = a_values["a2"].iloc[0]
                a3 = a_values["a3"].iloc[0]
                a4 = a_values["a4"].iloc[0]
                a5 = a_values["a5"].iloc[0]
                cs = (
                    a1
                    + a2 * U
                    + a3 / (1 + U)
                    + a4 / ((1 + U) ** 3)
                    + a5 / ((1 + U) ** 5)
                )
            else:
                # Full expression of the cross section for high OV :
                # Eb/(Eb + b*Ek)*4*pi*a0**2*A/B**2 *((ln(X**2)-B**2)*(1+g1/X) + g2 + g3*(1-B**2)**(-1/4) + g4/X)
                param_values = self.high_OV_table.loc[
                    (self.high_OV_table["Z"] == element)
                    & (self.high_OV_table["S"] == shell)
                ]
                me = 511000  # electron rest mass in eV
                beam_ev = self.beam_energy * 1000
                B = np.sqrt(beam_ev * (beam_ev + 2 * me)) / (beam_ev + me)
                X = np.sqrt(beam_ev * (beam_ev + 2 * me)) / me
                b_ = param_values["b-"].iloc[0]
                A = param_values["Anlj"].iloc[0]
                g1 = param_values["g1"].iloc[0]
                g2 = param_values["g2"].iloc[0]
                g3 = param_values["g3"].iloc[0]
                g4 = param_values["g4"].iloc[0]
                cs = (
                    U
                    / (U + b_)
                    * A
                    * (
                        (np.log(np.power(X, 2)) - B ** 2) * (1 + g1 / X)
                        + g2
                        + g3 * np.power((1 - B ** 2), -0.25)
                        + g4 / X
                    )
                )

            return cs
        # returns the same error as the xraylib to have homogeneous execptions
        except IndexError as exc:
            raise ValueError("first ionization : wrong shell") from exc

    def CK_ionization(self, element, shell):
        """
        Calculates the ionization cross-section corrected to include coster kronig transitions. This simple model corresponds to the first part of Schoonjans2011 cross section calculation. A more complete model can be implemented using the second part of Schoonjans2011.
        """
        K = re.match("K", shell)
        L = re.match(r"(L)([1-3])", shell)
        M = re.match(r"(M)([1-5])", shell)
        # For K shell there is no need for a correction because there is no sub-shell
        if K:
            cs = 1
        elif L:
            # The first term is the direct ionization of the sub-shell
            cs = self.first_ionization(element, shell)
            # For the L1, there is not coster kronig
            # For the higher shells there are transitions from the lower shells added too
            for i in range(1, int(L.group(2))):
                try:
                    string_CK = "FL{}{}_TRANS".format(i, L.group(2))
                    string_shell = "L{}".format(i)
                    cs += self.first_ionization(
                        element, string_shell
                    ) * xr.CosKronTransProb(element, self.xr_dict[string_CK])
                except ValueError:
                    pass
        # Same as L with more terms.
        elif M:
            cs = self.first_ionization(element, shell)
            for i in range(1, int(M.group(2))):
                try:
                    string_CK = "FM{}{}_TRANS".format(i, M.group(2))
                    string_shell = "M{}".format(i)
                    cs += self.first_ionization(
                        element, string_shell
                    ) * xr.CosKronTransProb(element, self.xr_dict[string_CK])
                except ValueError:
                    pass
        # If there is something wrong : raise the same error as xraylib for consistency
        else:
            raise ValueError("CK ionization : wrong shell")
        return cs

    def save_table(self, filename):
        """
        Saves the self.table to a json file with nice indents.
        """
        with open(filename, "w") as f:
            json.dump(self.table, f, indent=4)

    # Functions to calculate the attenuation coefficients tables. Not very useful yet.
    def wernisch_abs_table(self):
        """
        docstring
        """
        L, M = False, False
        list_L = ["L1", "L2", "L3"]
        list_M = ["M1", "M2", "M3", "M4", "M5"]
        element_dict = {}
        for elt in self.elements:
            shells_dict = {}
            try:
                xr.EdgeEnergy(elt, self.xr_dict["L1_SHELL"])
                L = True
                for shell in list_L:
                    shells_dict[shell] = [
                        xr.EdgeEnergy(elt, self.xr_dict[shell + "_SHELL"]),
                        self.wernisch_H[shell],
                    ]
            except ValueError:
                pass
            try:
                xr.EdgeEnergy(elt, self.xr_dict["M1_SHELL"])
                M = True
                for shell in list_M:
                    shells_dict[shell] = [
                        xr.EdgeEnergy(elt, self.xr_dict[shell + "_SHELL"]),
                        self.wernisch_H[shell],
                    ]
            except ValueError:
                pass
            if M and L:
                d = [
                    self.wernisch_dK(elt),
                    self.wernisch_dL(elt),
                    self.wernisch_dM(elt),
                    self.wernisch_dN(elt),
                ]
                k = [-2.685, -2.669, -2.514, -2.451]
            elif L and not (M):
                d = [
                    self.wernisch_dK(elt),
                    self.wernisch_dL(elt),
                    self.wernisch_dM(elt),
                ]
                k = [-2.685, -2.669, -2.514]
            else:
                d = [self.wernisch_dK(elt), self.wernisch_dL(elt)]
                k = [-2.685, -2.669]
            shells_dict["K"] = [xr.EdgeEnergy(elt, self.xr_dict["K_SHELL"]), 1.0]
            shells_dict["d"] = d
            shells_dict["exp"] = k
            L, M = False, False
            element_dict[str(elt)] = shells_dict
        self.table["table"] = element_dict

    def wernisch_dK(self, Z):
        return (
            5.955
            + 3.917e-1 * Z
            - 1.054e-2 * np.power(Z, 2)
            + 1.520e-4 * np.power(Z, 3)
            - 8.508e-7 * np.power(Z, 4)
        )

    def wernisch_dL(self, Z):
        return (
            3.257
            + 3.936e-1 * Z
            - 8.483e-3 * np.power(Z, 2)
            + 9.491e-5 * np.power(Z, 3)
            - 4.058e-7 * np.power(Z, 4)
        )

    def wernisch_dM(self, Z):
        return (
            2.382 + 2.212e-1 * Z - 2.028e-3 * np.power(Z, 2) + 6.891e-6 * np.power(Z, 3)
        )

    def wernisch_dN(self, Z):
        return 4.838 + 4.911e-2 * Z

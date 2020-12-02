import numpy as np
import json
import os
from math import sqrt, pi
from scipy.special import erf

# I will shortly aim at redisigning this part of the code so that energy values, intensity ratios, etc... will be taken from xraylib, etc ....

class Gaussians:
    """
    Contains utility functions to handle energy spectra composed of Gaussians and X-ray continuum.
    One of the functions is used to build the G matrix in SNMF.
    """

    XRAYS_FILE = os.path.join(os.path.dirname(__file__), "Data", "xrays_V2.json")

    def __init__(self, e_offset=0.20805000000000007, e_size=1980, e_scale=.01,width_slope=0.01,width_intercept=0.065):
        """
        Create an instance of the Gaussians class
        :e_offset: offset in energy (float)
        :esize: number of channels in the data (int)
        :e_scale: energy scale in keV / channel (float)
        :width_slope: slope of the linear approximation to detector broadening (float)
        :width_intercept: intercept of the linear approximation to detector broadening (float)
        """
        # Create energy scale
        self.x = np.arange(e_offset, e_offset+e_size*e_scale, e_scale)
        # Initialize a spectrum
        self.y = np.zeros_like(self.x)

        # The gaussian width is determined by the detector. This width follows a linear relationship with energy
        # These parameters were fitted using experimental data
        self.width_slope = 0.01
        self.width_intercept = 0.065

        # dict extracted from the X_rays database
        with open(Gaussians.XRAYS_FILE, "r") as f:
            self.xrays = json.load(f)["table"]

    def create_spectrum(self, elements):
        """
        Create an 'ideal' Gaussian spectrum for the provided elements
        :param elements: dictionary mapping an element, to the concentration
        :return: spectrum (array 1D)
        """
        for element in self.xrays:
            # gets the concentration specified in the input dict
            conc = elements.get(element["element"])
            if conc is not None:
                # Create the gaussian corresponding to the input element
                for i, energy in enumerate(element["energies"]):
                    # Generates the corresponding energy width
                    width = self.width_slope * energy + self.width_intercept
                    # Adds the gaussian to the total spectrum
                    self.y += conc * element["ratios"][i] * Distributions.pdf_gaussian(energy,
                                                                                  width/2.3548,
                                                                                  self.x)
        # Normalize the spectrum
        self.y=self.y/np.max(self.y)
        return self.y

    def create_matrix(self):
        """
        Creates a matrix containing all possible Gaussians (based on the elements in the xrays file).
        """
        g_matr = np.zeros((self.x.size, 0))
        for entry in self.xrays :
            signal = np.zeros((self.x.size, 1))
            # Generates a series of peaks for on given element with the corresponding ratios between peaks
            for i, energy in enumerate(entry["energies"]):
                # Generates the corresponding energy width
                width = self.width_slope * energy + self.width_intercept
                signal += entry["ratios"][i] * Distributions.pdf_gaussian(energy,width/2.3548,self.x)[np.newaxis].T
            # Add the series of peaks of the element to the g_matr
            g_matr = np.concatenate((g_matr, signal), axis=1)
        return g_matr

    def add_bremsstrahlung(self,method,scale,*params) :
        """
        Adds the continuum X-rays to the current spectrum
        :method: Selection of the method used to model the continuum (string). If method is an array it is directly used as the continuum model
        :scale: Intensity of the bremsstrahlung (float)
        :*params: arguments of the Distributions functions
        """
        if method == "eggert" :
            self.y=self.y*scale + Distributions.eggert_brstlg(self.x,*params)
        elif method == "analytical" :
            self.y = self.y*scale + Distributions.analytical_brstlg(self.x,*params)
        elif method == "simplified" :
            self.y = self.y*scale + Distributions.simplified_brstlg(self.x,*params)
        elif method == "simplified2" :
            self.y = self.y*scale + Distributions.simplified_brstlg_2(self.x,*params)
        # Bypass of modelling functions
        elif type(method) == np.ndarray :
            self.y = self.y + method
        else :
            pass

    ############################
    # Old and unused functions #
    ############################

    # These functions were designed so that it would be possible to select the elements used in the model

    # def create_elt_dict(self) :
    #     #There should be a better of keeping track of the gaussians in the g_matrix and their indices
    #     elt_dict={}
    #     i=0
    #     for element in sorted(self.xrays, key=lambda x: x["element"]):
    #         elt_dict[element["element"]]=i
    #         i+=1
    #     return elt_dict

    # def create_matrix(self,elements=None):
    #     """
    #     Creates a matrix containing all possible Gaussians (based on the elements in the xrays file
    #     """
    #     #To modify so that it takes elements as inputs and therefore limits the size of the g_matrix
    #     g_matr = np.zeros((self.x.size, 0))
    #     elt_dict={}
    #     j=0
    #     for entry in self.xrays :
    #         if elements is None :
    #             signal = np.zeros((self.x.size, 1))
    #             for i, energy in enumerate(entry["energies"]):
    #                 width = self.width_slope * energy + self.width_intercept
    #                 signal += entry["ratios"][i] * Distributions.pdf_gaussian(energy,width/2.3548,self.x)[np.newaxis].T
    #             g_matr = np.concatenate((g_matr, signal), axis=1)
    #         else :
    #             if entry["element"] in elements :
    #                 elt_dict[entry["element"]]=j
    #                 j+=1
    #                 signal = np.zeros((self.x.size, 1))
    #                 for i, energy in enumerate(entry["energies"]):
    #                     width = self.width_slope * energy + self.width_intercept
    #                     signal += entry["ratios"][i] * Distributions.pdf_gaussian(energy,width/2.3548,self.x)[np.newaxis].T
    #                 g_matr = np.concatenate((g_matr, signal), axis=1)
    #     return g_matr,elt_dict

    # def create_spectrum_decomp(self, elements):
    # """
    # Create an array of peak values of the Gaussian peaks in the spectrum
    # :param elements: dictionary mapping an element, to the concentration
    # :return: array of peaks
    # """
    # y = []
    # for element in sorted(self.xrays, key=lambda x: x["element"]):
    #     conc = elements.get(element["element"].split("-")[0], 0.)
    #     y.append(conc)

    # return np.array(y)


class Distributions:
    """
    Functions used to calculate the EDXS spectra
    """

    @staticmethod
    def pdf_gaussian(mu, sigma, scale):
        """
        Gaussian distribution 
        :mu: average (float)
        :sigma: standard deviation (float)
        :scale: x-axis (np.array)
        """
        return np.exp(-np.power(scale - mu, 2) / (2 * np.power(sigma, 2)))

    #######################################
    # Old versions of the X-ray continuum #
    #######################################

    @staticmethod
    def eggert_brstlg(x,a0,a1,a2,mu) :
        return (a0*((100-x)/x)+a1*((100-x)**2/x))*np.exp(mu/np.power(x,a2))

    @staticmethod
    def analytical_brstlg(x,c0,c1,n0,n1,n2) :
        mu = c0/np.power(x,3)
        mu_det = c1/np.power(x,3)
        # A=1/(a0+a1*mu+a2*np.power(mu,2))
        #A = 1/(a0+a2*np.power(mu,2))
        A = (1-np.exp(-mu))/mu
        #D = np.exp(-d0*mu_det)*(1-np.exp(-d1*mu_det))
        D= 1 - np.exp(-mu_det)
        N = n0/x + n1 + n2*x
        return N*D*A

    @staticmethod
    def simplified_brstlg(x,b0,b1,b2,c0) :
        mu = c0/np.power(x,3)
        A = (1-np.exp(-mu))/mu
        D = 1 - np.exp(-mu)
        B = b0/x + b1 + b2*x
        return A*B*D

    ##############################################
    # Up to date function of the X-ray continuum #
    ##############################################

    @staticmethod
    def simplified_brstlg_2(x,b0,b1,b2,c0,c1,c2) :
        """
        Full calculation of the X-ray continuum.
        :b0: first bremsstrahlung parameter (float)
        :b1: second bremsstrahlung parameter (float)
        :b2: third bremsstrahlung parameter (float)
        :c0: self-absorption parameter (float)
        :c1: first detector parameter (float)
        :c2: second detector parameter (float)
        """
        # Mass attenuation coefficients
        mu = c0/np.power(x,3)
        mu_det = c1/np.power(x,3)
        mu_dl = c2/np.power(x,3)
        # Self-absorption
        A = (1-np.exp(-mu))/(mu)
        # Detector
        D= (1 - np.exp(-mu_det))*np.exp(-mu_dl)
        # Bremsstrahlung
        B = b0/x + b2*x  + b1
        return B*D*A

########################################
# Old functions used by Thomas Holvoet #
########################################

    # @staticmethod
    # def _cdf_log_normal(mus, sigmas, scale):
    #     """
    #     Cumulative distribution function of the Log Normal Distribution
    #     :param mus: array of means (or single mean)
    #     :param sigmas: array of std's (or single std)
    #     :param scale: the x scale
    #     :return: array with the cum distr for every (mu, sigma) pair
    #     """
    #     with np.errstate(divide='ignore'):
    #         return .5 + .5 * erf((np.log(scale) - mus) / sqrt(2) / sigmas)

    # @staticmethod
    # def _cdf_log_normal_dmu(mus, sigmas, scale):
    #     """
    #     Derivative with respect to mu of the cumulative distribution function of the Log Normal Distribution
    #     :param mus: array of means (or single mean)
    #     :param sigmas: array of std's (or single std)
    #     :param scale: the x scale
    #     :return: array with the derivative of the cum distr for every (mu, sigma) pair
    #     """
    #     with np.errstate(divide='ignore'):
    #         return -1 / sqrt(2*pi) / sigmas * np.exp(-1 / 2 / sigmas**2 * (np.log(scale) - mus) ** 2)

    # @staticmethod
    # def _cdf_log_normal_dsigma(mus, sigmas, scale):
    #     """
    #     Derivative with respect to sigma of the cumulative distribution function of the Log Normal Distribution
    #     :param mus: array of means (or single mean)
    #     :param sigmas: array of std's (or single std)
    #     :param scale: the x scale
    #     :return: array with the derivative of the cum distr for every (mu, sigma) pair
    #     """
    #     with np.errstate(divide='ignore'):
    #         if scale[0, 0] == 0:
    #             return np.concatenate((np.zeros((1, len(mus))), - (np.log(scale[1:]) - mus) / sqrt(2*pi) / sigmas**2 *
    #                                    np.exp(-1 / 2 / sigmas**2 * (np.log(scale[1:]) - mus)**2)), axis=0)
    #         else:
    #             return - (np.log(scale) - mus) / sqrt(2 * pi) / sigmas ** 2 * \
    #                 np.exp(-1 / 2 / sigmas ** 2 * (np.log(scale) - mus) ** 2)

    # @staticmethod
    # def pdf_log_normal(mus, sigmas, scale):
    #     """
    #     Density function of the Log Normal Distribution
    #     :param mus: array of means (or single mean)
    #     :param sigmas: array of std's (or single std)
    #     :param scale: the x scale
    #     :return: array with the density for every (mu, sigma) pair
    #     """
    #     cdf = Distributions._cdf_log_normal(mus, sigmas, scale)
    #     return cdf[1:] - cdf[:-1]

    # @staticmethod
    # def pdf_log_normal_dmu(mus, sigmas, scale):
    #     """
    #     Derivative with respect to mu of the density function of the Log Normal Distribution
    #     :param mus: array of means (or single mean)
    #     :param sigmas: array of std's (or single std)
    #     :param scale: the x scale
    #     :return: array with the derivative of the density for every (mu, sigma) pair
    #     """
    #     cdf = Distributions._cdf_log_normal_dmu(mus, sigmas, scale)
    #     return cdf[1:] - cdf[:-1]

    # @staticmethod
    # def pdf_log_normal_dsigma(mus, sigmas, scale):
    #     """
    #     Derivative with respect to sigma of the density function of the Log Normal Distribution
    #     :param mus: array of means (or single mean)
    #     :param sigmas: array of std's (or single std)
    #     :param scale: the x scale
    #     :return: array with the derivative of the density for every (mu, sigma) pair
    #     """
    #     cdf = Distributions._cdf_log_normal_dsigma(mus, sigmas, scale)
    #     return cdf[1:] - cdf[:-1]


# class MatrixUtils:
#     """
#     Some utility matrix operations
#     """
#     @staticmethod
#     def shift_matrix_vert(matr, k):
#         """
#         Shift a matrix by k rows
#         """
#         if k >= 0:
#             return np.concatenate((np.zeros((k, *matr.shape[1:])), matr[:-k]), axis=0)
#         else:
#             k = -k
#             return np.concatenate((matr[k:], np.zeros((k, *matr.shape[1:]))), axis=0)

#     @staticmethod
#     def shift_matrix_horiz(matr, k):
#         """
#         Shift a matrix by k columns
#         """
#         if k >= 0:
#             return np.concatenate((np.zeros((matr.shape[0], k, *matr.shape[2:])), matr[:, :-k]), axis=1)
#         else:
#             k = -k
#             return np.concatenate((matr[:, k:], np.zeros((matr.shape[0], k, *matr.shape[2:]))), axis=1)

#     @staticmethod
#     def shift_matrix_diag(matr, k):
#         """
#         Shift a matrix by k rows and k columns
#         """
#         if k >= 0:
#             res = np.concatenate((np.zeros((k, matr.shape[1]-k, *matr.shape[2:])), matr[:-k, :-k]), axis=0)
#             return np.concatenate((np.zeros((matr.shape[0], k, *matr.shape[2:])), res), axis=1)
#         else:
#             k = -k
#             res = np.concatenate((matr[k:, k:], np.zeros((k, matr.shape[1]-k, *matr.shape[2:]))), axis=0)
#             return np.concatenate((np.zeros((matr.shape[0], k, *matr.shape[2:])), res), axis=1)

########################
# Comparison functions #
########################

class MetricsUtils :
    """
    Utility functions to assess the performances of the algorithm
    """

    @staticmethod
    def spectral_angle(v1, v2):
        """
        Calculates the angle between two spectra. They have to have the same dimension.
        :v1: first spectrum (np.array 1D)
        :v2: second spectrum (np.array 1D)
        """
        v1_u = v1/np.linalg.norm(v1)
        v2_u = v2/np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))*180/np.pi

    @staticmethod
    def MSE_map(map1,map2) :
        """
        Calculates the mean squared error between two 2D arrays. They have to have the same dimension.
        :map1: first array (np.array 2D)
        :map2: second array (np.array 2D)
        """
        tr_m1_m1=np.einsum("ij,ij->", map1, map1)
        tr_m2_m2=np.einsum("ij,ij->", map2, map2)
        tr_m1_m2=np.trace(map1.T@map2)
        return tr_m1_m1 - 2*tr_m1_m2 + tr_m2_m2

    
import numpy as np
from scipy.special import erfc


class Functions:
    @staticmethod
    def gaussian(x, mu, sigma):
        """
        docstring
        """
        return (
            1
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-np.power(x - mu, 2) / (2 * np.power(sigma, 2)))
        )

    @staticmethod
    def simple_abs_coeff(x):
        return 1 / np.power(x, 3)

    @staticmethod
    def self_abs(abs_coeff, c0):
        """
        docstring
        """
        return (1 - np.exp(-c0 * abs_coeff)) / (c0 * abs_coeff)

    @staticmethod
    def bremsstrahlung(x, b0, b1, b2):
        return b0 / x + b1 + b2 * x

    @staticmethod
    def detector(abs_coeff, c1, c2):
        return np.exp(-c2 * abs_coeff) * (1 - np.exp(-c1 * abs_coeff))

    @staticmethod
    def shelf(x, height, length):
        return height * erfc(x - length)

    @staticmethod
    def spectral_angle(v1, v2):
        """
        Calculates the angle between two spectra. They have to have the same dimension.
        :v1: first spectrum (np.array 1D)
        :v2: second spectrum (np.array 1D)
        """
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / np.pi

    @staticmethod
    def MSE_map(map1, map2):
        """
        Calculates the mean squared error between two 2D arrays. They have to have the same dimension.
        :map1: first array (np.array 2D)
        :map2: second array (np.array 2D)
        """
        tr_m1_m1 = np.einsum("ij,ij->", map1, map1)
        tr_m2_m2 = np.einsum("ij,ij->", map2, map2)
        tr_m1_m2 = np.trace(map1.T @ map2)
        return tr_m1_m1 - 2 * tr_m1_m2 + tr_m2_m2

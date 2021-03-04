import numpy as np
from scipy.special import erfc

    
def gaussian(x, mu, sigma):
    """
    docstring
    """
    return (
        1
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-np.power(x - mu, 2) / (2 * np.power(sigma, 2)))
    )


def simple_abs_coeff(x):
    return 1 / np.power(x, 3)


def self_abs(abs_coeff, c0):
    """
    docstring
    """
    return (1 - np.exp(-c0 * abs_coeff)) / (c0 * abs_coeff)


def bremsstrahlung(x, b0, b1, b2):
    return b0 / x + b1 + b2 * x


def detector(abs_coeff, c1, c2):
    return np.exp(-c2 * abs_coeff) * (1 - np.exp(-c1 * abs_coeff))


def shelf(x, height, length):
    return height * erfc(x - length)

def continuum_xrays(x,params_dict,abs_coeff):
    """
    Computes the continuum X-ray based on the brstlg_pars set during init.
    The function is built in a way so that even an incomplete brstlg_pars dict is not a problem.
    """
    try:
        B = bremsstrahlung(
            x,
            params_dict["b0"],
            params_dict["b1"],
            params_dict["b2"],
        )
    except KeyError:
        B = np.ones_like(x)
    # Both A and D are built to use the attenuation coefficien (abs_coeff).
    # This needs to change because the attenuation coefficients in D should be different than the one in A.
    try:
        A = self_abs(abs_coeff, params_dict["c0"])
    except KeyError:
        A = np.ones_like(x)
    try:
        D = detector(
            abs_coeff, params_dict["c1"], params_dict["c2"]
        )
    except KeyError:
        D = np.ones_like(x)
    try:
        S = shelf(x, params_dict["h"], params_dict["l"])
    except KeyError:
        S = np.zeros_like(x)
    return B * A * D + S
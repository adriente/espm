r""" The :mod:`espm.measures` module implements different measures for the matrix factorisation problem.
In particular it contains the different losses and regularizers used in :mod:`espm.estimator` module. 
It also contains different metrics to evaluate the results.

"""

import numpy as np
from espm.conf import log_shift
import warnings as w
from itertools import permutations
from sklearn.metrics import r2_score

def spectral_angle(v1, v2):
    r"""Spectral angle

    Calculate the angle between two spectra of the same dimension.
    
    :param np.array 1D v1: first spectrum
    :param np.array 1D v2: second spectrum

    :returns: the answer
    
    :rtype: float

    Examples
    --------
    
    >>> import numpy as np
    >>> from espm.measures import spectral_angle
    >>> v1 = np.array([0, 1, 0])
    >>> v2 = np.array([1, 0, 1])
    >>> spectral_angle(v1, v2)
        90.0

    """

    if len(v1.shape)==1:
        if v1.shape != v2.shape:
            raise ValueError("v1 and v2 should have the same shape.")
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / np.pi
    else:
        if v1.shape[1] != v2.shape[1]:
            raise ValueError("The second dimensions of v1 and v2 should be the same.")
        v1_u = v1 / np.sqrt(np.sum((v1**2), axis=1, keepdims=True))
        v2_u = v2 / np.sqrt(np.sum((v2**2), axis=1, keepdims=True))
        return np.arccos(np.clip(v1_u @ v2_u.T, -1.0, 1.0)) * 180 / np.pi


def mse(map1, map2):
    r"""Mean square error

    Calculate the mean squared error between two 2D arrays of the same dimension.

    :param np.array 2D map1: first array
    :param np.array 2D map2: second array

    :returns: the answer

    Examples
    --------
    
    >>> import numpy as np
    >>> from espm.measures import mse
    >>> map1 = np.array([[0, 1][0, 1]])
    >>> map2 = np.array([[1, 1][1, 1]])
    >>> mse(map1, map2)
        0.5

    """
    return np.mean((map1-map2)**2)


def mae(map1, map2):
    r"""Mean average error

    Calculate the mean average error between two 2D arrays of the same dimension.

    :param np.array 2D map1: first array
    :param np.array 2D map2: second array
    
    :returns: the answer

    Examples
    --------
    
    >>> import numpy as np
    >>> from espm.measures import mae
    >>> map1 = np.array([[0, 1][0, 1]])
    >>> map2 = np.array([[1, 1][1, 1]])
    >>> mae(map1, map2)
        0.5

    """
    return np.mean(np.abs(map1-map2))


def r2(map_true, map_pred):
    r""":math:`R^2` - Coefficient of determination

    Calculates the coefficient of determination between two 2D arrays of the same dimension.
    This is also called regression score function. 
    See `wikipedia <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_.

    This function is a wrapper for the function  :mod:`sklearn.metrics.r2_score` of Scikit Learn.

    
    :param np.array 2D map1: first array
    :param np.array 2D map2: second array
    
    :returns: the answer

    :rtype: float
    """
    def reshape_2d(x):
        return x.reshape(x.shape[0], -1)
    return r2_score(reshape_2d(map_true), reshape_2d(map_pred))



# This function will find the best matching endmember for each true spectrum.
# This is useful since the A and P matrice are initialized at random.
# This function works but can probably greatly improved
def find_min_angle(true_vectors, algo_vectors, get_ind = False, unique=False):
    # This function calculates all the possible angles between endmembers and true spectra
    # For each true spectrum a best matching endmember is found
    # The function returns the angles of the corresponding pairs
    r"""Compare all the angles between ground truth spectra and NMF spectra and find the minimum configuration.

    :param np.array 2D true_vectors: true spectra with shape (number of phases, number of energy channels)
    :param np.array 2D algo_vectors: NMF spectra with shape (number of phases, number of energy channels)
    :param boolean get_ind: If True, the function will also return the indices of the NMF spectra corresponding to the ground truth
    :param boolean unique: If False it will find the global minimum but several spectra can be associated to the same ground truth
    
    :returns: list of angles, (optionally) tuple of indices
    :rtype: (list[float],list[int])
    ..warning:: 
        The output being either a list or a tuple of list isn't a great idea. It has to change.
    """
    angle_matr = spectral_angle(true_vectors,algo_vectors)
    if unique :
        ordered_angles = unique_min(angle_matr)
    else :
        ordered_angles = global_min(angle_matr)
    #unique minimum angles are ordered
    if get_ind :
        # if unique :
        #     print("Impossible to get indices when searching with unique minima.")
        # else : 
        return ordered_angles
    else : 
        return ordered_angles[0]

def global_min (matr) :
    res = []
    ind_res = []
    for i in matr : 
        ind_min = np.argmin(i)
        min = np.min(i)
        res.append(min)
        ind_res.append(ind_min)
    if any(ind_res.count(x) > 1 for x in ind_res) :
        w.warn("Several results share the same truth")
    return res, ind_res
    

def unique_min (matrix) : 
    '''

    From a square matrix of float values, finds the combination of elements with 
    different lines which mimises the sum of elements.
    
    It is a brute force algorithm, it is not recommended to input a matrix bigger than 20 x 20

    :param np.array 2D matrix: square matrix
    
    :returns: list of unique min values and corresponding indices in the same order

    :rtype: (list, list[int])

    Examples
    --------

    >>> import numpy as np
    >>> from espm.measures import unique_min
    >>> matrix = np.array([[1.2,  1.3,  3.5],
                        [4.9,  2.2,  6.5],
                       [9.0,  4.1,  1.8]])
    >>> unique_min(v1, v2)   
        ([1.2, 2.2, 1.8], (0, 1, 2))

    '''
    shape = matrix.shape[0]
    perms = list(permutations(range(shape),shape))
    list_sum_angles = []
    for perm in perms : 
        sum_angles = 0
        for i in range(shape) :
            sum_angles += matrix[perm[i],i]
        list_sum_angles.append(sum_angles)
    
    min_ind = list_sum_angles.index(min(list_sum_angles))
    mins = []
    for i in range(shape) : 
        mins.append(matrix[perms[min_ind][i],i])

    return mins, perms[min_ind]
    

# def unique_min (matr) : 
#     mins= []
#     ind_mins = []
#     for vec in matr :
#         mins.append(np.min(vec))
#         ind_min = np.argmin(vec)
#         ind_mins.append(ind_min)
#         matr[:,ind_min] = np.inf * np.ones(matr.shape[0])

#     return mins, ind_mins

def find_min_config(true_maps,true_spectra, algo_maps, algo_spectra,angles = True) : 
    min_MSE_config = find_min_MSE(true_maps,algo_maps,get_ind=True,unique=True)[1]
    min_angle_config = find_min_angle(true_spectra,algo_spectra,get_ind=True,unique=True)[1]
    warning = False

    if min_MSE_config != min_angle_config : 
        print("WARNING : angles and mse disagree there's probably an issue")
        warning = True

    if angles : 
        corresponding_angles = find_min_angle(true_spectra,algo_spectra,unique=True)
        min_config = min_angle_config
        corresponding_mse = ordered_mse(true_maps,algo_maps,min_angle_config)

    else : 
        corresponding_mse = find_min_MSE(true_maps,algo_maps,unique=True)
        min_config = min_MSE_config
        corresponding_angles = ordered_angles(true_spectra,algo_spectra,min_MSE_config)

    return corresponding_angles, corresponding_mse, min_config, warning



# This function works but can probably greatly improved
def find_min_MSE(true_maps, algo_maps, get_ind = False, unique=False):
    # This function calculates all the possible MSE between abundances and true maps
    # For each true map a best matching abundance is found
    # The function returns the MSE of the corresponding pairs
    r"""Compare all the mean squared errors between ground truth spectra and NMF spectra and find the minimum configuration.

    :param np.array 2D true_maps: true maps with shape (number of phases, number of pixels)
    :param np.array 2D algo_maps: NMF maps with shape (number of phases, number of pixels)
    :param boolean get_ind: If True, the function will also return the indices of the NMF maps corresponding to the ground truth
    :param boolean unique: If False it will find the global minimum but several maps can be associated to the same ground truth
    
    :returns: list of mse, (optionally) tuple of indices
    :rtype: (list[float],list[int])
    ..warning:: 
        The output being either a list or a tuple of list isn't a great idea. It has to change.
    """
    mse_matr = squared_distance(true_maps, algo_maps)
    if unique :
        ordered_maps = unique_min(mse_matr)
    else :
        ordered_maps = global_min(mse_matr)
    if get_ind :
        # if unique :
        #     print("Impossible to get indices when searching with unique minima.")
        # else : 
        return ordered_maps
    else : 
        return ordered_maps[0]

def ordered_mse (true_maps, algo_maps, input_inds) :
    '''
    input : p x Npx matrix of floats, p x Npx matrix of floats, list of integers
    output : list of floats
    %-------------------------%
    Takes true maps of p phases and Npx pixels, reconstructed maps of the same size and
    indices of the correspondance between true phases and reconstructed phases
    returns the mean squared errors of each phase in truth order.
    '''
    ordered_maps = []
    for i,j in enumerate(input_inds) : 
        ordered_maps.append(float(mse(true_maps[j], algo_maps[i])))
    return ordered_maps

def ordered_mae (true_maps, algo_maps, input_inds) :
    '''
    input : p x Npx matrix of floats, p x Npx matrix of floats, list of integers
    output : list of floats
    %-------------------------%
    Takes true maps of p phases and Npx pixels, reconstructed maps of the same size and
    indices of the correspondance between true phases and reconstructed phases
    returns the mean average errors of each phase in truth order.
    '''
    ordered_maps = []
    for i,j in enumerate(input_inds) : 
        ordered_maps.append(float(mae(true_maps[j], algo_maps[i])))
    return ordered_maps

def ordered_r2(true_maps, algo_maps, input_inds) :
    '''
    input : p x Npx matrix of floats, p x Npx matrix of floats, list of integers
    output : list of floats
    %-------------------------%
    Takes true maps of p phases and Npx pixels, reconstructed maps of the same size and
    indices of the correspondance between true phases and reconstructed phases
    returns the coefficient of determination of each phase in truth order.
    '''
    ordered_maps = []
    for i,j in enumerate(input_inds) : 
        ordered_maps.append(float(r2(true_maps[j], algo_maps[i])))
    return ordered_maps


def ordered_angles (true_spectra, algo_spectra, input_inds) :
    '''
    See ordered mse
    '''
    ordered_angles = []
    for i,j in enumerate(input_inds) : 
        ordered_angles.append(spectral_angle(true_spectra[j],algo_spectra[i]))
    return ordered_angles


# This function gives the residuals between the model determined by snmf and the data that were fitted
def residuals(data, model):
    X_sum = data.sum(axis=0).sum(axis=0)
    model_sum = (
        model.get_phase_map(0).sum() * model.get_phase_spectrum(0)
        + model.get_phase_map(1).sum() * model.get_phase_spectrum(1)
        + model.get_phase_map(2).sum() * model.get_phase_spectrum(2)
    )
    return X_sum - model_sum

def Frobenius_loss(X, W, H, average=False):
    r"""Frobenius norm of the difference between X and WH.
    
    Compute the Froebenius norm (elementwise L2 norm of a matrix) given :math:`X,W,H`:

    .. math::

        \| X - WH \|_F = \sum_{ji} \left| X_{ij} - (W H)_{ij} \right|^2

    :param np.array 2D X: n x m matrix
    :param np.array 2D W: n x k matrix
    :param np.array 2D H: k x m matrix
    :param boolean average: replace the sum with a mean, i.e.,
        divide the result by n*m (default False)

    :returns: the answer
    
    Examples
    --------
    
    >>> import numpy as np
    >>> from espm.measures import Frobenius_loss
    >>> X = np.array([[1, 1, -1], [2, 4, 5]])
    >>> W = np.array([[1], [1]])
    >>> H = np.array([[1, 2, 3]])
    >>> Frobenius_loss(X, W, H)
        26

    """

    DH = W @ H

    if average:
        return np.mean((DH - X)**2)
    else:
        return np.sum((DH - X)**2)

def KLdiv(X, D, H, log_shift=log_shift, average=False):
    r"""Generalized KL (Kullback–Leibler) divergence

    Compute the generalized KL divergence given :math:`X,W,H`:
    
    .. math::

        D_KL(X || WH) =  \sum_{ji} X_{ij} \log (X / D A)_{ij} + (D A - X)_{ij}


    :param np.array 2D X: n x m matrix
    :param np.array 2D W: n x k matrix
    :param np.array 2D H: k x m matrix
    :param float log_shift: small constant to ensure the KL divergence does 
        not explode (default value set in module :mod:`esppy.conf`)
    :param boolean average: replace the sum with a mean, i.e.,
        divide the result by n*m (default False)

    :returns: the answer

    :rtype: float
    
    Examples
    --------
    
    >>> import numpy as np
    >>> from espm.measures import KLdiv
    >>> X = np.array([[1, 1, 1], [2, 4, 5]])
    >>> W = np.array([[1], [1]])
    >>> H = np.array([[1, 2, 3]])
    >>> KLdiv(X, W, H)
        2.921251732961556

    """
    D = np.maximum(D, log_shift)
    H = np.maximum(H, log_shift)    
    X = np.maximum(X, log_shift)
    DH = D @ H
    return KL(X, DH, log_shift, average)

def KL(X, Y, log_shift=log_shift, average=False):
    r"""Generalized KL (Kullback–Leibler) divergence for two matrices

    .. math::

        D_KL(X || Y) =  \sum_{ji} X_{ij} \log (X / Y)_{ij} + (Y - X)_{ij}

    :param np.array 2D X: n x m matrix
    :param np.array 2D W: n x k matrix
    :param np.array 2D H: k x m matrix
    :param float log_shift: small constant to ensure the KL divergence does 
        not explode (default value set in module :mod:`esppy.conf`)
    :param boolean average: replace the sum with a mean, i.e.,
        divide the result by n*m (default False)

    :returns: the answer

    :rtype: float
    """
    X = np.maximum(X, log_shift)
    Y = np.maximum(Y, log_shift)
    if average:
        x_lin = np.mean(Y) - np.mean(X)
        x_log = np.mean(X*np.log(X)) - np.mean(X*np.log(Y))                
    else:
        x_lin = np.sum(Y) - np.sum(X)
        x_log = np.sum(X*np.log(X)) - np.sum(X*np.log(Y))
    return x_lin + x_log

def KLdiv_loss(X, W, H, log_shift=log_shift, average=False):
    r""" Generalized Generalized KL (Kullback–Leibler) divergence loss

    Compute the loss based on the generalized KL divergence given :math: `X,W,H`:

    .. math::

        \sum_{ji} X_{ij} \log (D W)_{ij} + (D W)_{ij}

    This does not contains all the term of the KL divergence, only the ones
    depending on W and H.

    :param np.array 2D X: n x m matrix
    :param np.array 2D Y: n x m matrix
    :param float log_shift: small constant to ensure the KL divergence does 
        not explode (default value set in module :mod:`esppy.conf`)
    :param boolean average: replace the sum with a mean, i.e.,
        divide the result by n*m (default False)

    :returns: the answer

    :rtype: float
    
    Examples
    --------

    >>> import numpy as np
    >>> from espm.measures import KLdiv, KLdiv_loss
    >>> X = np.array([[1, 1, 1], [2, 4, 5]])
    >>> W = np.array([[1], [1]])
    >>> H = np.array([[1, 2, 3]])
    >>> KLdiv_loss(X, W, H)
        1.9425903651915402ZSXDerzA
    >>> KLdiv(X, W, H)
        2.921251732961556
    """

    W = np.maximum(W, log_shift)
    H = np.maximum(H, log_shift)
    X = np.maximum(X, log_shift)

    Y = W @ H
    if average:
        x_lin = np.mean(Y)
        x_log = np.mean(X*np.log(Y))        
    else:
        x_lin = np.sum(Y)
        x_log = np.sum(X*np.log(Y))
    return x_lin - x_log

def KL_loss_surrogate(X, W, H, Ht, log_shift=log_shift, average=False):
    r""" Surrogate loss for the KL divergence."""

    W = np.maximum(W, log_shift)
    H = np.maximum(H, log_shift)
    Ht = np.maximum(Ht, log_shift)
    X = np.maximum(X, log_shift)

    WHT = np.expand_dims(W, axis=2) * np.expand_dims(Ht, axis=0)

    U = WHT/(np.sum(WHT, axis=1, keepdims=True))
    
    DH = np.expand_dims(W, axis=2) * np.expand_dims(H, axis=0)
    if average:
        return np.mean(X * np.sum(U * np.log(U/ DH), axis=1) + np.sum(WHT, axis=1))
    else:
        return np.sum(X * np.sum(U * np.log(U/ DH), axis=1)) + np.sum(WHT)

def log_reg(H, mu, epsilon=1, average=False):
    r""" Log regularisation

    Compute the regularization loss: 
    
    .. math:: 
        
        R(\mu, H, \epsilon) = \sum_{ij} \mu_i \log \left( H_{ij} + \epsilon \right).

    :param np.array 2D H: n x m matrix
    :param np.array 1D mu: n vector
    :param float epsilon: value of :math:`\epsilon` (default 1)
    :param boolean average: replace the sum with a mean, i.e.,
        divide the result by n*m (default False)

    :returns: the answer

    :rtype: float
    """
    if not(np.isscalar(mu)):
        mu = np.expand_dims(mu, axis=1)
    if average:
        return np.mean(mu* np.log(H+epsilon))        
    else:
        return np.sum(mu* np.log(H+epsilon))

def log_surrogate(H, Ht, mu, epsilon, average=False):
    r"""Surrogate loss for the log function."""
    if not(np.isscalar(mu)):
        mu = np.expand_dims(mu, axis=1)
    tmp = np.log(Ht+epsilon) + (H-Ht) / (epsilon + Ht)
    if average:
        return np.mean(mu* tmp)        
    else:
        return np.sum(mu* tmp)

def trace_xtLx(L, x, average=False):
    r""" Trace of :math:`X^T L X`

    Compute the following expression :math:`\text{Tr} (X^T L X)`.

    :param np.array 2D L: n x n matrix
    :param np.array 2D mu: n x k martrix of k n-sized vector
    :param boolean average: replace the sum with a mean, i.e.,
        divide the result by n*m (default False)

    :returns: the answer

    :rtype: float
    """
    if average:
        return np.mean(x * (L @ x))
    else:
        return np.sum(x * (L @ x))

def squared_distance(x, y=None):
    r""" Squared distance between two between all colon vectors matrices.

    Calculate the squared L2 distance between all pairs of vectors of two matrices.
    If only one matrix is given, the function uses each pair of vector of this matrix.

    :param np.array 2D x: n x m matrix of first colon vectors
    :param np.array 2D y: n x m matrix of second colon vectors (optional)

    :returns: the answer

    :rtype: np.array (m x m)
        
    Examples
    --------
    
    >>> import numpy as np
    >>> from espm.measures import square_distance
    >>> x = np.arange(3)
    >>> square_distance(x, x)
        array([[ 0.,  1.,  2.],
        [ 1.,  0.,  1.],
        [ 2.,  1.,  0.]])

    """

    try:
        x.shape[1]
    except IndexError:
        x = x.reshape(1, x.shape[0])    
    if y is None:
        y = x    
    else:
        try:

            y.shape[1]
        except IndexError:
            y = y.reshape(1, y.shape[0])    
    rx, cx = x.shape
    ry, cy = y.shape    
    # Size verification
    if cx != cy:
        raise ValueError("The sizes of x and y do not fit")    
    xx = (x * x).sum(axis=1)
    yy = (y * y).sum(axis=1)
    xy = np.dot(x, y.T)    
    d = abs(np.kron(np.ones((ry, 1)), xx).T +np.kron(np.ones((rx, 1)), yy) - 2 * xy)    
    
    return d / cx


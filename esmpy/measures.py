import numpy as np
from esmpy.conf import log_shift
import warnings as w
from itertools import permutations
from sklearn.metrics import r2_score

def spectral_angle(v1, v2):
    """
    Calculates the angle between two spectra. They have to have the same dimension.
    :v1: first spectrum (np.array 1D)
    :v2: second spectrum (np.array 1D)
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
    """Mean square error.

    Calculates the mean squared error between two 2D arrays. They have to have the same dimension.
    :map1: (np.array 2D) first array
    :map2: (np.array 2D) second array
    """
    return np.mean((map1-map2)**2)


def mae(map1, map2):
    """Mean square error.

    Calculates the mean average error between two 2D arrays. They have to have the same dimension.
    :map1: (np.array 2D) first array
    :map2: (np.array 2D) second array
    """
    return np.mean(np.abs(map1-map2))


def r2(map_true, map_pred):
    """ (coefficient of determination) regression score function.

    Calculates the coefficient of determination between two 2D arrays. They have to have the same dimension.
    :map1: (np.array 2D) first array
    :map2: (np.array 2D) second array
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
    

def unique_min (matr) : 
    '''
    input : N x N matrix of floats
    output : list of N unique min values and corresponding indices in the same order
    %-----------------------------------%
    From a N x N matrix of float values, finds the combination of elements with different lines which mimises the sum of elements.
    From :
    1.2  1.3  3.5
    4.9  2.2  6.5
    9.0  4.1  1.8

    it returns : 
    [1.2, 2.2, 1.8], [0, 1, 2]

    It is a very simple brute force algorithm, it is not recommended to input a matrix bigger than 20 x 20
    '''
    shape = matr.shape[0]
    perms = list(permutations(range(shape),shape))
    list_sum_angles = []
    for perm in perms : 
        sum_angles = 0
        for i in range(shape) :
            sum_angles += matr[perm[i],i]
        list_sum_angles.append(sum_angles)
    
    min_ind = list_sum_angles.index(min(list_sum_angles))
    mins = []
    for i in range(shape) : 
        mins.append(matr[perms[min_ind][i],i])

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
    mse_matr = square_distance(true_maps, algo_maps)
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
    """
    Compute the generalized KL divergence.
 
    \sum_{ji} | X_{ij} - (D H)_{ij} |^2
    """
    
    DH = W @ H

    if average:
        return np.mean((DH - X)**2)
    else:
        return np.sum((DH - X)**2)

def KLdiv(X, D, H, eps=log_shift, safe=True, average=False):
    """
    Compute the generalized KL divergence.

    \sum_{ji} X_{ij} \log (X / D A)_{ij} + (D A - X)_{ij}
    """
    if safe:
        # Allow for very small negative values!
        assert(np.sum(H<-log_shift/2)==0)
        assert(np.sum(D<-log_shift/2)==0)
    
    DH = D @ H
    return KL(X, DH, log_shift, average)

def KL(X, DH, eps=log_shift, average=False):
    if average:
        x_lin = np.mean(DH) - np.mean(X)
        x_log = np.mean(X*np.log(X+ eps)) - np.mean(X*np.log(DH + eps))                
    else:
        x_lin = np.sum(DH) - np.sum(X)
        x_log = np.sum(X*np.log(X+ eps)) - np.sum(X*np.log(DH + eps))
    return x_lin + x_log

def KLdiv_loss(X, D, H, eps=log_shift, safe=True, average=False):
    """
    Compute the loss based on the generalized KL divergence.

    \sum_{ji} X_{ij} \log (D W)_{ij} + (D W)_{ij}

    This does not contains all the term of the KL divergence, only the ones
    depending on D and A.
    """
    if safe:
        # Allow for very small negative values!
        assert(np.sum(H<-log_shift/2)==0)
        assert(np.sum(D<-log_shift/2)==0)
    
    DH = D @ H
    if average:
        x_lin = np.mean(DH)
        x_log = np.mean(X*np.log(DH + eps))        
    else:
        x_lin = np.sum(DH)
        x_log = np.sum(X*np.log(DH + eps))
    return x_lin - x_log

def KL_loss_surrogate(X, D, H, Ht, eps=log_shift, safe=True, average=False):
    DHT = np.expand_dims(D, axis=2) * np.expand_dims(Ht, axis=0)
    U = DHT/(np.sum(DHT, axis=1, keepdims=True)+eps)
    
    DH = np.expand_dims(D, axis=2) * np.expand_dims(H, axis=0)
    if average:
        return np.mean(X * np.sum(U * np.log((U+eps)/ (DH+eps)), axis=1) + np.sum(DHT, axis=1))
    else:
        return np.sum(X * np.sum(U * np.log((U+eps)/ (DH+eps)), axis=1)) + np.sum(DHT)

def log_reg(H, mu, epsilon, average=False):
    """
    Compute the regularization loss: \sum_ij mu_i \log(A_{ij})
    """
    if not(np.isscalar(mu)):
        mu = np.expand_dims(mu, axis=1)
    if average:
        return np.mean(mu* np.log(H+epsilon))        
    else:
        return np.sum(mu* np.log(H+epsilon))

def log_surrogate(H, Ht, mu, epsilon, average=False):

    if not(np.isscalar(mu)):
        mu = np.expand_dims(mu, axis=1)
    tmp = np.log(Ht+epsilon) + (H-Ht) / (epsilon + Ht)
    if average:
        return np.mean(mu* tmp)        
    else:
        return np.sum(mu* tmp)

def trace_xtLx(L, x, average=False):
    if average:
        return np.mean(x * (L @ x))
    else:
        return np.sum(x * (L @ x))

def square_distance(x, y=None):
    r"""
    Calculate the distance between two colon vectors.    Parameters
    ----------
    x : ndarray
        First colon vector
    y : ndarray
        Second colon vector    Returns
    -------
    d : ndarray
        Distance between x and y    Examples
    --------
    >>> from pygsp import utils
    >>> x = np.arange(3)
    >>> utils.distanz(x, x)
    array([[ 0.,  1.,  2.],
           [ 1.,  0.,  1.],
           [ 2.,  1.,  0.]])    """
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


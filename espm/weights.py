"""Functions to generate weights for synthetic datasets."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from espm.conf import BASE_PATH
from skimage.filters import median


def load_toy_weights():
    r"""Load the toy problem weights.
    
    Returns
    -------
    Hdot : np.ndarray
        The weights of the toy problem.

    Example
    -------

    .. plot::
        :context: close-figs

        >>> from espm.weights import load_toy_weights
        >>> import matplotlib.pyplot as plt
        >>> Hdot = load_toy_weights()
        >>> print(Hdot.shape)
        (3, 200, 200)
        >>> print(Hdot.dtype)
        float64
        >>> plt.imshow(Hdot[0])
        >>> plt.show()       
     
    """

    im1 = plt.imread(BASE_PATH / Path("datasets/toy-problem/phase1.png"))
    im1 = (1-np.mean(im1, axis=2)) *0.5

    im2 = plt.imread(BASE_PATH / Path("datasets/toy-problem/phase2.png"))
    im2 = (1-np.mean(im2, axis=2)) *0.5

    im0 = 1 - im1 - im2 

    Hdot = np.array([im0, im1, im2])

    return Hdot
   

def random_weights(shape_2d, n_phases=3, seed=0) :
    """ Generates a random weight matrix with a uniform distribution.

    The random weights are then normalized to sum up to one.

    Parameters
    ----------
    shape_2d : tuple
        Shape of the weight matrix.
    n_phases : int, optional
        Number of phases. The default is 3.
    seed : int, optional
        Seed for the random number generator. The default is 0.
    
    Returns
    -------
    weights : numpy.ndarray
        Weight matrix with uniform distribution.
    
    """
    np.random.seed(seed)
    rnd_array = np.random.rand(*shape_2d, n_phases)
    weights = rnd_array/np.sum(rnd_array, axis=2, keepdims=True)
    return weights
    

def laplacian_weights(shape_2d, n_phases=3, seed=0) :
    """ Generates a random weight matrix with a laplacian distribution.

    The random weight are then filtered with median filter 2 times to smooth the distribution.
    Eventually, the result is normalized to sum up to one.

    Parameters
    ----------
    shape_2d : tuple
        Shape of the weight matrix.
    n_phases : int, optional
        Number of phases. The default is 3.
    seed : int, optional
        Seed for the random number generator. The default is 0.
    
    Returns
    -------
    weights : numpy.ndarray
        Weight matrix with laplacian distribution.

    Example
    -------

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> weights = laplacian_weights((100,100), 3, 0)
        >>> plt.imshow(weights[:,:,0])
        >>> plt.show()

    """
    np.random.seed(seed)
    rnd_array = np.random.rand(shape_2d[0], shape_2d[1], n_phases)
    rnd_f = []
    for i in range(rnd_array.shape[2]):
        rnd_f.append(median(median(rnd_array[:,:,i])))
    rnd_f = np.array(rnd_f).transpose([1,2,0])
    weights = rnd_f/np.sum(rnd_f, axis=2, keepdims=True)
    return weights
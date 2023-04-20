r"""
Predetermined weights
---------------------

The :mod: `espm.weights.generate_weights` module implements functions to generate predetermined weights to quickly and easily generate spatial phase distributions. For a more advanced weights creation, see the :mod: `espm.weights.abundance` module.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from espm.conf import BASE_PATH
from espm.weights.abundance import Abundance
import hyperspy.api as hs


def toy_weights(**kwargs):
    r"""Load the toy problem weights.
    
    Returns
    -------
    Hdot : np.ndarray
        The weights of the toy problem.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> from espm.weights.generate_weights import toy_weights
        >>> import matplotlib.pyplot as plt
        >>> weights = toy_weights()
        >>> fig = plt.figure(figsize=(10,3))
        >>> axs = fig.subplots(1,3)
        >>> for i in range(3):
        ...     axs[i].imshow(weights[:,:,i], cmap=plt.cm.gray_r)
        ...     axs[i].set_title(f"Map {i+1}")
        >>> fig.show()

    """

    im1 = plt.imread(BASE_PATH / Path("datasets/toy-problem/phase1.png")).sum(axis=-1)
    im2 = plt.imread(BASE_PATH / Path("datasets/toy-problem/phase2.png")).sum(axis=-1)
    a = Abundance(im1.shape,3)
    a.add_image(im1,1,0.0,0.5)
    a.add_image(im2,2,0.0,0.5)

    return a.weights

def random_weights(shape_2d, n_phases=3, seed=0) :
    r""" Generates a random weight matrix with a uniform distribution.

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

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> from espm.weights.generate_weights import random_weights
        >>> import matplotlib.pyplot as plt
        >>> weights = random_weights((100,100))
        >>> fig = plt.figure(figsize=(10,3))
        >>> axs = fig.subplots(1,3)
        >>> for i in range(3):
        ...     axs[i].imshow(weights[:,:,i], cmap=plt.cm.gray_r)
        ...     axs[i].set_title(f"Map {i+1}")
        >>> fig.show()
    
    """

    a = Abundance(shape_2d, n_phases)
    for i in range(1,n_phases) :
        a.add_random(seed+i,i,0.0,1/n_phases)
    return a.weights
    

def laplacian_weights(shape_2d, n_phases=3, seed=0, size_x = 10, size_y = 10,**kwargs) :
    r""" Generates a random weight matrix with a laplacian distribution.

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

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from espm.weights.generate_weights import laplacian_weights
        >>> weights = laplacian_weights((100,100), 3, 0)
        >>> fig = plt.figure(figsize=(10,3))
        >>> axs = fig.subplots(1,3)
        >>> for i in range(3):
        ...     axs[i].imshow(weights[:,:,i], cmap=plt.cm.gray_r)
        ...     axs[i].set_title(f"Map {i+1}")
        >>> fig.show()

    """
    
    a = Abundance(shape_2d, n_phases)
    for i in range(1,n_phases) :
        a.add_laplacian(seed+i,i,0.0,1/n_phases, size_x = size_x, size_y=size_y)
    return a.weights

def gaussian_ripple_weights(shape_2d, width = 1, seed = 0, **kwargs) : 
    r"""
    Generate a weight matrix with a gaussian ripple of the given width and randomly centered.

    Parameters
    ----------
    shape_2d : tuple
        Shape of the weight matrix.
    width : int, optional
        Width of the gaussian ripple. The default is 1.
    seed : int, optional
        Seed for the random number generator. The default is 0. If seed is 0, the gaussian ripple is centered at half the second dimension of the weight matrix.
    
    Returns
    -------
    weights : numpy.ndarray
        Weight matrix with gaussian ripple distribution.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> from espm.weights.generate_weights import gaussian_ripple_weights
        >>> import matplotlib.pyplot as plt
        >>> weights = gaussian_ripple_weights((100,100), 20, 0)
        >>> fig = plt.figure(figsize=(5,2))
        >>> axs = fig.subplots(1,2)
        >>> for i in range(2):
        ...     axs[i].imshow(weights[:,:,i], cmap=plt.cm.gray_r)
        ...     axs[i].set_title(f"Map {i+1}")
        >>> fig.show()

    """
    a = Abundance(shape_2d, 2)
    if seed == 0 : 
        a.add_gaussian_ripple(center = shape_2d[1]//2, width = width, conc_max= 1, phase_id=1)
    else : 
        np.random.seed(seed)
        a.add_gaussian_ripple(center=np.random.randint(1,shape_2d[1]),width=width, conc_max=1, phase_id=1)

    return a.weights
    
    
def spheres_weights(shape_2d=[80, 80], n_phases=3,  seed=0, radius = 1, **kwargs):
    r"""
    Generate a weight matrix with randomly placed spheres of the given radius.

    Parameters
    ----------
    shape_2d : tuple, optional
        Shape of the weight matrix. The default is [80, 80].
    n_phases : int, optional
        Number of phases. The default is 3. The first phase is the complementary of the other phases.
    seed : int, optional
        Seed for the random number generator. The default is 0.
    radius : int, optional
        Radius of the spheres in pixels. The default is 1.
    
    Returns
    -------
    weights : numpy.ndarray
        Weight matrix with spheres distribution.
    
    Examples
    --------

    .. plot::
        :context: close-figs

        >>> from espm.weights.generate_weights import spheres_weights
        >>> import matplotlib.pyplot as plt
        >>> weights = spheres_weights((100,100), 3, 0, 20)
        >>> fig = plt.figure(figsize=(10,3))
        >>> axs = fig.subplots(1,3)
        >>> for i in range(3):
        ...     axs[i].imshow(weights[:,:,i], cmap=plt.cm.gray_r)
        ...     axs[i].set_title(f"Map {i+1}")
        >>> fig.show()

    """
    a = Abundance(shape_2d, n_phases)
    np.random.seed(seed)
    for i in range(1,n_phases) :
        a.add_sphere((np.random.randint(1,shape_2d[0]),np.random.randint(1,shape_2d[1])),radius,1/n_phases,i)
    return a.weights

def wedge_weights(shape_2d=[80, 80]):
    r"""
    Generate a weight matrix with a wedge of phase 2 and complementary phase 1.

    Parameters
    ----------
    shape_2d : tuple, optional
        Shape of the weight matrix. The default is [80, 80].
    
    Returns
    -------
    weights : numpy.ndarray
        Weight matrix with wedge distribution.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> from espm.weights.generate_weights import wedge_weights
        >>> import matplotlib.pyplot as plt
        >>> weights = wedge_weights((100,100))
        >>> fig = plt.figure(figsize=(5,2))
        >>> axs = fig.subplots(1,2)
        >>> for i in range(2):
        ...     axs[i].imshow(weights[:,:,i], cmap=plt.cm.gray_r)
        ...     axs[i].set_title(f"Map {i+1}")
        >>> fig.show()

    """
    a = Abundance(shape_2d, 2)
    a.add_wedge((0,0),shape_2d[0],shape_2d[1],0.0,1/2,1) 
        
    return a.weights

def chemical_map_weights(file = None, line_list = [], conc_list = [] , **kwargs):
    r"""
    Generate a weight matrix based on an experimental EDS spectrum image. For each selected line a map is generated with the corresponding concentration.

    Parameters
    ----------
    shape_2d : tuple, optional
        Shape of the weight matrix. The default is [80, 80].
    file : str, optional
        Path to the EDS spectrum image. If None, the function does nothing.
    line_list : list, optional
        List of lines to be used. The default is []. If line_list is empty, the function does nothing. The lines must be in the form of a string, e.g. "Fe_Ka".
    conc_list : list, optional
        List of concentrations for each line. The default is []. If conc_list is empty, the concentration of each line is set to 1/len(line_list).
    
    Returns
    -------
    weights : numpy.ndarray
        Weight matrix with chemical map distribution.
    
    """
    if conc_list == [] :
        conc_list = [1/len(line_list)]*len(line_list)
    if file is None or line_list==[] :
        print("Please provide a file name and a list of X-ray emission lines. Nothing was done.")
    else : 
        shape_2d = hs.load(file).data.shape[:-1]
        a = Abundance(shape_2d, len(line_list)+1)
        for i in range(len(line_list)) :
            a.add_chemical_map(file,line_list[i],0.0,conc_list[i],sigma = 3,phase_id=1+i)

    return a.weights

def generate_weights(weight_type, shape_2d, n_phases=3, seed=0, **params):
    r"""
    Generate a weight matrix with the given type. Additional parameters can be passed to the function as keyword arguments.

    Parameters
    ----------
    weight_type : str
        Type of weight matrix. Accepted types : random, laplacian, sphere, gaussian_ripple, wedge, toy_problem, chemical_map.
    shape_2d : tuple
        Shape of the weight matrix.
    n_phases : int, optional
        Number of phases. The default is 3. The first phase is the complementary of the other phases.
    seed : int, optional
        Seed for the random number generator. The default is 0.
    **params : dict
        Additional parameters for the weight matrix generation. See the documentation of the corresponding function for more details.
    
    Returns
    -------
    weights : numpy.ndarray
        Weight matrix with the given type.

    """
    if weight_type=="random":
        return random_weights(shape_2d, n_phases, seed) 
    elif weight_type=="laplacian":
        return laplacian_weights(shape_2d, n_phases, seed,**params) 
    elif weight_type=="sphere":
        return spheres_weights(shape_2d, n_phases, seed, **params) 
    elif weight_type == "gaussian_ripple" : 
        return gaussian_ripple_weights(shape_2d = shape_2d, seed = seed,**params)
    elif weight_type=="wedge":
        return wedge_weights(shape_2d=shape_2d)
    elif weight_type=="toy_problem":
        return toy_weights()
    elif weight_type=="chemical_map":
        return chemical_map_weights(**params)
    else:
        raise ValueError("Wrong weight_type: {}. Accepted types : random, laplacian, sphere, gaussian_ripple, wedge, toy_problem, chemical_map".format(weight_type))
    
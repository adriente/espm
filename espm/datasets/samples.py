import numpy as np
from espm.models import ToyModel
from espm.weights import load_toy_weights

def create_toy_sample(L, C, n_poisson, seed=None):
    """Create a toy sample.

    Parameters
    ----------
    L : int
        Length of the phases.
    C : int
        Number of possible components in the phases.
    n_poisson : int
        Poisson parameter. The higher, the less noise.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    sample : dict
        Dictionary containing the sample.

    The dictionary contains the following keys:
    - model_parameters: dict
        Dictionary containing the model parameters.
    - misc_parameters: dict
        Dictionary containing the misc parameters. Default empty.
    - shape_2d: list
        List of length 2 containing the shape of the 2D images.
    - GW: np.array
        The marrix corresponding to the phases.
    - H: np.array
        The matrix corresponding to the weights.
    - X: np.array
        The matrix corresponding to the noisy data.
    - Xdot: np.array
        The matrix corresponding to the noiseless data.
    - G: np.array
        The matrix corresponding to the G matrix.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> import matplotlib.pyplot as plt
        >>> from espm.datasets import create_toy_sample
        >>> L, C, n_poisson = 200, 15, 400,
        >>> sample = create_toy_sample(L, C, n_poisson, seed=0)
        >>> Hdot = sample["H"]
        >>> GW = sample["GW"]
        >>> G = sample["G"]
        >>> X = sample["X"]
        >>> Xdot = sample["Xdot"]
        >>> shape_2d = sample["shape_2d"]
        >>> vmin, vmax = 0,1
        >>> cmap = plt.cm.gray_r
        >>> plt.figure(figsize=(10, 3))
        >>> for i, hdot in enumerate(Hdot):
        ...     plt.subplot(1,3,i+1)
        ...     plt.imshow(Hdot[i].reshape(shape_2d), cmap=cmap, vmin=vmin, vmax=vmax)
        ...     plt.axis("off")
        ...     plt.title(f"Map {i+1}")
        >>> plt.show()
        >>> plt.figure(figsize=(10, 3))
        >>> l = np.arange(0, 1, 1/L)
        >>> plt.plot(l, G[:,:3])
        >>> plt.title("Spectral response of each elements")
        >>> plt.show()


    """

    Hdot = load_toy_weights().transpose(2, 0, 1)
    shape_2d = Hdot.shape[1:]
    K = len(Hdot)
    model = ToyModel(L, C, K, seed=seed)
    model.generate_g_matr()
    model.generate_phases()
    G = model.G
    Ddot = model.phases
    Wdot = model.Wdot
    Hdot = Hdot.reshape([K, -1])
    Ydot = Ddot @ Hdot

    Y = 1/n_poisson * np.random.poisson(n_poisson * Ydot)
    

    # Build sample
    model_parameters = {}
    model_parameters["L"] = L
    model_parameters["C"] = C
    model_parameters["K"] = K
    model_parameters["seed"] = seed
    model_parameters["name"] = model.__class__.__name__
    sample = {}
    sample["model_parameters"] = model_parameters # default empty, dict
    sample["misc_parameters"] = {} # deault empty, dict
    sample["shape_2d"] = shape_2d # list of length 2
    sample["GW"] = Ddot # np.array
    sample["H"] = Hdot # np.array
    sample["X"] = Y # np.array
    sample["Xdot"] = Ydot
    sample["G"] = G

    return sample



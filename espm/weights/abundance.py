r"""
Weights creation class
----------------------

The :mod:`espm.weights.abundance` module implements the class :class:`Abundance` which is used to create the weights of the phases. The weights are stored in the attribute :attr:`weights` and are normalized to 1.
"""

import numpy as np
from espm.models.EDXS_function import gaussian
import scipy.ndimage as ndimage
from skimage.filters import threshold_otsu
import hyperspy.api as hs
from skimage.filters import median
from scipy.interpolate import RectBivariateSpline
# from scipy.interpolate import interp2d


class Abundance(object):
    
    def __init__(self, shape_2d, n_phases):
        self.shape_2d = shape_2d
        self.n_phases = n_phases
        self._weights = np.zeros([*shape_2d, n_phases])

    #####################
    # Utility functions #
    #####################

    @property
    def weights (self) : 
        if np.sum(self._weights[:,:,0]) == 0.0 : 
            self._weights[:,:,0] = 1 - np.sum(self._weights[:,:,0:], axis=2)
        return self._weights
    
    def check_add_weights(self, val, phase_id):
        r"""
        Check if the sum of the weights is below 1. If it is, add the new abundance. If not, print a message.

        Parameters
        ----------
        val : array
            Array of the new abundance.
        phase_id : int
            Index of the phase. It has to be between 1 and n_phases-1.
        
        Returns
        -------
        None.
        """
        s = self._weights.sum(axis=2)
        s += val
        test = s <= 1
        if np.all(test) :
            self._weights[:, :, phase_id] += val
        else : 
            print("The weights contain values above 1, adding the new abundance was aborted.")

    def scale_phase(self,values,conc_min,conc_max) : 
        r"""
        Scale the values of a phase between conc_min and conc_max. If the values are all zero, the function returns the values without scaling.

        Parameters
        ----------
        values : array
            Array of the abundance to scale.
        conc_min : float
            Minimum concentration of the phase. It has to be between 0.0 and 1.0.
        conc_max : float
            Maximum concentration of the phase. It has to be between 0.0 and 1.0.

        Returns
        -------
        scaled_values : array
            Array of the scaled abundance.
        """
        if np.max(values) == 0.0 and np.min(values) == 0.0 : 
            print('The abundance is zero everywhere, scaling was aborted.')
            return values
        else : 
            return (values - np.min(values))/(np.max(values) - np.min(values))*(conc_max - conc_min)+conc_min
        
    #################
    # Add functions #
    #################

    def add_wedge(self, ind_origin, length, width, conc_min, conc_max, phase_id):
        r"""
        Function to define a wedge abundance of a defined phase. The concentration is max at the top and min at the bottom.

        Parameters
        ----------
        ind_origin : tuple of integers
            Coordinates of the top left corner of the wedge. It has to be inside the current weights, i.e. ind_origin[0] + length <= self.shape_2d[0] and ind_origin[1] + width <= self.shape_2d[1].
        length : int
            Length of the wedge in pixels.
        width : int
            Width of the wedge in pixels.
        conc_min : float
            Minimum concentration of the wedge. It has to be between 0.0 and 1.0.
        conc_max : float
            Maximum concentration of the wedge. It has to be between 0.0 and 1.0.
        phase_id : int
            Index of the phase. It has to be between 1 and n_phases-1.
        
        Returns
        -------
        None.

        Examples
        --------
        >>> from espm.weights.abundance import Abundance
        >>> from matplotlib import pyplot as plt
        >>> shape_2d = (100,100)
        >>> n_phases = 2
        >>> wedge = Abundance(shape_2d, n_phases)
        >>> wedge.add_wedge((0,0), 50, 50, 0.0, 1.0, 1)
        >>> plt.imshow(wedge.weights[:,:,1])
        """
        assert phase_id != 0, "The phase_id cannot be 0, it has to be between 1 and n_phases-1."
        if (ind_origin[0] + length <= self.shape_2d[0]) or (
            ind_origin[1] + width <= self.shape_2d[1]
        ):
            # Constructs the wedge in 2D
            wedge = np.linspace(
                (np.linspace(0, 1, num=width)),
                (np.linspace(0, 1, num=width)),
                num=length,
            )
            val = np.zeros(self.shape_2d)
            val[
                ind_origin[0] : ind_origin[0] + length,
                ind_origin[1] : ind_origin[1] + width,
            ] = wedge
        else : 
            print('The wedge is at least partially outside the current weights, please choose other top left coordinates.')
            
        scaled_values = self.scale_phase(val,conc_min,conc_max)
        self.check_add_weights(scaled_values, phase_id)

   

    def add_sphere(self, ind_origin, radius, conc_max, phase_id, asym_x=1.0, asym_y=1.0):
        r"""
        Function to define a sphere abundance of a defined phase. The concentration is max at centre and 0.0 on the edges.

        Parameters
        ----------
        ind_origin : tuple of integers
            Coordinates of the centre of the sphere.
        radius : float 
            Radius of the sphere in pixels.
        conc_max : float   
            Maximum concentration of the sphere.
        phase_id : int
            Index of the phase
        asym_x : float, optional
            Asymmetry of the sphere in the x direction. The default is 1.0.
        asym_y : float, optional
            Asymmetry of the sphere in the y direction. The default is 1.0.
        
        Returns
        -------
        None.

        Examples
        --------
        >>> from espm.weights.abundance import Abundance
        >>> from matplotlib import pyplot as plt
        >>> shape_2d = (100,100)
        >>> n_phases = 2
        >>> sphere = Abundance(shape_2d, n_phases)
        >>> sphere.add_sphere((50,50), 30, 1.0, 1)
        >>> plt.imshow(sphere.weights[:,:,1])

        """
        assert phase_id != 0, "The phase_id cannot be 0, it has to be between 1 and n_phases-1."
        xx, yy = np.mgrid[: self.shape_2d[0], : self.shape_2d[1]]

        sq_sphere = (radius**2 - ((xx - ind_origin[0])/asym_x)**2 - ((yy - ind_origin[1])/asym_y)**2)
        mask = sq_sphere > 0
        sphere = np.zeros(self.shape_2d)
        sphere[mask] = 2*np.sqrt(sq_sphere[mask])
        scaled_sphere = self.scale_phase(sphere,0,conc_max)
        self.check_add_weights(scaled_sphere, phase_id)
    
    def add_gaussian_ripple(self, center, width, conc_max, phase_id) :
        r"""
        Function to define a gaussian ripple spanning over the whole length of the weights abundance of a defined phase. The concentration is max at the center and 0.0 on the edges.

        Parameters
        ----------
        center : int
            Position on the mean of the gaussian ripple in pixels.
        width : int
            Full width at half maximum of the gaussian in pixels.
        conc_max : float
            Maximum concentration of the gaussian ripple. It has to be between 0.0 and 1.0.
        phase_id : int
            Index of the phase. It has to be between 1 and n_phases-1.
        
        Returns
        -------
        None.

        Examples
        --------
        >>> from espm.weights.abundance import Abundance
        >>> from matplotlib import pyplot as plt
        >>> shape_2d = (100,100)
        >>> n_phases = 2
        >>> gaussian_ripple = Abundance(shape_2d, n_phases)
        >>> gaussian_ripple.add_gaussian_ripple(50, 15, 1.0, 1)
        >>> plt.imshow(gaussian_ripple.weights[:,:,1])

        """
        assert phase_id != 0, "The phase_id cannot be 0, it has to be between 1 and n_phases-1."
        x = np.arange(self.shape_2d[1])
        gauss_line = gaussian(x,center,width/2.355)
        gaussian_ripple = np.tile(gauss_line,(self.shape_2d[0],1))
        scaled_gaussian_ripple = self.scale_phase(gaussian_ripple,0,conc_max)   
        self.check_add_weights(scaled_gaussian_ripple, phase_id)

    def add_laplacian(self,seed, phase_id, conc_min, conc_max,size_x = 50, size_y = 50) : 
        r"""
        Function to generate smooth noise. The characteristic length scale of the noise variations is defined by the scale_x and scale_y parameters. 

        Parameters
        ----------
        seed : int
            Seed for the random number generator.
        phase_id : int
            Index of the phase. It has to be between 1 and n_phases-1.
        conc_min : float
            Minimum concentration. It has to be between 0.0 and 1.0.
        conc_max : float
            Maximum concentration. It has to be between 0.0 and 1.0.
        size_x : int, optional
            characteristic length scale of the noise variations in the x direction. The default is 50.
        size_y : int, optional
            characteristic length scale of the noise variations in the y direction. The default is 50.
        
        Returns
        -------
        None.

        Examples
        --------
        >>> from espm.weights.abundance import Abundance
        >>> from matplotlib import pyplot as plt
        >>> shape_2d = (100,100)
        >>> n_phases = 2
        >>> laplacian = Abundance(shape_2d, n_phases)
        >>> laplacian.add_laplacian(1, 1, 0.0, 1.0, 10, 10)
        >>> plt.imshow(laplacian.weights[:,:,1])

        """
        assert phase_id != 0, "The phase_id cannot be 0, it has to be between 1 and n_phases-1."
        np.random.seed(seed)
        rnd = np.random.rand(size_x,size_y)
        lapl = median(median(rnd))
        # f = interp2d(np.arange(size_x), np.arange(size_y), lapl, kind='cubic')
        f = RectBivariateSpline(np.arange(size_x), np.arange(size_y), lapl.T)
        # For some dumb reason, the interpolation function has to have the coordinates in the opposite order
        res = f(np.linspace(0,size_y,num = self.shape_2d[1]),np.linspace(0,size_x,num = self.shape_2d[0])).T

        scaled_res = self.scale_phase(res,conc_min,conc_max)
        self.check_add_weights(scaled_res, phase_id)

    def add_random(self,seed, phase_id, conc_min, conc_max) : 
        r"""
        Function to generate random noise.

        Parameters
        ----------
        seed : int
            Seed for the random number generator.
        phase_id : int 
            Index of the phase. It has to be between 1 and n_phases-1.
        conc_min : float
            Minimum concentration. It has to be between 0.0 and 1.0.
        conc_max : float
            Maximum concentration. It has to be between 0.0 and 1.0.
        
        Returns
        -------
        None.

        Examples
        --------
        >>> from espm.weights.abundance import Abundance
        >>> from matplotlib import pyplot as plt
        >>> shape_2d = (100,100)
        >>> n_phases = 2
        >>> random = Abundance(shape_2d, n_phases)
        >>> random.add_random(1, 1, 0.0, 1.0)
        >>> plt.imshow(random.weights[:,:,1])

        """
        assert phase_id != 0, "The phase_id cannot be 0, it has to be between 1 and n_phases-1."
        np.random.seed(seed)
        rnd = np.random.rand(self.shape_2d[0],self.shape_2d[1])
        scaled_rnd = self.scale_phase(rnd,conc_min,conc_max)
        self.check_add_weights(scaled_rnd, phase_id)

    def add_image(self, image, phase_id, conc_min, conc_max) : 
        r"""
        Function to add a 2D numpy array as a phase. 

        Parameters
        ----------
        image : numpy.ndarray
            2D numpy array with the phase. It has to be the same size as the shape_2d parameter.
        phase_id : int
            Index of the phase. It has to be between 1 and n_phases-1.
        conc_min : float
            Minimum concentration. It has to be between 0.0 and 1.0.
        conc_max : float
            Maximum concentration. It has to be between 0.0 and 1.0.

        Returns
        -------
        None.

        Examples
        --------
        >>> from espm.weights.abundance import Abundance
        >>> import hyperspy.api as hs
        >>> from matplotlib import pyplot as plt
        >>> shape_2d = (512,512)
        >>> n_phases = 2
        >>> image = Abundance(shape_2d, n_phases)
        >>> data = hs.datasets.example_signals.object_hologram()
        >>> image.add_image(data.data, 1, 0.0, 1.0)
        >>> plt.imshow(image.weights[:,:,1])
        """
        assert phase_id != 0, "The phase_id cannot be 0, it has to be between 1 and n_phases-1."
        scaled_image = self.scale_phase(image,conc_min,conc_max)
        self.check_add_weights(scaled_image, phase_id)

    def add_chemical_map(self, file, element_line, conc_min,conc_max, sigma, phase_id, **kwargs) :
        r"""
        Function to add a chemical map extracted from a EDS spectrum image as a phase. A threshold and blurring are automatically applied to the map to suppress noise.

        Parameters
        ----------
        file : str
            Path to the EDS spectrum image.
        element_line : str
            Element line to be used for the chemical map. The corresponding element has to be in the metadata of the spectrum image. For example, if the element is 'Cu' and the line is 'K', the element_line parameter should be 'Cu_K'.
        conc_min : float
            Minimum concentration. It has to be between 0.0 and 1.0.
        conc_max : float
            Maximum concentration. It has to be between 0.0 and 1.0.
        sigma : float
            Sigma parameter for the Gaussian filter. It has to be greater than 0.0. Note that the automatic threshold is applied to the blurred map.
        phase_id : int
            Index of the phase. It has to be between 1 and n_phases-1.
        **kwargs :
            Keyword arguments for the get_lines_intensity method of the EDS spectrum image.
        
        Returns
        -------
        None.

        """
        assert phase_id != 0, "The phase_id cannot be 0, it has to be between 1 and n_phases-1."
        spim = hs.load(str(file))
        map = spim.get_lines_intensity([element_line], **kwargs)
        blur = ndimage.gaussian_filter(map[0].data, sigma=sigma, order=0)
        
        thresh = threshold_otsu(blur)
        mask = np.where(blur > thresh, blur, np.zeros_like(blur))

        mask[mask>0.0] = self.scale_phase(mask[mask>0.0],conc_min,conc_max)
        self.check_add_weights(mask, phase_id)

################################
# Old version of chemical maps #
################################

# def chemical_maps_weights(file, element_lines, conc_max, sigma = 4, **kwargs) : 
#     spim = hs.load(str(file))
#     maps = spim.get_lines_intensity(element_lines, **kwargs)
#     blurs = []
#     for map in maps : 
#         blurs.append(ndimage.gaussian_filter(map.data, sigma=sigma, order=0))

#     masks = []
#     for blur in blurs : 
#         thresh = threshold_otsu(blur)
#         masks.append(np.where(blur > thresh, blur, np.zeros_like(blur)))
        
#     for mask in masks : 
#         ind_0 = np.where(mask > 0)
#         min_mask = mask[ind_0] - np.min(mask[ind_0])
#         norm_mask = min_mask / ((1/conc_max)* np.max(min_mask))
#         mask[ind_0] = norm_mask

#     complementary = 1 - (np.sum(np.stack(masks),axis = 0))
#     masks.append(complementary)
#     weights = np.moveaxis(np.stack(masks),0,2)

#     return weights

#####################################
# Old version of the sphere weights #
#####################################

# def add_sphere(self, ind_origin, radius_x, radius_y, conc_min, conc_max, phase_id):
#     r"""
#     Function to define a sphere of a defined phase (from phases). The concentration is max at centre and min on the edges. The "spheres" can be defined to be elliptic
#     Inputs :
#         ind_origin : center of the sphere (tuple of integers)
#         radius_x and radius_y : x and y radius of the spheres. They depend on conc_min and conc_max and therefore do not represent a  (floats)
#         conc_min and conc_max : min and max concentration of the sphere (floats between 0.0 and 1.0)
#         phase_id : index of the phase in self.phases (integer)
#     """
#     # Defines a cartesian grid in 2D
#     xx, yy = np.mgrid[: self.shape_2d[0], : self.shape_2d[1]]

#     # Selects the area in which the concentration is above conc_min
#     calc_circle = (
#         conc_max
#         - ((xx - ind_origin[0]) / (10 * radius_x)) ** 2
#         - ((yy - ind_origin[1]) / (10 * radius_y)) ** 2
#     )
#     mask = calc_circle > conc_min
#     circle = mask * calc_circle

#     if self.check_add_weights(circle):
#         # Adds the sphere to the weights
#         self.weights[:, :, phase_id] += circle
#     else:
#         print("the phases concentrations add up to more than one")


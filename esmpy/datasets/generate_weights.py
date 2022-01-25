import numpy as np
from skimage.filters import median
from esmpy.models.EDXS_function import gaussian

class Material(object):
    
    def __init__(self, shape_2d, n_phases):
        self.shape_2d = shape_2d
        self.n_phases = n_phases
        self.weights = np.zeros([*shape_2d, n_phases])

    
    def wedge(self, ind_origin, length, width, conc_min, conc_max, phase_id):
        """
        Function to define a wedge of a defined phase (from phases). The concentration in the wedge range from conc_min to conc_max. The wedge start from the top left corner and the concentration ramps from left to right only.
        Inputs :
            ind_origin : top left corner (tuple of integers)
            lenght : length of the wedge in pixels (integer)
            width : width of the wedge in pixels (integer)
            conc_min and conc_max : min and max concentration of the wedge (floats between 0.0 and 1.0)
            phase_id : index of the phase in self.phases (integer)
        """
        if (ind_origin[0] + length < self.shape_2d[0]) or (
            ind_origin[1] + width < self.shape_2d[1]
        ):
            # Constructs the wedge in 2D
            wedge = np.linspace(
                (np.linspace(conc_min, conc_max, num=width)),
                (np.linspace(conc_min, conc_max, num=width)),
                num=length,
            )

            val = np.zeros(self.shape_2d)
            val[
                ind_origin[0] : ind_origin[0] + length,
                ind_origin[1] : ind_origin[1] + width,
            ] = wedge
            

        if self.check_add_weights(val):
            # Adds the wedge to the weights
            self.weights[:, :, phase_id] += val
        else:
            print("Wedge has a too large width or length")
        # A slab is a wedge with constant concentration

    def sphere(self, ind_origin, radius_x, radius_y, conc_min, conc_max, phase_id):
        """
        Function to define a sphere of a defined phase (from phases). The concentration is max at centre and min on the edges. The "spheres" can be defined to be elliptic
        Inputs :
            ind_origin : center of the sphere (tuple of integers)
            radius_x and radius_y : x and y radius of the spheres. They depend on conc_min and conc_max and therefore do not represent a  (floats)
            conc_min and conc_max : min and max concentration of the sphere (floats between 0.0 and 1.0)
            phase_id : index of the phase in self.phases (integer)
        """
        # Defines a cartesian grid in 2D
        xx, yy = np.mgrid[: self.shape_2d[0], : self.shape_2d[1]]

        # Selects the area in which the concentration is above conc_min
        calc_circle = (
            conc_max
            - ((xx - ind_origin[0]) / (10 * radius_x)) ** 2
            - ((yy - ind_origin[1]) / (10 * radius_y)) ** 2
        )
        mask = calc_circle > conc_min
        circle = mask * calc_circle

        if self.check_add_weights(circle):
            # Adds the sphere to the weights
            self.weights[:, :, phase_id] += circle
        else:
            print("the phases concentrations add up to more than one")

    def gaussian_ripple(self, center, width, conc_max, phase_id) :
        x = np.arange(self.shape_2d[1])
        gauss_line = gaussian(x,center,width/2.355)
        norm_gauss = conc_max*(gauss_line - np.min(gauss_line))/(np.max(gauss_line) - np.min(gauss_line))
        gaussian_ripple = np.tile(norm_gauss,(self.shape_2d[0],1))
        if self.check_add_weights(gaussian_ripple) : 
            self.weights[:,:,phase_id] += gaussian_ripple
        else : 
            print("the phases concentrations add up to more than one")

            
    def check_add_weights(self, val):
        # Construct the sum of weigths of each phase to evaluate later if they add up to more than one
        s = self.weights.sum(axis=2)
        s += val
        test = s <= 1
        return np.all(test)
    
    def finalize_weight(self):
        self.weights[:,:,0] = 1 - np.sum(self.weights[:,:,0:], axis=2)
        return self.weights
    

def random_weights(shape_2d, n_phases=3, seed=0) :
    np.random.seed(seed)
    rnd_array = np.random.rand(shape_2d[0], shape_2d[1], n_phases)
    weights = rnd_array/np.sum(rnd_array, axis=2, keepdims=True)
    return weights
    
def laplacian_weights(shape_2d, n_phases=3, seed=0) :
    np.random.seed(seed)
    rnd_array = np.random.rand(shape_2d[0], shape_2d[1], n_phases)
    rnd_f = []
    for i in range(rnd_array.shape[2]):
        rnd_f.append(median(median(rnd_array[:,:,i])))
    rnd_f = np.array(rnd_f).transpose([1,2,0])
    weights = rnd_f/np.sum(rnd_f, axis=2, keepdims=True)
    return weights

def gaussian_ripple_weights(shape_2d, width = 1, seed = 0, **kwargs) : 
    mat = Material(shape_2d, 2)
    np.random.seed(seed)
    if seed == 0 : 
        mat.gaussian_ripple(center = shape_2d[1]//2, width = width, conc_max= 1, phase_id=1)
    else : 
        c = np.random.randint(1,shape_2d[1])
        mat.gaussian_ripple(center = c, width = width, conc_max= 1, phase_id= 1)

    return mat.finalize_weight()
    
    
def spheres_weights(shape_2d=[80, 80], n_phases=3,  seed=0, radius = 2.5, **kwargs):
    mat = Material(shape_2d, n_phases)
    
    intersect = 10*radius*np.sqrt(2)/2

    assert shape_2d[0] > intersect
    assert shape_2d[1] > intersect
    if seed == 0 and n_phases==3 and shape_2d == [80, 80]:
        mat.sphere((25, 30), 3.5, 3.5, 0.0, 0.5, 1)
        mat.sphere((55, 30), 3.5, 3.5, 0.0, 0.5, 2)
    else:
        np.random.seed(seed)
        for i in range(1, n_phases):
            p1 = np.random.randint(int(intersect), int(shape_2d[0]- intersect))
            p2 = np.random.randint(int(intersect), int(shape_2d[1]- intersect))
            mat.sphere([p1,p2], radius, radius, 0.0, 0.5, i)     
        
    return mat.finalize_weight()

def generate_weights(weight_type, shape_2d, n_phases=3, seed=0, **params):
    if weight_type=="random":
        return random_weights(shape_2d, n_phases, seed) 
    elif weight_type=="laplacian":
        return laplacian_weights(shape_2d, n_phases, seed) 
    elif weight_type=="sphere":
        return spheres_weights(shape_2d, n_phases, seed, **params) 
    elif weight_type == "gaussian_ripple" : 
        return gaussian_ripple_weights(shape_2d = shape_2d, seed = seed , **params)
    # elif weight_type=="gradient":
    #     return spheres_gradient(shape_2d, n_phases, seed) 
    else:
        raise ValueError("Wrong weight_type: {}".format(weight_type))
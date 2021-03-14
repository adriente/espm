import numpy as np
import json
import os
import hyperspy.api as hs


class ArtificialSpim:
    """
    This class is used to generate artificial spectrum image. It corresponds to the linear mixing of several spectroscopic signatures (phases) through spatial overlap.
    Using this class is done in 3 steps :
        1) Define the spatial distribution of the different phases using wedge, sphere or input images. The sum of concentration in each pixel shall not exceed one.
        2) Generate the corresponding noiseless spectrum image (generated_spim). The first phase can be set as a matrix to fill the part where there is no other phases.
        3) Generate a spectrum image (stochastic_spim) by randomly choosing N times in the spectra of the noiseless spectrum image (generated_spim). For each pixel the value of N is determined randomly following a Poisson distribtuion weighted by the local density.

    The spectra of the generated spim are normalized so that it mimics a probability distribution.
    """

    def __init__(self, phases, densities, weights):
        """
        Inputs :
        phases : It shoud be a (nb_phases,nb_spectral_channels) shape array or a list of array with (nb_spectral_channels,) shape. It corresponds to the spectra of the different phases. The bremsstrahlung (or background) signal should be included in phases.
        shape_2D : A tuple or a list containing 2 integers such as (nb_px_x,nb_px_y)
        """
        # I don't use phases as Gaussians instances within this class for generality purposes.
        # Initialisation of the relevant quantities
        
        # Normalize the phases
        self.shape_2D = weights.shape[:2]
        
        self.phases = phases / np.sum(phases, axis=1, keepdims=True)
        self.densities = densities
        self.spectral_len = phases.shape[1]
        self.num_phases = phases.shape[0]

        assert(self.num_phases == weights.shape[2])
        
        # The weights correspond to the spatial distribution of the different phases.
        self.weights = weights
        
        np.testing.assert_allclose(np.sum(self.weights, axis=2), 1)
        
        # The density map correspond to the spatial distribution of the density
        self.density_map = None

        # Stored objects
        self.generated_spim = None
        self.stochastic_spim = None
        self.continuous_spim = None
        self.N = None



    # def intensity_map(self, ind_origin, image, conc_max, phase_id):
    #     """
    #     Function to use an image as a distribution of one of the phases. The image is normalized so that the max intensity corresponds to conc_max. The image top left corner is positionned at ind_origin. If the image is too large or wide, it's truncated to fit in shape_2D.
    #     Inputs :
    #         ind_origin : top left corner of the image (tuple of integers)
    #         conc_max : max concentration of the image (floats between 0.0 and 1.0)
    #         phase_id : index of the phase in self.phases (integer)
    #     Note : In this version of the code the concentration goes from 0 to conc_max. It could be changed in the future to go from conc_min to conc_max.
    #     """
    #     canvas = np.zeros(self.shape_2D)
    #     # Normalisation of the image
    #     n_image = image / np.max(image) * conc_max

    #     # Check whether shape_2D is larger or wider than the image
    #     # 4 case are possible : image larger than canvas on both axes, image smaller than canvas on both axes, image larger than canvas on axis 0 (and a another case for axis 1)
    #     if self.shape_2D[0] > (image.shape[0] + ind_origin[0]) and self.shape_2D[1] > (
    #         image.shape[1] + ind_origin[1]
    #     ):
    #         # No truncation
    #         canvas[
    #             ind_origin[0] : image.shape[0] + ind_origin[0],
    #             ind_origin[1] : image.shape[1] + ind_origin[1],
    #         ] = n_image
    #     elif self.shape_2D[0] < (image.shape[0] + ind_origin[0]) and self.shape_2D[
    #         1
    #     ] > (image.shape[1] + ind_origin[1]):
    #         # axis 0 truncation
    #         canvas[
    #             ind_origin[0] :, ind_origin[1] : image.shape[1] + ind_origin[1]
    #         ] = n_image[: self.shape_2D[0] - ind_origin[0], :]
    #     elif self.shape_2D[0] > (image.shape[0] + ind_origin[0]) and self.shape_2D[
    #         1
    #     ] < (image.shape[1] + ind_origin[1]):
    #         # axis 1 truncation
    #         canvas[
    #             ind_origin[0] : image.shape[0] + ind_origin[0], ind_origin[1] :
    #         ] = n_image[:, : self.shape_2D[1] - ind_origin[1]]
    #     else:
    #         # Truncation
    #         canvas[ind_origin[0] :, ind_origin[1] :] = n_image[
    #             : self.shape_2D[0] - ind_origin[0], : self.shape_2D[1] - ind_origin[1]
    #         ]

    #     # Construct the sum of weigths of each phase to evaluate later if they add up to more than one
    #     sum = self.weights.sum(axis=2)
    #     sum += canvas
    #     test = sum < 1

    #     if np.all(test):
    #         # Adds the image to the weights
    #         self.weights[:, :, phase_id] = canvas
    #         return self.weights[:, :, phase_id]
    #     else:
    #         print("the phases concentrations add up to more than one")



    def flatten_weights (self) :
        # Nati: This is ok because of the transpose...
        return self.weights.T.reshape(self.num_phases,self.shape_2D[0]*self.shape_2D[1])

    def flatten_Xdot (self) :
        return self.continuous_spim.T.reshape(self.spectral_len, self.shape_2D[0]*self.shape_2D[1])

    def flatten_X (self) :
        return self.stochastic_spim.T.reshape(self.spectral_len, self.shape_2D[0]*self.shape_2D[1])

    def flatten_gen_spim (self) :
        return self.generated_spim.T.reshape(self.spectral_len, self.shape_2D[0]*self.shape_2D[1])


    def generate_spim_deterministic(self):
        """
        Function to generate an ideal spectrum image based on weights and phases. The different phases
        are linearly mixed according to the local sum of weights. The first phase can be set as the
        matrix. With that option the concentration sums to one in every pixel.
        """
        # TODO: @Adrien Do we need this function? We might just delete it
        
        self.generated_spim = (
            self.weights.reshape(-1, self.weights.shape[-1]) @ self.phases
        ).reshape(*self.shape_2D, -1)

    ###################################################
    # Quick and dirty way to add gaussian noise begin #
    ###################################################

    # def generate_spim_gaussian (self, sigma, matrix=True,clip=True) :
    #     self.generated_spim*=200
    #     gauss_spim=np.random.normal(0,sigma,self.generated_spim.shape)+self.generated_spim
    #     if clip :
    #         self.stochastic_spim=gauss_spim.clip(min=0)
    #     else :
    #         self.stochastic_spim=gauss_spim

    #################################################
    # Quick and dirty way to add gaussian noise end #
    #################################################

    def generate_spim_stochastic(self, N, seed=0, old=False):
        """
        Function to generate a noisy spectrum image based on an ideal one. For each pixel,
        local_N random spectroscopic events are drown from the probabilities given by the
        ideal spectra. local_N is drawn from a poisson distribution with average N weighted
        by the local density. The density of the matrix is set similarly as for
        generate_spim_deterministic.

        Inputs :
            N : (integer) average number of events
            seed : (integer) the seed for reproducible result. Default: 0.
            old : (boolean) use the old way to generate data. Default: False.
        """
        # Set the seed
        np.random.seed(seed)
        self.N = N
        self.generate_spim_deterministic()
        # if old:

        #     self.density_map = np.sum(
        #         self.weights * np.expand_dims(self.densities, axis=(0, 1)), axis=2
        #     )

        #     self.continuous_spim = (
        #         N * self.generated_spim * np.expand_dims(self.density_map, axis=2)
        #     )

        #     self.stochastic_spim = np.zeros([*self.shape_2D, self.spectral_len])

        #     # generating the spectroscopic events
        #     for i in range(self.shape_2D[0]):
        #         for j in range(self.shape_2D[1]):
        #             # Draw a local_N based on the local density
        #             local_N = np.random.poisson(N * self.density_map[i, j])
        #             # draw local_N events from the ideal spectrum
        #             counts = np.random.choice(
        #                 self.spectral_len, local_N, p=self.generated_spim[i, j]
        #             )
        #             # Generate the spectrum based on the drawn events
        #             hist = np.bincount(counts, minlength=self.spectral_len)
        #             self.stochastic_spim[i, j] = hist
        # else:

        # n D W A
        self.continuous_spim = N * (
            self.weights.reshape(-1, self.weights.shape[-1])
            @ (self.phases * np.expand_dims(self.densities, axis=1))
        ).reshape(*self.shape_2D, -1)

        self.stochastic_spim = np.zeros([*self.shape_2D, self.spectral_len])
        for k, w in enumerate(self.densities):
            # generating the spectroscopic events
            for i in range(self.shape_2D[0]):
                for j in range(self.shape_2D[1]):
                    # Draw a local_N based on the local density
                    local_N = np.random.poisson(N * w * self.weights[i, j, k])
                    # draw local_N events from the ideal spectrum
                    counts = np.random.choice(
                        self.spectral_len, local_N, p=self.phases[k]
                    )
                    # Generate the spectrum based on the drawn events
                    hist = np.bincount(counts, minlength=self.spectral_len)
                    self.stochastic_spim[i, j] += hist

    # def map_phase(self,phase_id,matrix=True) :
    #     #matrix boolean should change to self.matrix
    #     """
    #     Function to access the spatial distribution of the chosen phase.
    #     Input :
    #         phase_id : index of the phase in self.phases (integer)
    #     """
    #     if (matrix) and (phase_id==0) :
    #         sum=self.weights.sum(axis=2)
    #         matrix_weight=1-sum
    #         return matrix_weight
    #     else :
    #         return self.weights[:,:,phase_id]

    def save(self, filename):
        """
        Function to save the noisy simulated spectrum image as well as the associated spectra and spatial distribution of the phases.
        Input :
            filename : name of the file you want to save (string)
        Note : You might want to change the scale and offset to suit your needs
        """
        d = {}  # dictionary of everything we would like to save

        # signal=hs.signals.Signal1D(self.stochastic_spim)
        # signal.axes_manager[2].name="Energy"
        # signal.axes_manager[2].scale=0.01
        # signal.axes_manager[2].offset=0.20805000000000007
        # signal.axes_manager[2].unit="keV"
        # d["signal"] = signal

        # signal.save(filename,extension="hspy")
        d["X"] = self.stochastic_spim
        d["X_flat"] = self.flatten_X()
        d["Xdot"] = self.continuous_spim
        d["phases"] = self.phases
        d["densities"] = self.densities
        d["weights"] = self.weights
        d["flat_weights"] = self.flatten_weights()
        d["N"] = self.N
        np.savez(filename, **d)
        # for i in range(len(self.phases)) :
        #     np.save(filename+"map_p{}".format(i),self.map_phase(i))
        #     np.savetxt(filename+"spectrum_p{}".format(i),self.phases[i].T)

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

    def __init__(self, phases, densities, weights, G=None):
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
        self.G = G

    def flatten_weights (self) :
        return self.weights.transpose([2,0,1]).reshape(self.num_phases,self.shape_2D[0]*self.shape_2D[1])

    def flatten_Xdot (self) :
        return self.continuous_spim.transpose([2,0,1]).reshape(self.spectral_len, self.shape_2D[0]*self.shape_2D[1])

    def flatten_X (self) :
        return self.stochastic_spim.transpose([2,0,1]).reshape(self.spectral_len, self.shape_2D[0]*self.shape_2D[1])

    def flatten_gen_spim (self) :
        return self.generated_spim.transpose([2,0,1]).reshape(self.spectral_len, self.shape_2D[0]*self.shape_2D[1])


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


    def generate_spim_stochastic(self, N, seed=0):
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
        # d["X_flat"] = self.flatten_X()
        d["Xdot"] = self.continuous_spim
        d["phases"] = self.phases
        d["densities"] = self.densities
        d["weights"] = self.weights
        # d["flat_weights"] = self.flatten_weights()
        d["N"] = self.N
        if self.G is not None:
            d["G"] = self.G
        np.savez(filename, **d)
        # for i in range(len(self.phases)) :
        #     np.save(filename+"map_p{}".format(i),self.map_phase(i))
        #     np.savetxt(filename+"spectrum_p{}".format(i),self.phases[i].T)
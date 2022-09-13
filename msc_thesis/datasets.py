import math
import numpy as np
import matplotlib.pyplot as plt

# Obtain dataset cropped from spim 
def get_experimental_dataset(spim, crop_size):
    P1, P2 = int(math.sqrt(spim.X.shape[1])),int(math.sqrt(spim.X.shape[1]))
    YVol = spim.X.reshape(-1,P1,P2)
    Dtilde = spim.phases
    Htilde = spim.maps
    K = Htilde.shape[0]
    Htilde_vol = Htilde.reshape((K, P1,P2))

    # Crop Maps
    Htilde_vol = Htilde_vol[:,:crop_size[0],:crop_size[1]]
    Htilde = Htilde_vol.reshape((K,crop_size[0]*crop_size[1]))

    # Crop Y
    YVol = YVol[:,:crop_size[0],:crop_size[1]]

    G = spim.build_G("bremsstrahlung", norm = True)()

    return Dtilde, Htilde, Htilde_vol, YVol, G, K


def syntheticG(L=200, C=15, seed=None):

    np.random.seed(seed=seed)
    n_el = 45
    n_gauss = np.random.randint(2, 5,[C])
    l = np.arange(0, 1, 1/L)
    mu_gauss = np.random.rand(n_el)
    sigma_gauss = 1/n_el + np.abs(np.random.randn(n_el))/n_el/5

    G = np.zeros([L,C])

    def gauss(x, mu, sigma):
        # return np.exp(-(x-mu)**2/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
        return np.exp(-(x-mu)**2/(2*sigma**2))

    for i, c in enumerate(n_gauss):
        inds = np.random.choice(n_el, size=[c] , replace=False)
        for ind in inds:
            w = 0.1+0.9*np.random.rand()
            G[:,i] += w * gauss(l, mu_gauss[ind], sigma_gauss[ind])
    return G
 
def load_toy_images():
    im1 = plt.imread("../../esmpy/datasets/toy-problem/phase1.png")
    im1 = (1-np.mean(im1, axis=2)) *0.5

    im2 = plt.imread("../../esmpy/datasets/toy-problem/phase2.png")
    im2 = (1-np.mean(im2, axis=2)) *0.5

    im0 = 1 - im1 - im2 

    Htilde = np.array([im0, im1, im2])

    return Htilde


def get_toy_dataset(L, C, n_poisson, seed=None, seedNoise=None):
    np.random.seed(seed=seed)
    G = syntheticG(L,C, seed=seed) 
    Htilde_vol = load_toy_images()
    K = len(Htilde_vol)
    Htilde = Htilde_vol.reshape(K, -1)
    Wdot = np.abs(np.random.laplace(size=[C, K]))
    Wdot = Wdot / np.mean(Wdot)/L
    Dtilde = (G @ Wdot)*n_poisson
    Ydot = (Dtilde) @ Htilde
    
    np.random.seed(seed=seedNoise)
    Y = np.random.poisson(Ydot)
    Y_vol = Y.reshape(L, Htilde_vol.shape[1], Htilde_vol.shape[2])
    return G, Wdot, Dtilde, Htilde_vol, Htilde, Ydot, Y_vol, Y, K


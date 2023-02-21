import numpy as np
import matplotlib.pyplot as plt
from esmpy.conf import BASE_PATH
from pathlib import Path

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
    im1 = plt.imread(BASE_PATH / Path("datasets/toy-problem/phase1.png"))
    im1 = (1-np.mean(im1, axis=2)) *0.5

    im2 = plt.imread(BASE_PATH / Path("datasets/toy-problem/phase2.png"))
    im2 = (1-np.mean(im2, axis=2)) *0.5

    im0 = 1 - im1 - im2 

    Hdot = np.array([im0, im1, im2])

    return Hdot


def create_toy_problem(L, C, n_poisson, seed=None):
    np.random.seed(seed=seed)
    G = syntheticG(L,C, seed=seed)
    Hdot = load_toy_images()
    K = len(Hdot)
    Hdotflat = Hdot.reshape(K, -1)
    Wdot = np.abs(np.random.laplace(size=[C, K]))
    Wdot = Wdot / np.mean(Wdot)/L
    Ddot = G @ Wdot
    Ydot = Ddot @ Hdotflat

    Y = 1/n_poisson * np.random.poisson(n_poisson * Ydot)
    shape_2d = Hdot.shape[1:]


    return G, Wdot, Ddot, Hdot, Hdotflat, Ydot, Y, shape_2d, K


def data2hysample(out):


    return ...
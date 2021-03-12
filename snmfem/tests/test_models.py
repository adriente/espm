import numpy as np
from snmfem.models import Toy
from snmfem.conf import seed_max
from snmfem.generate_data import ArtificialSpim
from snmfem.generate_weights import random_weights, laplacian_weights

def test_toy_model() : 
    seed = np.random.randint(seed_max)
    pars_dict = {"c" : 25, "k" : 5}
    shape_2D = (30,30)
    densities = pars_dict["k"]*[1.0]
    N = 200
    
    toy = Toy(e_offset = 0,e_size = 200,e_scale = 1,params_dict = pars_dict,seed = seed)
    toy.generate_g_matr()
    toy.generate_spectrum()
    toy.generate_phases()
    G = toy.G
    s = toy.spectrum
    P = toy.P
    phases = toy.phases
    n_GP = (toy.phases/np.sum(toy.phases,axis=1,keepdims=True)).T

    weights = random_weights(shape_2D, pars_dict["k"])

    aspim = ArtificialSpim(phases,densities,weights)
    aspim.generate_spim_deterministic()
    aspim.generate_spim_stochastic(N,seed)

    X = aspim.flatten_X()
    Xdot = aspim.flatten_Xdot()
    A = aspim.flatten_weights()

    np.testing.assert_allclose(Xdot,N*n_GP@A)
    np.testing.assert_array_less(0,Xdot)
    # np.testing.assert_array_less(0,X)


# def test_create_toy_problem():
#     l = 13
#     k = 3
#     p = 100
#     c = 7
#     n_poisson=200
#     toy = ToyModel(l, k, p, c, n_poisson)
#     G, P, A, X, Xdot = toy.get_variables()
#     np.testing.assert_almost_equal(G @ P @ A, X)
#     np.testing.assert_array_less(0, G)
#     np.testing.assert_array_less(0, P)
#     np.testing.assert_array_less(0, A)
#     np.testing.assert_array_less(0, X)
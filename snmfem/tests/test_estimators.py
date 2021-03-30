from sklearn.utils.estimator_checks import check_estimator
from snmfem.estimators import NMF, SmoothNMF
import numpy as np
from snmfem.datasets.generate_data import ArtificialSpim
from snmfem.models import EDXS
from snmfem.datasets.generate_weights import generate_weights
from snmfem.measures import trace_xtLx
from snmfem.laplacian import create_laplacian_matrix

def generate_one_sample():
    model_parameters  = {"params_dict" : {"c0" : 4.8935e-05, 
                                            "c1" : 1464.19810,
                                            "c2" : 0.04216872,
                                            "b0" : 0.15910789,
                                            "b1" : -0.00773158,
                                            "b2" : 8.7417e-04},
                            "db_name" : "simple_xrays_threshold.json",
                            "e_offset" : 0.208,
                            "e_scale" : 0.01,
                            "e_size": 1980,
                            "width_slope" : 0.01,
                            "width_intercept" : 0.065,
                            "seed" : 1}


    g_parameters = {"elements_list" : [8,13,14,12,26,29,31,72,71,62,60,92,20],
                        "brstlg" : 1}

    phases_parameters =  [
        {"elements_dict":{"8": 1.0, "12": 0.51, "14": 0.61, "13": 0.07, "20": 0.04, "62": 0.02,
                            "26": 0.028, "60": 0.002, "71": 0.003, "72": 0.003, "29": 0.02}, 
            "scale" : 1},
        {"elements_dict":{"8": 0.54, "26": 0.15, "12": 1.0, "29": 0.038,
                            "92": 0.0052, "60": 0.004, "31": 0.03, "71": 0.003},
            "scale" : 1},   
            {"elements_dict":{"8": 1.0, "14": 0.12, "13": 0.18, "20": 0.47,
                            "62": 0.04, "26": 0.004, "60": 0.008, "72": 0.004, "29": 0.01}, 
            "scale" : 1} 
        ]

    # Generate the phases
    model = EDXS(**model_parameters)
    model.generate_g_matr(**g_parameters)
    model.generate_phases(phases_parameters)
    phases = model.phases
    G = model.G

    seed = 0
    n_phases = 3
    weights_parameters = {"weight_type": "laplacian",
                            "shape_2D": [15, 15]}

    weights = generate_weights(**weights_parameters, n_phases=n_phases, seed=seed)

    # list of densities which will give different total number of events per spectra
    densities = np.array([1.0, 1.33, 1.25])

    spim = ArtificialSpim(phases, densities, weights, G=G)

    N = 50
    spim.generate_spim_stochastic(N)

    D = spim.phases.T
    A = spim.flatten_weights()
    P = np.abs(np.linalg.lstsq(G, D, rcond=None)[0])
    X = spim.flatten_gen_spim()

    w = spim.densities
    Xdot = spim.flatten_Xdot()
    return G, P, A, D, w, X, Xdot/N


def test_generate_one_sample():
    G, P, A, D, w, X, Xdot = generate_one_sample()
    np.testing.assert_allclose(G @ P , D, atol=1e-5)
    np.testing.assert_allclose(D @ np.diag(w) @ A , Xdot)
    np.testing.assert_allclose(G @ P @ np.diag(w) @ A , Xdot, atol=1e-5)


# def test_NMF_scikit () : 
#     estimator = NMF(n_components= 5,max_iter=200,force_simplex = True,mu = 1.0, epsilon_reg = 1.0)
#     check_estimator(estimator)

# def test_smooth_NMF() :
#     estimator = SmoothNMF(shape_2d=[12, 32], lambda_L=2, n_components= 5,max_iter=200,force_simplex = True,mu = 1.0, epsilon_reg = 1.0)

def test_general():
    G, P, A, D, w, X, Xdot = generate_one_sample()

    estimator = NMF(n_components= 3,max_iter=200,force_simplex = True,mu = 0, epsilon_reg = 1)
    D2 = estimator.fit_transform(G=G, A=A, X=Xdot)
    np.testing.assert_allclose(D@np.diag(w), D2, atol=1e-5)

    estimator = NMF(n_components= 3,max_iter=200,force_simplex = True,mu = 0, epsilon_reg = 1)
    D2 = estimator.fit_transform( A=A, X=Xdot)
    np.testing.assert_allclose(D@np.diag(w), D2, atol=1e-5)

    estimator = NMF(n_components= 3,max_iter=200,force_simplex = False,mu = 0, epsilon_reg = 1)
    D2 = estimator.fit_transform(G =G, P=P@np.diag(w), X=Xdot)
    np.testing.assert_allclose(D@np.diag(w), D2, atol=1e-5)

    estimator = NMF(n_components= 3,max_iter=200,force_simplex = True,mu = 0, epsilon_reg = 1)
    D2 = estimator.fit_transform(G =G, P=P@np.diag(w), X=Xdot)
    np.testing.assert_allclose(D@np.diag(w), D2, atol=1e-5)

    estimator = NMF(n_components= 3,max_iter=200,force_simplex = True,mu = 0, epsilon_reg = 1)
    estimator.fit_transform(G=G, X=Xdot)
    P2, A2 = estimator.P_, estimator.A_ 
    np.testing.assert_allclose(G @ P2 @ A2,  Xdot, atol=1e-2)

    estimator = NMF(n_components= 3,max_iter=200,force_simplex = True,mu = 0, epsilon_reg = 1)
    estimator.fit_transform(G=G, X=X)
    P2, A2 = estimator.P_, estimator.A_ 
    np.testing.assert_allclose(G @ P2 @ A2,  X, atol=1e-2)

    estimator = SmoothNMF(lambda_L=0, n_components= 3,max_iter=200,force_simplex = True,mu = 0, epsilon_reg = 1, )
    estimator.fit_transform(G=G, X=X, shape_2d=[15,15])
    P2, A2 = estimator.P_, estimator.A_ 
    np.testing.assert_allclose(G @ P2 @ A2,  X, atol=1e-2)

    estimator = SmoothNMF(lambda_L=10, n_components= 3,max_iter=200,force_simplex = True,mu = 0, epsilon_reg = 1, )
    estimator.fit_transform(G=G, X=X, shape_2d=[15,15])
    P3, A3 = estimator.P_, estimator.A_ 
    np.testing.assert_allclose(G @ P3 @ A3,  X, atol=1e-2)
    L = create_laplacian_matrix(15, 15)

    assert(trace_xtLx(L, A3.T) < trace_xtLx(L, A2.T))
    assert(trace_xtLx(L, A.T) < trace_xtLx(L, A2.T) )
    assert(trace_xtLx(L, A3.T) < trace_xtLx(L, A.T) )

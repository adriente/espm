from sklearn.utils.estimator_checks import check_estimator
from espm.estimators.surrogates import diff_surrogate, smooth_l2_surrogate, smooth_dgkl_surrogate
from espm.estimators import SmoothNMF
from espm.estimators.base import normalization_factor
import numpy as np
from espm.models import EDXS
from espm.datasets.generate_weights import generate_weights
from espm.datasets.base import generate_spim
from espm.measures import trace_xtLx
from espm.utils import create_laplacian_matrix
from espm.models.edxs import G_EDXS
import hyperspy.api as hs

def generate_one_sample():
    model_parameters = {
        "e_offset" : 0.2,
        "e_size" : 2000,
        "e_scale" : 0.01,
        "width_slope" : 0.01,
        "width_intercept" : 0.065,
        "db_name" : "default_xrays.json",
        "E0" : 200,
        "params_dict" : {
            "Abs" : {
                "thickness" : 100.0e-7,
                "toa" : 35,
                "density" : 5
            },
            "Det" : "SDD_efficiency.txt"
        }
    }

    misc_params = {
    "N" : 100,
    "densities" : [1.3,1.6],
    "data_folder" : "test_gen_data",
    "seed" : 42,
    "weight_type" : "laplacian",
    "shape_2d" : (10,20),
    "weights_params" : {"radius" : 1.5},
    "model" : "EDXS"}
    
    
    phases_parameters = [{"b0" : 5e-3,
                            "b1" : 3e-2,
                            "scale" : 0.05,
                            "elements_dict" : {"Fe" : 0.54860348,
                                      "Pt" : 0.38286879,
                                      "Mo" : 0.03166235}},
                            {"b0" : 7e-3,
                            "b1" : 5e-2,
                            "scale" : 0.05,
                            "elements_dict" : {"Ca" : 0.54860348,
                                      "Si" : 0.38286879,
                                      "O" : 0.15166235}}]

    N = misc_params["N"]
    
    # Generate the phases
    model = EDXS(**model_parameters)
    model.generate_phases(phases_parameters)
    phases = model.phases
    model.generate_g_matr(g_type="bremsstrahlung", elements=["Fe", "Mo", "Ca", "Si", "O", "Pt"] ,reference_elt={})
    G = model.G

    weights = generate_weights(misc_params["weight_type"], misc_params["shape_2d"], n_phases=len(phases_parameters), seed=misc_params["seed"], **misc_params["weights_params"])

    stoch = generate_spim(phases, weights, misc_params["densities"], misc_params["N"], seed=misc_params["seed"],continuous = False)
    cont = generate_spim(phases, weights, misc_params["densities"], misc_params["N"], seed=misc_params["seed"],continuous = True)

    spim_stoch = hs.signals.Signal1D(stoch)
    spim_stoch.set_signal_type("EDS_espm")

    spim_cont = hs.signals.Signal1D(cont)
    spim_cont.set_signal_type("EDS_espm")

    X = spim_stoch.X
    X_cont = spim_cont.X
    
    D = phases.T
    H = weights.reshape((misc_params["shape_2d"][0]*misc_params["shape_2d"][1],len(phases_parameters))).T
    
    W = np.abs(np.linalg.lstsq(G, D, rcond=None)[0])
    for i in range(10) : 
        G = G_EDXS(model_parameters, {"g_type" : "bremsstrahlung", "elements" : ["Fe", "Mo", "Ca", "Si", "O", "Pt"]},W[:-2,:],G)
        W = np.abs(np.linalg.lstsq(G, D, rcond=None)[0])

    w = np.array(misc_params["densities"])

    return G, W, H, D, w, X, X_cont, N

def gen_fixed_mat () : 
    fixed_W = -1*np.ones((8,2))
    fixed_W[0,0] = 0.0
    fixed_W[1,0] = 0.0

    fixed_H = -1*np.ones((2,200))
    fixed_H[0,0:20] = 1.0
    fixed_H[1,0:20] = 0.0

    return fixed_W, fixed_H


def test_generate_one_sample():
    G, W, H, D, w, X, Xdot, N = generate_one_sample()
    np.testing.assert_allclose(G @ W , D, atol=1e-3)
    np.testing.assert_allclose( N * D @ np.diag(w) @ H , Xdot)
    np.testing.assert_allclose(N * G @ W @ np.diag(w) @ H , Xdot, atol=1e-1)

def test_NMF_scikit () : 
    estimator = SmoothNMF(n_components= 5,max_iter=200,force_simplex = True,mu = 1.0, epsilon_reg = 1.0,hspy_comp = False)
    check_estimator(estimator)
    estimator = SmoothNMF( n_components= 5,lambda_L=2,max_iter=200,force_simplex = True,mu = 1.0, epsilon_reg = 1.0,hspy_comp = False)
    check_estimator(estimator)

def test_general():
    G, W, H, D, w, X, Xdot, N = generate_one_sample()

    estimator = SmoothNMF(G=G,n_components= 2,max_iter=200,force_simplex = True,mu = 0, epsilon_reg = 1, hspy_comp = False)
    D2 = estimator.fit_transform(H=H, X=Xdot)
    np.testing.assert_allclose(N*D@np.diag(w), D2, atol=3e-1)

    estimator = SmoothNMF(n_components= 2,max_iter=200,force_simplex = True,mu = 0, epsilon_reg = 1, hspy_comp = False)
    D2 = estimator.fit_transform(H=H, X=Xdot)
    np.testing.assert_allclose(N*D@np.diag(w), D2, atol=3e-1)

    estimator = SmoothNMF(G=G,n_components= 2,max_iter=200,force_simplex = False,mu = 0, epsilon_reg = 1, hspy_comp = False)
    D2 = estimator.fit_transform( W=W@np.diag(w), X=Xdot)
    np.testing.assert_allclose(N*D@np.diag(w), D2, atol=3e-1)

    estimator = SmoothNMF(G =G, n_components= 2,max_iter=200,force_simplex = True,mu = 0, epsilon_reg = 1, hspy_comp = False)
    D2 = estimator.fit_transform(W=N*W@np.diag(w), X=Xdot)
    np.testing.assert_allclose(N*D@np.diag(w), D2, atol=3e-1)

    estimator = SmoothNMF(G=G, n_components= 2,max_iter=200,force_simplex = True,mu = 0, epsilon_reg = 1, hspy_comp = False)
    estimator.fit_transform(X=Xdot)
    P2, A2 = estimator.W_, estimator.H_ 
    np.testing.assert_allclose(G @ P2 @ A2,  Xdot, atol=1)

    estimator = SmoothNMF(G=G, n_components= 2,max_iter=200,force_simplex = True,mu = 0, epsilon_reg = 1, hspy_comp = False)
    estimator.fit_transform(X=X)
    P2, A2 = estimator.W_, estimator.H_ 
    np.testing.assert_allclose(G @ P2 @ A2,  Xdot, atol=1)

    estimator = SmoothNMF(G=G, shape_2d=[10,20], lambda_L=0, n_components= 2,max_iter=200,force_simplex = True,mu = 0, epsilon_reg = 1, hspy_comp = False)
    estimator.fit_transform(X=X)
    P2, A2 = estimator.W_, estimator.H_ 
    np.testing.assert_allclose(G @ P2 @ A2,  Xdot, atol=1)

    estimator = SmoothNMF(G=G, lambda_L=100, n_components= 2,max_iter=200,force_simplex = True,mu = 0, epsilon_reg = 1, shape_2d=[10,20], hspy_comp = False)
    estimator.fit_transform(X=X)
    P3, A3 = estimator.W_, estimator.H_ 
    np.testing.assert_allclose(G @ P3 @ A3,  Xdot, atol=1)
    L = create_laplacian_matrix(10, 20)

    assert(trace_xtLx(L, A3.T) < trace_xtLx(L, A2.T))
    # assert(trace_xtLx(L, A.T) < trace_xtLx(L, A2.T) )
    assert(trace_xtLx(L, A3.T) < trace_xtLx(L, H.T) )

def test_fixed_mat () :
    G, W, H, D, w, X, Xdot, N = generate_one_sample()
    fW, fH = gen_fixed_mat()
    estimator = SmoothNMF(G=G, n_components= 2,max_iter=200,force_simplex = True, fixed_W = fW, hspy_comp = False)
    estimator.fit_transform(X=X)
    P2, A2 = estimator.W_, estimator.H_ 
    np.testing.assert_allclose(P2[fW >= 0],fW[fW>=0])

    estimator = SmoothNMF(G=G, n_components= 2,max_iter=200,force_simplex = True, fixed_H = fH, hspy_comp = False)
    estimator.fit_transform(X=X)
    P2, A2 = estimator.W_, estimator.H_ 
    np.testing.assert_allclose(A2[fH >= 0],fH[fH>=0])

def test_surrogate_smooth_nmf():
    L = create_laplacian_matrix(4, 3)
    for i in range(10):
        A1 = np.random.randn(3, 12)
        v1 = smooth_l2_surrogate(A1, L)
        v2 = smooth_l2_surrogate(A1, L, A1)
        v3 = trace_xtLx(L, A1.T) / 2
        np.testing.assert_almost_equal(v1, v2)
        np.testing.assert_almost_equal(v1, v3)

        for j in range(10):
            A2 = np.random.randn(3, 12)
            v4 = smooth_l2_surrogate(A1, L, A2)
            v5 = trace_xtLx(L, A2.T) / 2
            assert v4 >= v5
            d = diff_surrogate(A1, A2, L=L, algo="l2_surrogate")
            np.testing.assert_almost_equal(v4 - v5 , d)
            
def test_surrogate_smooth_dgkl_nmf():

    L = create_laplacian_matrix(4, 3)
    for i in range(10):
        A1 = np.random.rand(3, 12)
        v1 = smooth_dgkl_surrogate(A1, L)
        v2 = smooth_dgkl_surrogate(A1, L, A1)
        v3 = trace_xtLx(L, A1.T) / 2
        np.testing.assert_almost_equal(v1, v2)
        np.testing.assert_almost_equal(v1, v3)

        for j in range(10):
            A2 = np.random.rand(3, 12)
            v4 = smooth_dgkl_surrogate(A1, L, A2)
            v5 = trace_xtLx(L, A2.T) / 2
            assert v4 >= v5
            d = diff_surrogate(A1, A2, L=L, algo="log_surrogate")
            np.testing.assert_almost_equal(v4 - v5 , d)


def test_X_normalize () : 
    X = np.random.rand(10,32)
    fac = np.random.rand()*50
    X_plus = np.concatenate([X/fac,X/fac],axis = 0)

    nc = 5

    estim = SmoothNMF(n_components = nc,lambda_L=1.0, max_iter = 10, init = "nndsvd", normalize=True, shape_2d = [8, 4])
    GP = estim.fit_transform(X)
    GP_plus = estim.fit_transform(X_plus)

    np.testing.assert_allclose(GP_plus*fac, np.concatenate([GP, GP], axis=0))

def test_normalization_factor () : 
    X_high = np.random.rand(10,32)
    fac = np.random.rand()*50
    X_low = X_high / fac

    nc = 5

    X_high_norm = normalization_factor(X_high,nc) * X_high
    X_low_norm = normalization_factor(X_low,nc) * X_low

    np.testing.assert_allclose(X_high_norm, X_low_norm)


# def test_losses():
#     G, P, A, D, w, X, Xdot, N = generate_one_sample()
#     true_spectra = (G @ P @ np.diag(w))
#     true_maps = A 

#     estimator = SmoothNMF(G = G, shape_2d = (10,20), n_components = 2, max_iter = 10, true_A = true_maps, true_D = true_spectra, lambda_L = 1, hspy_comp = False)

#     estimator.fit_transform(X = X)

#     loss = estimator.get_losses()
#     # k = 2
#     # default_params = {
#     # "n_components" : k,
#     # "tol" : 1e-6,
#     # "max_iter" : 10,
#     # "init" : "random",
#     # "random_state" : 1,
#     # "verbose" : 0
#     # }

#     # params_snmf = {
#     #     "force_simplex" : True,
#     #     "mu": np.random.rand(k)
#     # }

#     # params_evalution = {
#     #     "u" : True,
#     # }

#     # # All parameters are contained here
#     # exp = {"name": "snmfem smooth 30", "method": "SmoothNMF", "params": {**default_params, **params_snmf, "lambda_L" : 100.0}}

#     # spim = hs.signals.Signal1D(X)
#     # spim.set_signal_type("EDXSsnmfem")

#     # estimator = SmoothNMF(**exp["params"])
    
#     # m, (G, P, A), loss  = run_experiment(spim,estimator,exp)
    
#     values = np.array([list(e) for e in loss])
#     np.testing.assert_allclose(KL(X, G@P @ A, average=True), values[-1,1])
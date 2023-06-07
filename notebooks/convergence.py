import numpy as np
from espm.conf import log_shift, dicotomy_tol, sigmaL
from espm.utils import create_laplacian_matrix
from espm.estimators import SmoothNMF
from espm.models import ToyModel
from copy import deepcopy
from espm.weights import generate_weights as gw
from espm.datasets.base import generate_spim_sample
from espm.estimators.updates import initialize_algorithms
from tqdm import tqdm


# global parameters
global_param = dict()
global_param["l2"] = False
global_param["verbose"]= 0
global_param["tol"] = 0
global_param["max_iter"] = 10000
global_param["dicotomy_tol"] = dicotomy_tol
global_param["debug"] = False
global_param["log_shift"] = log_shift 
global_param["eval_print"] = 10
global_param["hspy_comp"] = False
global_param["no_stop_criterion"] = True
global_param["init"] = "nndsvda"

k = 3
shape_2d = (25, 25)
l = 25
c = 10
n_poisson = 500
lambda_L = 1000

def create_toy_problem(l = 25, k = 3, shape_2d = [10, 10], c = 10, n_poisson=200, seed=0, force_simplex=True):
    p = np.prod(shape_2d)
    assert len(shape_2d) == 2
    np.random.seed(seed)
    H = np.random.rand(k,p)
    if force_simplex:
        H = H/np.sum(H, axis=0, keepdims=True)
    
    G = np.random.rand(l,c)
    W = np.random.rand(c,k)
    D = G @ W

    X = D @ H

    Xdot = 1/n_poisson * np.random.poisson(n_poisson * X)

    return G, D, H, X, Xdot

def get_toy_sample(l = 25, k = 3, shape_2d = [10, 10], c = 10, n_poisson=200, seed=0):
    model_params = {"L": l, "C": c, "K": k, "seed": seed}
    # densities = np.random.uniform(0.1, 2.0, 3)
    densities = np.ones([k])
    misc_params = {"N": n_poisson, "seed": seed, 'densities' : densities, "model": "ToyModel"}

    toy_model = ToyModel(**model_params)
    toy_model.generate_phases()
    phases = toy_model.phases.T
    weights = gw.generate_weights("laplacian", shape_2d=shape_2d, k=k, seed=seed)

    sample = generate_spim_sample(phases, weights, model_params,misc_params, seed = seed)
    return sample

def create_laplacian_problem(l = 25, k = 3, shape_2d = [10, 10], c = 10, n_poisson=200, seed=0):
    sample = get_toy_sample(l=l, k =k, shape_2d = shape_2d, c=c, n_poisson=n_poisson, seed=seed)
    def to_vec(X):
        n = X.shape[2]
        return X.transpose(2,0,1).reshape(n, -1)
    D = sample["GW"].T
    G = sample["G"]
    H = to_vec(sample["H"])
    X = to_vec(sample["X"])
    Xdot = to_vec(sample["Xdot"])
    shape_2d = sample["shape_2d"]

    return G, D, H, X, Xdot


def one_experiment(X, experiment_param, algo_param, global_param):
    est = SmoothNMF(**algo_param, **experiment_param, **global_param)
    force_simplex_init = True
    if force_simplex_init:
        _, W0, H0 = initialize_algorithms(X, est.G, None, None, n_components=est.n_components, init=est.init, random_state=est.random_state, force_simplex=True, logshift=log_shift)
        W = est.fit_transform(X, W=W0, H=H0)
    else:
        W = est.fit_transform(X)
    H = est.H_
    losses = est.get_losses()
    loss = losses["full_loss"].copy()
    final_loss = loss[-1]
    gamma = losses["gamma"].copy()
    return loss, final_loss, W.copy(), H.copy(), gamma




def run_experiment_set(laplacian, noise, force_simplex, seed = 0):

    if laplacian:
        G, D, H, X, Xdot = create_laplacian_problem(l=l, k =k, shape_2d = shape_2d, c=c, n_poisson=n_poisson, seed=seed)
    else:
        G, D, H, X, Xdot = create_toy_problem(l=l, k =k, shape_2d = shape_2d, c=c, n_poisson=n_poisson, seed=seed)

    if noise:
        Y = X
    else:
        Y = Xdot

    true_D = D
    true_H = H
    L = create_laplacian_matrix(*shape_2d)

    # experiment parameters
    experiment_param = dict()
    experiment_param["force_simplex"] = force_simplex
    experiment_param["lambda_L"] = lambda_L 
    experiment_param["mu"] = 0
    experiment_param["epsilon_reg"] = 1
    experiment_param["normalize"] = False
    experiment_param["G"] = None
    experiment_param["shape_2d"] = shape_2d
    experiment_param["n_components"] = k
    experiment_param["true_D"] = true_D
    experiment_param["true_H"] = true_H
    experiment_param["random_state"] = seed

    losses = []
    final_losses = []
    Ws = []
    Hs = []
    params = []
    captions = []
    gammas = []
    if laplacian:
        algos = ["log_surrogate", "l2_surrogate"]
    else:
        algos = ["log_surrogate", "l2_surrogate", "projected_gradient"]
    
    for algo in algos:
        for linesearch in [False, True]:
            # for sL in [sigmaL/4, sigmaL/2, sigmaL]:
            for sL in [sigmaL]:
                # algo parameters
                algo_param = dict()
                algo_param["linesearch"] = linesearch
                algo_param["algo"] = algo
                # algo_param["gamma"] = sL
                if algo == "projected_gradient":
                    algo_param["gamma"] = [500*sL, 500*sL]
                else:
                    algo_param["gamma"] = sL
                params.append(deepcopy([experiment_param, algo_param, global_param]))

                loss, final_loss, W, H, gamma = one_experiment(Y, experiment_param, algo_param, global_param)
                losses.append(loss)
                final_losses.append(final_loss)
                Ws.append(W)
                Hs.append(H)
                gammas.append(gamma)
                cl = "" if linesearch else "no"
                # captions.append(f"{algo} - {cl} linesearch - $\gamma_0$={sL}")
                captions.append(f"{algo} - {cl} linesearch")

    i = np.argmin(final_losses)
    global_param_m = deepcopy(global_param)
    global_param_m["max_iter"] = global_param_m["max_iter"]*3
    loss, l_infty, W, H, _ = one_experiment(Y, experiment_param, params[i][1], global_param_m)       
    np.testing.assert_array_equal(loss[:len(losses[i])], losses[i])
    # plt.plot(loss)
    # plt.plot(losses[i])
    # plt.yscale("log")
    return losses, final_losses, Ws, Hs, params, captions, gammas, l_infty, W, H, true_D, true_H, L, X, Xdot



def test_create_toy_problem():
    k = 3
    shape_2d = (25, 25)
    l = 25
    c = 10
    n_poisson=200

    for seed in range(3):
        G, D, H, Xtrue, X = create_toy_problem(shape_2d=shape_2d, k=k, l=l, c=c, n_poisson=n_poisson, seed=seed)
        G2, D2, H2, Xtrue2, X2 = create_toy_problem(shape_2d=shape_2d, k=k, l=l, c=c, n_poisson=n_poisson, seed=seed)
        np.testing.assert_array_equal(G, G2)
        np.testing.assert_array_equal(D, D2)
        np.testing.assert_array_equal(H, H2)
        np.testing.assert_array_equal(Xtrue, Xtrue2)
        np.testing.assert_array_equal(X, X2)

        G3, D3, H3, Xtrue3, X3 = create_laplacian_problem(shape_2d=shape_2d, k=k, l=l, c=c, n_poisson=n_poisson, seed=seed)
        G4, D4, H4, Xtrue4, X4 = create_laplacian_problem(shape_2d=shape_2d, k=k, l=l, c=c, n_poisson=n_poisson, seed=seed)   
        np.testing.assert_array_equal(G3, G4)
        np.testing.assert_array_equal(D3, D4)
        np.testing.assert_array_equal(H3, H4)
        np.testing.assert_array_equal(Xtrue3, Xtrue4)
        np.testing.assert_array_equal(X3, X4)

        assert G.shape == G3.shape == (l, c)
        assert D.shape == D3.shape == (l, k)
        assert H.shape == H3.shape == (k, np.prod(shape_2d))
        assert Xtrue.shape == Xtrue3.shape == (l, np.prod(shape_2d))
    
if __name__ == "__main__":

    import argparse

    # get parameter from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--laplacian", type=bool, default=True)
    parser.add_argument("--noise", type=bool, default=True)
    parser.add_argument("--force_simplex", type=bool, default=True)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()

    laplacian = args.laplacian
    noise = args.noise
    force_simplex = args.force_simplex
    repetitions = args.repetitions


    test_create_toy_problem()



    losses_l = []
    l_infty_l = []

    for seed in tqdm(range(repetitions), total=repetitions):
        losses, final_losses, Ws, Hs, params, captions, gammas, l_infty, W, H, true_D, true_H, L, X, Xdot = run_experiment_set(laplacian, noise, force_simplex, seed=seed)
        losses_l.append(np.array(losses))
        l_infty_l.append(l_infty)

    losses = np.array(losses_l)
    l_infty = np.array(l_infty_l)

    # Save the results
    filename = f"losses_{laplacian}_{noise}_{force_simplex}.npz"
    np.savez(filename, losses=losses, l_infty=l_infty, params=params, captions=captions, true_D=true_D, true_H=true_H, X=X, Xdot=Xdot, H=H, W=W, gammas=gammas)

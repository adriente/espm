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
from time import time, process_time


# global parameters
global_param = dict()
global_param["l2"] = False
global_param["verbose"]= 0
global_param["tol"] = 0
global_param["dicotomy_tol"] = dicotomy_tol
global_param["debug"] = False
global_param["log_shift"] = log_shift 
global_param["eval_print"] = 10
global_param["hspy_comp"] = False
global_param["no_stop_criterion"] = True
global_param["init"] = "nndsvda"

k = 3
shape_2d = (64, 64)

c = 10
n_poisson = 200
lambda_L = 1000

def create_toy_problem(l = 25, k = 3, shape_2d = [10, 10], c = 10, n_poisson=200, seed=0, simplex_H=True):
    p = np.prod(shape_2d)
    assert len(shape_2d) == 2
    np.random.seed(seed)
    H = np.random.rand(k,p)
    if simplex_H:
        H = H/np.sum(H, axis=0, keepdims=True)
    
    # G = np.random.rand(l,c)
    # W = np.random.rand(c,k)
    # D = G @ W

    # Let us ignore G
    D = np.random.rand(l,k)
    G = None

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
    # # Option 1
    # weights = gw.generate_weights("laplacian", shape_2d=shape_2d, n_phases=k, seed=seed)
    # scale = np.max(weights[:,:,1] + weights[:,:,2])
    # weights[:,:,1:] = weights[:,:,1:] / scale
    # weights[:,:,0] = 1 - np.sum(weights[:,:,1:], axis=-1)

    # Option 2
    weights = gw.generate_weights("laplacian", shape_2d=shape_2d, n_phases=k+1, seed=seed)
    weights = weights[:,:,1:]
    weights = weights / np.sum(weights, axis=-1, keepdims=True)

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
    simplex_H_init = True
    start_time = time()
    if simplex_H_init:
        _, W0, H0 = initialize_algorithms(X, est.G, None, None, n_components=est.n_components, init=est.init, random_state=est.random_state, simplex_H=True, simplex_W=False, logshift=log_shift)
        W = est.fit_transform(X, W=W0, H=H0)
    else:
        W = est.fit_transform(X)
    end_time = time()
    H = est.H_
    losses = est.get_losses()
    loss = losses["full_loss"].copy()
    final_loss = loss[-1]
    gamma = losses["gamma"].copy()
    return loss, final_loss, W.copy(), H.copy(), gamma, end_time - start_time




def run_experiment_set(laplacian, noise, simplex_H, seed = 0, max_iter=1000, l = 25):

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
    experiment_param["simplex_H"] = simplex_H
    experiment_param["simplex_W"] = False
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
    experiment_param["max_iter"] = max_iter

    losses = []
    final_losses = []
    Ws = []
    Hs = []
    params = []
    captions = []
    gammas = []
    times = []
    if laplacian:
        algos = ["log_surrogate", "bmd", "l2_surrogate"]
        # algos = ["bmd"]
    else:
        algos = ["log_surrogate", "l2_surrogate", "bmd", "projected_gradient"]
        # algos = [ "bmd"]
    
    for algo in algos:
        for linesearch in [False, True]:
        # for linesearch in [False]:
            # for sL in [sigmaL/4, sigmaL/2, sigmaL]:
            for sL in [sigmaL]:
                # algo parameters
                algo_param = dict()
                algo_param["linesearch"] = linesearch
                algo_param["algo"] = algo
                # algo_param["gamma"] = sL
                if algo == "projected_gradient":
                    algo_param["gamma"] = [10000*sL, 10000*sL]
                else:
                    algo_param["gamma"] = sL
                params.append(deepcopy([experiment_param, algo_param, global_param]))

                loss, final_loss, W, H, gamma, delta_t = one_experiment(Y, experiment_param, algo_param, global_param)
                times.append(delta_t)
                losses.append(loss)
                final_losses.append(final_loss)
                Ws.append(W)
                Hs.append(H)
                gammas.append(gamma)
                cl = "" if linesearch else "no"
                # captions.append(f"{algo} - {cl} linesearch - $\gamma_0$={sL}")
                captions.append(f"{algo} - {cl} linesearch")

    times = np.array(times)
    i = np.argmin(final_losses)
    experiment_param_m = deepcopy(experiment_param)
    experiment_param_m["max_iter"] = experiment_param_m["max_iter"]*3
    loss, l_infty, W, H, _, _ = one_experiment(Y, experiment_param_m, params[i][1], global_param)       
    np.testing.assert_array_equal(loss[:len(losses[i])], losses[i])
    # plt.plot(loss)
    # plt.plot(losses[i])
    # plt.yscale("log")
    return losses, final_losses, Ws, Hs, params, captions, gammas, l_infty, W, H, true_D, true_H, X, Xdot, times



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

        # assert G.shape == G3.shape == (l, c)
        assert D.shape == D3.shape == (l, k)
        assert H.shape == H3.shape == (k, np.prod(shape_2d))
        assert Xtrue.shape == Xtrue3.shape == (l, np.prod(shape_2d))
    
if __name__ == "__main__":

    # import argparse

    # # get parameter from command line
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--laplacian', action=argparse.BooleanOptionalAction, default=True)
    # parser.add_argument('--noise', action=argparse.BooleanOptionalAction, default=True)
    # parser.add_argument('--simplex_H', action=argparse.BooleanOptionalAction, default=True)
    # parser.add_argument("--repetitions", type=int, default=10)
    # args = parser.parse_args()

    # laplacian = args.laplacian
    # noise = args.noise
    # simplex_H = args.simplex_H
    # repetitions = args.repetitions


    test_create_toy_problem()
    # Time test
    simplex_H = True
    repetitions = 5
    for l in [25, 100, 500, 1000]:
        for laplacian in [True, False]:
            for noise in [True, False]:
                losses_l = []
                l_infty_l = []
                times_l = []
                for seed in tqdm(range(repetitions), total=repetitions):
                    losses, final_losses, Ws, Hs, params, captions, gammas, l_infty, W, H, true_D, true_H, X, Xdot, times = run_experiment_set(laplacian, noise, simplex_H, seed=seed, max_iter=100, l=l)
                    losses_l.append(np.array(losses))
                    l_infty_l.append(l_infty)
                    times_l.append(times)

                losses = np.array(losses_l)
                l_infty = np.array(l_infty_l)
                times = np.array(times_l)

                # Save the results
                filename = f"losses_{laplacian}_{noise}_{simplex_H}_{l}.npz"
                np.savez(filename, losses=losses, l_infty=l_infty, params=params, captions=captions, true_D=true_D, true_H=true_H, X=X, Xdot=Xdot, H=H, W=W, gammas=gammas, times=times)


    # Convergence test
    simplex_H = True
    repetitions = 50
    for laplacian in [True, False]:
        for noise in [True, False]:
            losses_l = []
            l_infty_l = []
            times_l = []
            for seed in tqdm(range(repetitions), total=repetitions):
                losses, final_losses, Ws, Hs, params, captions, gammas, l_infty, W, H, true_D, true_H, X, Xdot, times = run_experiment_set(laplacian, noise, simplex_H, seed=seed, max_iter=1000, l=25)
                losses_l.append(np.array(losses))
                l_infty_l.append(l_infty)
                times_l.append(times)

            losses = np.array(losses_l)
            l_infty = np.array(l_infty_l)
            times = np.array(times_l)

            # Save the results
            filename = f"losses_{laplacian}_{noise}_{simplex_H}.npz"
            np.savez(filename, losses=losses, l_infty=l_infty, params=params, captions=captions, true_D=true_D, true_H=true_H, X=X, Xdot=Xdot, H=H, W=W, gammas=gammas, times=times)


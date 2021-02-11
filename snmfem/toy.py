import numpy as np

def create_toy_problem(l = 25, k = 3, p = 100, c = 10, n_poisson=200, force_simplex=True):

    A = np.random.rand(k,p)
    if force_simplex:
        A = A/np.sum(A, axis=1, keepdims=True)
    
    G = np.random.rand(l,c)
    P = np.random.rand(c,k)
    GP = G @ P

    X = GP @ A

    Xdot = 1/n_poisson * np.random.poisson(n_poisson * X)

    return G, P, A, X, Xdot
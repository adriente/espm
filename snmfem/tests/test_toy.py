import numpy as np
from snmfem.toy import create_toy_problem


def test_create_toy_problem():
    l = 13
    k = 3
    p = 100
    c = 7
    n_poisson=200
    G, P, A, X, Xdot = create_toy_problem(l, k, p, c, n_poisson)
    np.testing.assert_almost_equal(G @ P @ A, X)
    np.testing.assert_array_less(0, G)
    np.testing.assert_array_less(0, P)
    np.testing.assert_array_less(0, A)
    np.testing.assert_array_less(0, X)
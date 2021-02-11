import numpy as np

from snmfem.updates import dichotomy_simplex, multiplicative_step_p, multiplicative_step_a
from snmfem.measures import KLdiv_loss, log_reg
from snmfem.conf import log_shift, dicotomy_tol


def make_step_a(x_matr, g_matr, p_matr , a_matr, mu_sparse=0, eps=log_shift, eps_sparse=1, mask=None):
    """
    Multiplicative step in A.
    The main terms are calculated first.
    With mu_sparse = 0, the steps in A are calculated once. For mu_sparse != 0, the steps in A are calculated first with particle regularization. Then only the entries allowed by the mask are calculaed, without particle regularization.
    To calculate the regularized step, we make a linear approximation of the log.
    """
    # Update of the d_matr (maybe it should be done in a dedicated function for clarity)
    d_matr = g_matr @ p_matr

    # Multiplicative update numerator U and denominator V
    d_a = d_matr @ a_matr
    U = d_matr.T @ (x_matr.clip(min=1e-150) / d_a)
    V = d_matr.sum(axis=0)
    # Reset of the Lagrangian multiplier (Maybe non-necessary .. ?)
    # lambda_s = np.zeros((x_shape[1] * x_shape[0],))

    if mu_sparse != 0:
        # Regularized part of the algorithm
        if mask is None:
            # In the linear approximation, the slope is constant. We modifiy this slope to approximate the log every 10 iterations.
            # The number of iterations in between two slope changes is arbitrary.
            # if num_iterations % 10 == 0:
            fixed_a = a_matr.copy()
            # Vectorized version of the regularization
            vec_sparse = np.array([0] + (a_matr.shape[0] - 1) * [mu_sparse])
            denum = V[:, np.newaxis] + vec_sparse[:, np.newaxis] / (
                fixed_a + eps_sparse
            )
            # Lagragian multiplier
            lambda_s = dichotomy_simplex(a_matr * U, denum, dicotomy_tol)
            # A update (regularized)
            a_matr = (
                a_matr
                / (
                    V[:, np.newaxis]
                    + vec_sparse[:, np.newaxis] / (fixed_a + eps_sparse)
                    + lambda_s
                )
                * U
            )
        else:
            # Lagragian multiplier
            lambda_s = dichotomy_simplex(
                a_matr * U, V[:, np.newaxis], dicotomy_tol
            )
            # Update the entry that did not meet the sparsity requirements
            n_mask = np.invert(mask)
            # A update (masked)
            a_matr[n_mask] = (
                a_matr[n_mask]
                / (V[:, np.newaxis] + lambda_s)[n_mask]
                * U[n_mask]
            )

    else:
        # Lagragian multiplier
        lambda_s = dichotomy_simplex(
            a_matr * U, V[:, np.newaxis], dicotomy_tol
        )
        # A update (not regularized)
        a_matr = a_matr / (V[:, np.newaxis] + lambda_s) * U

    return a_matr

def make_step_p(x_matr, g_matr, p_matr , a_matr, eps = log_shift):
    """
    Multiplicative step in P.
    """
    d_matr = g_matr @ p_matr

    d_a = d_matr @ a_matr
    term1 = (
        g_matr.T @ (x_matr.clip(min=1e-150) / (d_a + eps)) @ a_matr.T
    )  # The data are clipped to avoid issues during computation
    term2 = (
        g_matr.sum(axis=0)[:, np.newaxis]
        @ a_matr.sum(axis=1)[:, np.newaxis].T
    )
    # P update
    p_matr = p_matr / term2 * term1
    return p_matr

def test_dichotomy_simmplex():
    num = np.random.rand(1,1) + 1
    denum = np.random.rand(1,1)
    sol = num - denum
    tol = 1e-6
    sol2 = dichotomy_simplex(num, denum, tol )
    assert(np.abs(sol -sol2 )< tol)

    n = 10
    num = np.random.rand(1,n)
    denum = np.random.rand(1, n)
    sol = np.squeeze(num - denum)
    sol2 = dichotomy_simplex(num, denum, tol )
    np.testing.assert_allclose(sol2, sol, atol=tol)

    num = np.random.rand(n, 1)
    denum = np.random.rand(n, 1)
    tol = 1e-6
    sol = dichotomy_simplex(num, denum, tol )
    np.sum(num/(denum + sol))
    np.testing.assert_allclose(np.sum(num/(denum + sol)), 1, atol=tol)

    num = np.random.rand(n, 6)
    denum = np.random.rand(n, 6)
    tol = 1e-6
    sol = dichotomy_simplex(num, denum, tol )
    v = np.sum(num/(denum + sol), axis=0)
    np.testing.assert_allclose(v, np.ones([6]), atol=tol)

    num = np.random.rand(n, 1)
    denum = np.random.rand(n, 1)
    num = num/(np.sum(num/denum))
    np.testing.assert_allclose(np.sum(num/(denum)), 1)

    tol = 1e-6
    sol = dichotomy_simplex(num, denum, tol )
    np.sum(num/(denum + sol))
    np.testing.assert_allclose(sol, 0, atol=tol)

def test_multiplicative_step_p():
    l = 26
    k = 5
    p = 100
    c = 17

    A = np.random.rand(k,p)
    A = A/np.sum(A, axis=1, keepdims=True)
    
    G = np.random.rand(l,c)
    P = np.random.rand(c,k)
    GP = G @ P

    X = GP @ A

    Pp = multiplicative_step_p(X, G, P, A, eps=0)
    np.testing.assert_array_almost_equal(Pp, P)

    for _ in range(10):
        P = np.random.rand(c,k)
        Pp = multiplicative_step_p(X, G, P, A)
        Pp2 = make_step_p(X, G, P, A)
        np.testing.assert_array_almost_equal(Pp, Pp2)
        GP = G @ P
        GPp = G @ Pp
        val1 = KLdiv_loss(X, GP, A)
        val2 = KLdiv_loss(X, GPp, A)
        np.testing.assert_array_less(0, Pp)
        assert(val1 > val2)



def test_multiplicative_step_a():
    l = 26
    k = 5
    p = 100
    c = 17

    A = np.random.rand(k,p)
    A = A/np.sum(A, axis=0, keepdims=True)
    
    G = np.random.rand(l,c)
    P = np.random.rand(c,k)
    GP = G @ P

    X = GP @ A
    np.testing.assert_allclose(np.sum(A, axis=0), np.ones([A.shape[1]]), atol=dicotomy_tol)

    Ap = multiplicative_step_a(X, G, P, A, force_simplex=False, mu=0, eps=0, epsilon_reg=1, safe=True)
    np.testing.assert_array_almost_equal(A, Ap)
    np.testing.assert_allclose(np.sum(Ap, axis=0), np.ones([Ap.shape[1]]), atol=dicotomy_tol)

    Ap = multiplicative_step_a(X, G, P, A, force_simplex=True, mu=0, eps=0, epsilon_reg=1, safe=True)
    np.testing.assert_array_almost_equal(A, Ap)        

    for _ in range(10):
        A = np.random.rand(k,p)
        A = A/np.sum(A, axis=1, keepdims=True)
        Ap =  multiplicative_step_a(X, G, P, A, force_simplex=False, mu=0, eps=0, epsilon_reg=1, safe=True)
        val1 = KLdiv_loss(X, GP, A)
        val2 = KLdiv_loss(X, GP, Ap)
        np.testing.assert_array_less(0, Ap)
        assert(val1 > val2)

        Ap =  multiplicative_step_a(X, G, P, A, force_simplex=True, mu=0, eps=log_shift, epsilon_reg=1, safe=True)
        Ap2 =  make_step_a(X, G, P, A, mu_sparse=0, eps=log_shift, eps_sparse=1, mask=None)
        np.testing.assert_array_almost_equal(Ap2, Ap)
        np.testing.assert_allclose(np.sum(Ap, axis=0), np.ones([Ap.shape[1]]), atol=dicotomy_tol)

        val1 = KLdiv_loss(X, GP, A)
        val2 = KLdiv_loss(X, GP, Ap)
        np.testing.assert_array_less(0, Ap)
        assert(val1 > val2)

        epsilon_reg = 1
        mu = np.ones(k)
        mu[0] = 0
        Ap =  multiplicative_step_a(X, G, P, A, force_simplex=True, mu=mu, eps=log_shift, epsilon_reg=epsilon_reg, safe=True)
        Ap2 =  make_step_a(X, G, P, A, mu_sparse=1, eps=log_shift, eps_sparse=epsilon_reg, mask=None)
        np.testing.assert_array_almost_equal(Ap2, Ap)
        np.testing.assert_allclose(np.sum(Ap, axis=0), np.ones([Ap.shape[1]]), atol=dicotomy_tol)


        Ap =  multiplicative_step_a(X, G, P, A, force_simplex=True, mu=3*mu, eps=log_shift, epsilon_reg=epsilon_reg, safe=True)
        Ap2 =  make_step_a(X, G, P, A, mu_sparse=3, eps=log_shift, eps_sparse=epsilon_reg, mask=None)
        np.testing.assert_array_almost_equal(Ap2, Ap)
        np.testing.assert_allclose(np.sum(Ap, axis=0), np.ones([Ap.shape[1]]), atol=dicotomy_tol)


        val1 = KLdiv_loss(X, GP, A) + log_reg(A, 3*mu, epsilon_reg)
        val2 = KLdiv_loss(X, GP, Ap) + log_reg(A, 3*mu, epsilon_reg)
        np.testing.assert_array_less(0, Ap)
        assert(val1 > val2)

        # Ap =  multiplicative_step_a(X, G, P, A, force_simplex=True, mu=3, eps=log_shift, epsilon_reg=1, safe=True)
        # Ap2 =  make_step_a(X, G, P, A, mu_sparse=3, eps=log_shift, eps_sparse=1, mask=np.zeros([k])>0)
        # np.testing.assert_array_almost_equal(Ap2, Ap)
        # np.testing.assert_allclose(np.sum(Ap, axis=0), np.ones([Ap.shape[1]]), atol=dicotomy_tol)


import numpy as np

from espm.estimators.updates import dichotomy_simplex, multiplicative_step_w, multiplicative_step_h, update_q, dichotomy_simplex_acc, multiplicative_step_hq
from espm.estimators.updates import estimate_Lipschitz_bound_h, estimate_Lipschitz_bound_w, gradW, gradH, proj_grad_step_h, proj_grad_step_w
from espm.measures import KLdiv_loss, log_reg, Frobenius_loss, trace_xtLx
from espm.conf import log_shift, dicotomy_tol
from espm.utils import create_laplacian_matrix
import pytest

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
            lambda_s = dichotomy_simplex(a_matr * U, denum, 0, tol=dicotomy_tol)
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
                a_matr * U, V[:, np.newaxis], 0, tol=dicotomy_tol
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
            a_matr * U, V[:, np.newaxis], 0, tol=dicotomy_tol
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

def test_dichotomy_simplex():
    num = np.random.rand(1,1) + 1
    denum = np.random.rand(1,1)
    sol = num - denum
    tol = 1e-8
    sol2 = dichotomy_simplex(num, denum, 0, tol=tol )
    assert(np.abs(sol -sol2 )< 2*tol)

    n = 10
    num = np.random.rand(1,n)
    denum = np.random.rand(1, n)
    sol = np.squeeze(num - denum)
    sol2 = dichotomy_simplex(num, denum, 0, tol=tol )
    np.testing.assert_allclose(sol2, sol, atol=tol)

    num = np.random.rand(n, 1)
    denum = np.random.rand(n, 1)
    tol = 1e-6
    sol = dichotomy_simplex(num, denum, 0, tol=tol)
    np.sum(num/(denum + sol))
    np.testing.assert_allclose(np.sum(num/(denum + sol)), 1, atol=tol)

    num = np.random.rand(n, 6)
    denum = np.random.rand(n, 6)
    tol = 1e-6
    sol = dichotomy_simplex(num, denum, 0, tol=tol )
    v = np.sum(num/(denum + sol), axis=0)
    np.testing.assert_allclose(v, np.ones([6]), atol=tol)

    num = np.random.rand(n, 1)
    denum = np.random.rand(n, 1)
    num = num/(np.sum(num/denum))
    np.testing.assert_allclose(np.sum(num/(denum)), 1)
    tol = 1e-6
    sol = dichotomy_simplex(num, denum, 0, tol=tol )
    np.testing.assert_allclose(sol, 0, atol=tol)

    tol = 1e-6
    num = np.array([[1,1,0,0,0,2]]).T
    denum = np.array([[1,1,3,5,4,2]]).T
    sol = dichotomy_simplex(num, denum, 0, tol=tol )
    np.testing.assert_allclose(np.sum(num/(denum + sol)), 1, atol=tol)

    tol = 1e-6
    num = np.array([[1,1,0,0,0,2]]).T
    denum = np.array([[1,1,0,5,0,2]]).T
    sol = dichotomy_simplex(num, denum, 0, tol=tol )
    np.sum(num/(denum + sol))
    np.testing.assert_allclose(np.sum(num/(denum + sol)), 1, atol=tol)

def test_dichotomy_simplex_contraint():
    # Test the other constraint
    tol = 1e-8
    num = np.array([[3,0.5]]).T
    denum = np.array([[1,1]]).T
    sol = np.array([[3]]).T
    tol = 1e-8
    sol2 = dichotomy_simplex(num, denum, 1/4, tol=tol )
    assert(np.abs(sol -sol2 )< 2*tol)

    n = 10
    num = np.random.rand(1,n)
    denum = np.random.rand(1, n)
    sol = np.squeeze(num - denum)
    sol2 = dichotomy_simplex(num, denum, 0.5, tol=tol )
    np.testing.assert_allclose(sol2, sol, atol=tol)

    with pytest.raises(ValueError) as _:
        num = np.random.rand(1,n)
        denum = np.random.rand(1, n)
        dichotomy_simplex(num, denum, 1.1, tol=tol )
    with pytest.raises(ValueError) as _:
        num = np.random.rand(3,n)
        denum = np.random.rand(3, n)
        dichotomy_simplex(num, denum, 0.5, tol=tol )

    n = 10
    num = np.random.rand(1,n)
    denum = np.random.rand(1, n)
    sol = np.squeeze(num - denum)
    sol2 = dichotomy_simplex(num, denum, 0.99, tol=tol )
    np.testing.assert_allclose(sol2, sol, atol=tol)

    n = 10
    num = np.random.rand(n, 6)
    denum = np.random.rand(n, 6)
    tol = 1e-6
    log_shift = 0.05
    sol = dichotomy_simplex(num, denum, log_shift, tol=tol )
    v = np.sum(np.maximum(num/(denum + sol), log_shift), axis=0)
    np.testing.assert_allclose(v, np.ones([6]), atol=tol)

def test_dicotomy2():
    k = 5
    p = 6400
    span = np.logspace(-6,6,num=17)
    its = 25
    tol = 0
    maxit = 100
    np.random.seed(0)
    for _ in range(its) : 
        scale_num = np.random.choice(span,size=(k,p))
        num = scale_num * np.random.rand(k,p)
        scale_denum = np.random.choice(span,size=(k,p))
        denum = scale_denum * np.random.rand(k,p)
        sol = dichotomy_simplex(num, denum, 0, tol=tol, maxit=maxit)
        v = np.sum(num/(denum + sol), axis=0)
        np.testing.assert_allclose(v, np.ones([v.shape[0]]), atol=1e-2)

        log_shift = 0.1 / k
        sol = dichotomy_simplex(num, denum, log_shift, tol=tol, maxit=maxit)
        v = np.sum(np.maximum(num/(denum + sol), log_shift), axis=0)
        np.testing.assert_allclose(v, np.ones([v.shape[0]]), atol=1e-2)
        
    for _ in range(its) : 
        scale_num = np.random.choice(span,size=(k,p))
        num = scale_num * np.random.rand(k,p)
        num[np.tile(np.arange(k),p//k),np.arange(p)] = 0
        scale_denum = np.random.choice(span,size=(k,p))
        denum = scale_denum * np.random.rand(k,p)
        sol = dichotomy_simplex(num, denum, 0, tol=tol, maxit=maxit)
        v = np.sum(num/(denum + sol), axis=0)
        np.testing.assert_allclose(v, np.ones([v.shape[0]]), atol=1e-2)

        log_shift = 0.1 / k
        sol = dichotomy_simplex(num, denum, log_shift, tol=tol, maxit=maxit)
        v = np.sum(np.maximum(num/(denum + sol), log_shift), axis=0)
        np.testing.assert_allclose(v, np.ones([v.shape[0]]), atol=1e-2)
        
    for _ in range(its) : 
        scale_num = np.random.choice(span,size=(k,p))
        num = scale_num * np.random.rand(k,p)
        scale_denum = np.random.choice(span,size=(k,1))
        denum = scale_denum * np.random.rand(k,1)
        sol = dichotomy_simplex(num, denum, 0, tol=tol, maxit=maxit)
        v = np.sum(num/(denum + sol), axis=0)
        np.testing.assert_allclose(v, np.ones([v.shape[0]]), atol=1e-2)

        log_shift = 0.1 / k
        sol = dichotomy_simplex(num, denum, log_shift, tol=tol, maxit=maxit)
        v = np.sum(np.maximum(num/(denum + sol), log_shift), axis=0)
        np.testing.assert_allclose(v, np.ones([v.shape[0]]), atol=1e-2)
        
    for _ in range(its) : 
        scale_num = np.random.choice(span,size=(k,p))
        num = scale_num * np.random.rand(k,p)
        num[np.tile(np.arange(k),p//k),np.arange(p)] = 0
        scale_denum = np.random.choice(span,size=(k,1))
        denum = scale_denum * np.random.rand(k,1)
        sol = dichotomy_simplex(num, denum, 0, tol=tol, maxit=maxit)
        v = np.sum(num/(denum + sol), axis=0)
        np.testing.assert_allclose(v, np.ones([v.shape[0]]), atol=1e-2)

        log_shift = 0.1 / k
        sol = dichotomy_simplex(num, denum, log_shift, tol=tol, maxit=maxit)
        v = np.sum(np.maximum(num/(denum + sol), log_shift), axis=0)
        np.testing.assert_allclose(v, np.ones([v.shape[0]]), atol=1e-2)

def test_dicotomy_aq():
    def func_abc(x, a, b, minus_c):
        n_p = len(b)
        return n_p * x + 2*a + np.sum( - np.sqrt( (b + x)**2 + 4*a*minus_c) + b, axis=0)

    # def func_abc_onedim(x, a, b, minus_c):
    #     return x + 2*a + b - np.sqrt( (b + x)**2 + 4*a*minus_c) 


    a = np.random.rand() 
    b = np.random.rand(1,1)
    minus_c = np.random.rand(1,1)
    # x + 2*a + b - np.sqrt( (b + x)**2 + 4*a*minus_c) = 0
    # x + 2*a + b =  np.sqrt( (b + x)**2 + 4*a*minus_c) 
    # (x + b)^2 + 4*a^2 + 4*a (x + b) =   (b + x)**2 + 4*a*minus_c
    # a + (x + b) = minus_c
    # x  = minus_c - a -b 
    sol = minus_c - a - b
    tol = 1e-8
    sol2 = dichotomy_simplex_acc(a, b, minus_c, log_shift=0, tol=tol )
    assert(np.abs(sol -sol2 )< 10*tol)

    a = np.random.rand() 
    b = np.zeros([1,1])
    minus_c = np.random.rand(1,1)
    sol = minus_c - a - b
    tol = 1e-8
    sol2 = dichotomy_simplex_acc(a, b, minus_c, log_shift=0, tol=tol )
    assert(np.abs(sol -sol2 )< 10*tol)

    a = np.random.rand() 
    b = np.random.rand(1,1)
    minus_c = np.zeros([1,1])
    sol = minus_c - a - b
    tol = 1e-8
    sol2 = dichotomy_simplex_acc(a, b, minus_c, log_shift=0, tol=tol )
    assert(np.abs(sol -sol2 )< 10*tol)

    a = np.random.rand() 
    b = np.zeros([1,1])
    minus_c = np.zeros([1,1])
    sol = minus_c - a - b
    tol = 1e-8
    sol2 = dichotomy_simplex_acc(a, b, minus_c, log_shift=0, tol=tol )
    assert(np.abs(sol -sol2 )< 10*tol)


    n = 10
    a = np.random.rand() 
    b = np.random.rand(1,n)
    minus_c = np.random.rand(1,n)
    sol = np.squeeze(minus_c - a - b)
    sol2 = dichotomy_simplex_acc(a, b, minus_c, log_shift=0, tol=tol )
    np.testing.assert_allclose(sol2, sol, atol=10*tol)

    a = np.random.rand() 
    b = np.random.rand(n,6)
    minus_c = np.random.rand(n,6)
    tol = 1e-6
    sol = dichotomy_simplex_acc(a, b, minus_c, log_shift=0, tol=tol)
    np.testing.assert_allclose(func_abc(sol, a, b, minus_c), np.zeros([6]), atol=10*tol)


def test_dichotomy_aq_constraint():
    # Test the other constraint
    tol = 1e-8
    a = 4
    log_shift = 1/8
    b = np.array([[1,1,1, 8/log_shift, 8/log_shift]]).T
    minus_c = np.array([[1,1,1, 1, 1]]).T

    sol = 2
    tol = 1e-8
    sol2 = dichotomy_simplex_acc(a, b, minus_c, log_shift=log_shift, tol=tol )
    assert(np.abs(sol -sol2 )< 10*tol)


    n = 10
    a = np.random.rand() 
    b = np.random.rand(1,n)
    minus_c = np.random.rand(1,n)
    sol = np.squeeze(minus_c - a - b)
    sol2 = dichotomy_simplex_acc(a, b, minus_c, log_shift=0.5, tol=tol )
    np.testing.assert_allclose(sol2, sol, atol=10*tol)


    with pytest.raises(ValueError) as _:
        a = np.random.rand(1,n)
        b = np.random.rand(1, n)
        minus_c = np.random.rand(1,n)
        sol2 = dichotomy_simplex_acc(a, b, minus_c, log_shift=1.1, tol=tol )

    with pytest.raises(ValueError) as _:
        a = np.random.rand(3,n)
        b = np.random.rand(3, n)
        minus_c = np.random.rand(3,n)
        sol2 = dichotomy_simplex_acc(a, b, minus_c, log_shift=0.5, tol=tol )

    n = 10
    a = np.random.rand() 
    b = np.random.rand(1,n)
    minus_c = np.random.rand(1,n)
    sol = np.squeeze(minus_c - a - b)
    sol2 = dichotomy_simplex_acc(a, b, minus_c, log_shift=0.99, tol=tol )
    np.testing.assert_allclose(sol2, sol, atol=10*tol)

    def func_abc(x, a, b, minus_c, log_shift):
        return  np.sum( np.maximum(np.sqrt( (b + x)**2 + 4*a*minus_c) - b - x, log_shift*2*a), axis=0) - 2*a

    n = 10
    a = np.random.rand() 
    b = np.random.rand(6,n)
    minus_c = np.random.rand(6,n)
    log_shift = 0.1
    sol = dichotomy_simplex_acc(a, b, minus_c, log_shift=log_shift, tol=tol )
    v = func_abc(sol, a, b, minus_c, log_shift)
    np.testing.assert_allclose(v, np.zeros([n]), atol=tol)


def test_dicotom_aq2():
    k = 5
    p = 6400
    maxit = 100
    span = np.logspace(-6,5,num=17)
    its = 25
    tol = 1e-6
    # def func_abc(x, a, b, minus_c):
    #     n_p = len(b)
    #     return n_p * x + 2*a + np.sum( - np.sqrt( (b + x)**2 + 4*a*minus_c) + b, axis=0)
    def func_abc(x, a, b, minus_c, log_shift=0):
        return  np.sum( np.maximum(np.sqrt( (b + x)**2 + 4*a*minus_c) - b - x, log_shift*2*a), axis=0) - 2*a


    np.random.seed(0)
    for _ in range(its) : 
        a = np.random.rand()  
        scale_num = np.random.choice(span,size=(k,p))
        b = scale_num * np.random.rand(k,p)
        scale_denum = np.random.choice(span,size=(k,1))
        minus_c = scale_denum * np.random.rand(k,1)
        sol = dichotomy_simplex_acc(a, b, minus_c, log_shift=0, tol=tol, maxit=maxit )
        np.testing.assert_allclose(func_abc(sol, a, b, minus_c), np.zeros([p]), atol=tol)

        log_shift = 0.1 / k
        sol = dichotomy_simplex_acc(a, b, minus_c, log_shift=log_shift, tol=tol, maxit=maxit )
        np.testing.assert_allclose(func_abc(sol, a, b, minus_c, log_shift=log_shift), np.zeros([p]), atol=tol)

    for _ in range(its) :
        a = np.random.rand()  
        scale_num = np.random.choice(span,size=(k,p))
        b = scale_num * np.random.rand(k,p)
        scale_denum = np.random.choice(span,size=(k,p))
        minus_c = scale_denum * np.random.rand(k,p)
        sol = dichotomy_simplex_acc(a, b, minus_c, log_shift=0, tol=tol, maxit=maxit )
        np.testing.assert_allclose(func_abc(sol, a, b, minus_c), np.zeros([p]), atol=tol)

        log_shift = 0.1 / k
        sol = dichotomy_simplex_acc(a, b, minus_c, log_shift=log_shift, tol=tol, maxit=maxit )
        np.testing.assert_allclose(func_abc(sol, a, b, minus_c, log_shift=log_shift), np.zeros([p]), atol=tol)
        
    for _ in range(its) : 
        a = np.random.rand()  
        scale_num = np.random.choice(span,size=(k,p))
        b = scale_num * np.random.rand(k,p)
        b[np.tile(np.arange(k),p//k),np.arange(p)] = 0
        scale_denum = np.random.choice(span,size=(k,p))
        minus_c = scale_denum * np.random.rand(k,p)
        sol = dichotomy_simplex_acc(a, b, minus_c, log_shift=0, tol=tol, maxit=maxit )
        np.testing.assert_allclose(func_abc(sol, a, b, minus_c), np.zeros([p]), atol=tol)

        log_shift = 0.1 / k
        sol = dichotomy_simplex_acc(a, b, minus_c, log_shift=log_shift, tol=tol, maxit=maxit )
        np.testing.assert_allclose(func_abc(sol, a, b, minus_c, log_shift=log_shift), np.zeros([p]), atol=tol)        

    for _ in range(its) : 
        a = np.random.rand()  
        scale_num = np.random.choice(span,size=(k,p))
        b = scale_num * np.random.rand(k,p)
        b[np.tile(np.arange(k),p//k),np.arange(p)] = 0
        scale_denum = np.random.choice(span,size=(k,1))
        minus_c = scale_denum * np.random.rand(k,1)
        sol = dichotomy_simplex_acc(a, b, minus_c, log_shift=0, tol=tol, maxit=maxit )
        np.testing.assert_allclose(func_abc(sol, a, b, minus_c), np.zeros([p]), atol=tol)

        log_shift = 0.1 / k
        sol = dichotomy_simplex_acc(a, b, minus_c, log_shift=log_shift, tol=tol, maxit=maxit )
        np.testing.assert_allclose(func_abc(sol, a, b, minus_c, log_shift=log_shift), np.zeros([p]), atol=tol)
        
def test_multiplicative_step_w():
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

    Pp = multiplicative_step_w(X, G, P, A, log_shift=0)
    np.testing.assert_array_almost_equal(Pp, P)

    Pp = multiplicative_step_w(X, G, P, A, log_shift=0, l2=True)
    np.testing.assert_array_almost_equal(Pp, P)

    for _ in range(10):
        P = np.random.rand(c,k)
        Pp = multiplicative_step_w(X, G, P, A)
        Pp2 = make_step_p(X, G, P, A)
        np.testing.assert_array_almost_equal(Pp, Pp2)
        GP = G @ P
        GPp = G @ Pp
        val1 = KLdiv_loss(X, GP, A)
        val2 = KLdiv_loss(X, GPp, A)
        np.testing.assert_array_less(0, Pp)
        assert(val1 > val2)


    for _ in range(10):
        P = np.random.rand(c,k)
        Pp = multiplicative_step_w(X, G, P, A, l2=True)
        GP = G @ P
        GPp = G @ Pp
        val1 = Frobenius_loss(X, GP, A)
        val2 = Frobenius_loss(X, GPp, A)
        np.testing.assert_array_less(0, Pp)
        assert(val1 > val2)


def test_multiplicative_step_h():
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

    Ap = multiplicative_step_h(X, G, P, A, force_simplex=False, mu=0, log_shift=0, epsilon_reg=1, safe=True)
    np.testing.assert_array_almost_equal(A, Ap)
    np.testing.assert_allclose(np.sum(Ap, axis=0), np.ones([Ap.shape[1]]), atol=dicotomy_tol)

    Ap = multiplicative_step_h(X, G, P, A, force_simplex=True, mu=0, log_shift=0, epsilon_reg=1, safe=True)
    np.testing.assert_allclose(A, Ap, atol=dicotomy_tol)        

    # Same test for l2
    Ap = multiplicative_step_h(X, G, P, A, force_simplex=False, mu=0, log_shift=0, epsilon_reg=1, safe=True, l2=True)
    np.testing.assert_array_almost_equal(A, Ap)
    np.testing.assert_allclose(np.sum(Ap, axis=0), np.ones([Ap.shape[1]]), atol=dicotomy_tol)

    Ap = multiplicative_step_h(X, G, P, A, force_simplex=True, mu=0, log_shift=0, epsilon_reg=1, safe=True, l2=True)
    np.testing.assert_allclose(A, Ap, atol=dicotomy_tol)       

    for _ in range(10):
        A = np.random.rand(k,p)
        A = A/np.sum(A, axis=1, keepdims=True)
        Ap =  multiplicative_step_h(X, G, P, A, force_simplex=False, mu=0, log_shift=0, epsilon_reg=1, safe=True)
        val1 = KLdiv_loss(X, GP, A)
        val2 = KLdiv_loss(X, GP, Ap)
        np.testing.assert_array_less(0, Ap)
        assert(val1 > val2)

        Ap =  multiplicative_step_h(X, G, P, A, force_simplex=True, mu=0, log_shift=log_shift, epsilon_reg=1, safe=True)
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
        Ap =  multiplicative_step_h(X, G, P, A, force_simplex=True, mu=mu, log_shift=log_shift, epsilon_reg=epsilon_reg, safe=True)
        Ap2 =  make_step_a(X, G, P, A, mu_sparse=1, eps=log_shift, eps_sparse=epsilon_reg, mask=None)
        np.testing.assert_array_almost_equal(Ap2, Ap)
        np.testing.assert_allclose(np.sum(Ap, axis=0), np.ones([Ap.shape[1]]), atol=dicotomy_tol)


        Ap =  multiplicative_step_h(X, G, P, A, force_simplex=True, mu=3*mu, log_shift=log_shift, epsilon_reg=epsilon_reg, safe=True)
        Ap2 =  make_step_a(X, G, P, A, mu_sparse=3, eps=log_shift, eps_sparse=epsilon_reg, mask=None)
        np.testing.assert_array_almost_equal(Ap2, Ap)
        np.testing.assert_allclose(np.sum(Ap, axis=0), np.ones([Ap.shape[1]]), atol=dicotomy_tol)


        val1 = KLdiv_loss(X, GP, A) + log_reg(A, 3*mu, epsilon_reg)
        val2 = KLdiv_loss(X, GP, Ap) + log_reg(A, 3*mu, epsilon_reg)
        np.testing.assert_array_less(0, Ap)
        assert(val1 > val2)

        # Ap =  multiplicative_step_h(X, G, P, A, force_simplex=True, mu=3, eps=log_shift, epsilon_reg=1, safe=True)
        # Ap2 =  make_step_a(X, G, P, A, mu_sparse=3, eps=log_shift, eps_sparse=1, mask=np.zeros([k])>0)
        # np.testing.assert_array_almost_equal(Ap2, Ap)
        # np.testing.assert_allclose(np.sum(Ap, axis=0), np.ones([Ap.shape[1]]), atol=dicotomy_tol)

    for _ in range(10):
        A = np.random.rand(k,p)
        A = A/np.sum(A, axis=1, keepdims=True)
        Ap =  multiplicative_step_h(X, G, P, A, force_simplex=False, mu=0, log_shift=0, epsilon_reg=1, safe=True, l2=True)
        val1 = Frobenius_loss(X, GP, A)
        val2 = Frobenius_loss(X, GP, Ap)
        np.testing.assert_array_less(0, Ap)
        assert(val1 > val2)

        Ap =  multiplicative_step_h(X, G, P, A, force_simplex=True, mu=0, log_shift=log_shift, epsilon_reg=1, safe=True, l2=True)
        np.testing.assert_allclose(np.sum(Ap, axis=0), np.ones([Ap.shape[1]]), atol=dicotomy_tol)

        val1 = Frobenius_loss(X, GP, A)
        val2 = Frobenius_loss(X, GP, Ap)
        np.testing.assert_array_less(0, Ap)
        assert(val1 > val2)

def test_multiplicative_step_hq():

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

    Ap = multiplicative_step_hq(X, G, P, A, force_simplex=False, log_shift=0,safe=True)
    np.testing.assert_array_almost_equal(A, Ap)
    np.testing.assert_allclose(np.sum(Ap, axis=0), np.ones([Ap.shape[1]]), atol=dicotomy_tol)

    Ap = multiplicative_step_hq(X, G, P, A, force_simplex=True, log_shift=0, safe=True)
    np.testing.assert_allclose(A, Ap, atol=dicotomy_tol)        

    for _ in range(10):
        A = np.random.rand(k,p)
        A = A/np.sum(A, axis=1, keepdims=True)
        Ap =  multiplicative_step_hq(X, G, P, A, force_simplex=False,  log_shift=0, safe=True)
        val1 = KLdiv_loss(X, GP, A)
        val2 = KLdiv_loss(X, GP, Ap)
        np.testing.assert_array_less(0, Ap)
        assert(val1 > val2)

        Ap =  multiplicative_step_hq(X, G, P, A, force_simplex=True, log_shift=log_shift, safe=True)
        Ap2 =  make_step_a(X, G, P, A, mu_sparse=0, eps=log_shift, eps_sparse=1, mask=None)
        np.testing.assert_array_almost_equal(Ap2, Ap)
        np.testing.assert_allclose(np.sum(Ap, axis=0), np.ones([Ap.shape[1]]), atol=dicotomy_tol)

        val1 = KLdiv_loss(X, GP, A)
        val2 = KLdiv_loss(X, GP, Ap)
        np.testing.assert_array_less(0, Ap)
        assert(val1 > val2)




def test_Q_step():
    l = 4
    p = 5
    i = 3

    D = np.random.randn(l,i)
    A = np.random.randn(i,p)

    Q = np.zeros([l,p,i])
    for a in range(l):
        for b in range(p):
            for c in range(i):
                Q[a,b,c] = A[c,b] * D[a,c] / np.sum(A[:,b] * D[a,:])
    Q2 = update_q(D, A)
    np.testing.assert_array_almost_equal(Q, Q2)

def test_Q_step2():
    l = 4
    p = 5
    i = 3

    D = np.random.randn(l,i)
    A = np.random.randn(i,p)
    A[0,:] = 0
    A[:,0] = 0
    D[0,:] = 0
    D[:,0] = 0

    Q = np.zeros([l,p,i])
    for a in range(l):
        for b in range(p):
            for c in range(i):
                Q[a,b,c] = A[c,b] * D[a,c] / (np.sum(A[:,b] * D[a,:]) + log_shift)
    Q2 = update_q(D, A)
    np.testing.assert_array_almost_equal(Q, Q2)
    assert((np.isnan(Q)==False).all())

def create_toy_problem(l = 25, k = 3, p = 100, c = 10, n_poisson=200, force_simplex=True):

    A = np.random.rand(k,p)
    if force_simplex:
        A = A/np.sum(A, axis=0, keepdims=True)
    
    G = np.random.rand(l,c)
    P = np.random.rand(c,k)
    GP = G @ P

    X = GP @ A

    Xdot = 1/n_poisson * np.random.poisson(n_poisson * X)

    return G, P, A, X, Xdot

def test_proj_step_h():


    def full_loss(X, G, W, H, log_shift, mu, lambda_L, L):
        return KLdiv_loss(X, G@W, H, log_shift=log_shift) + lambda_L * trace_xtLx(L, H.T) + log_reg(H, mu)
        
    shape_2d = [10, 15]
    k = 5
    n_poisson = 200
    log_shift = 0.01
    epsilon_reg = 2
    L = create_laplacian_matrix(*shape_2d)

    for _ in range(10):

        G, W, H, Xtrue, X = create_toy_problem(p = shape_2d[0]*shape_2d[1], k=k, n_poisson=n_poisson)

        true_D = G @ W
        true_H = H

        for force_simplex in [True, False]:
            for lambda_L in [0, 1, 10]:
                for mu in [0, 1 ,10]:

                    W0 = np.random.rand(*W.shape)
                    H0 = np.random.rand(*H.shape)
                    W0 = np.maximum(W0, log_shift)
                    H0 = np.maximum(H0, log_shift)
                    X = np.maximum(X, log_shift)


                    gamma_h = estimate_Lipschitz_bound_h(log_shift, X, G, k, mu=mu, lambda_L=lambda_L, epsilon_reg=epsilon_reg)

                    loss1 = full_loss(X, G, W0, H0, log_shift, mu, lambda_L, L)
                    
                    grad = gradH(X, G, W0, H0, log_shift=log_shift, l2=False, mu=mu, lambda_L=lambda_L, L=L, epsilon_reg=epsilon_reg)
                    H1 = H0 - 1/gamma_h * grad
                    loss2 = full_loss(X, G, W0, H1, log_shift, mu, lambda_L, L)
                    H2 =proj_grad_step_h(X, G, W0, H0, gamma_h, log_shift=log_shift, safe=True, l2=False, force_simplex=force_simplex, mu=mu, lambda_L=lambda_L,  L=L, epsilon_reg=epsilon_reg)
                    loss3 = full_loss(X, G, W0, H2, log_shift, mu, lambda_L, L)
                    assert( loss1 >= loss2) 
                    assert(loss1 >= loss3)
                    assert((H2 >= log_shift).all())

                    maxit = 10
                    loss_old = loss3
                    grad = gradH(X, G, W0, H2, log_shift=log_shift, l2=False)
                    n_grad_old = np.sum(grad**2)
                    for _ in range(maxit):
                        H2 = proj_grad_step_h(X, G, W0, H2, gamma_h, log_shift=log_shift, safe=True, l2=False, force_simplex=force_simplex, mu=mu, lambda_L=lambda_L,  L=L, epsilon_reg=epsilon_reg)
                        grad = gradH(X, G, W0, H2, log_shift=log_shift,  mu=mu, lambda_L=lambda_L,  L=L, l2=False)
                        n_grad = np.sum(grad**2)
                        loss = full_loss(X, G, W0, H2, log_shift, mu, lambda_L, L)
                        assert loss<=loss_old+np.abs(loss_old)*0.00001
                        # assert n_grad<=n_grad_old
                        loss_old = loss
                        n_grad_old = n_grad

def test_proj_step_w():
    for _ in range(10):
        shape_2d = [10, 15]
        log_shift = 0.01

        k = 5
        n_poisson = 200
        G, W, H, Xtrue, X = create_toy_problem(p = shape_2d[0]*shape_2d[1], k=k, n_poisson=n_poisson)

        true_D = G @ W
        true_H = H

        W0 = np.random.rand(*W.shape)
        H0 = np.random.rand(*H.shape)


        gamma_w = estimate_Lipschitz_bound_w(log_shift, X, G, k)

        loss1 = KLdiv_loss(X, G@W0, H0, log_shift=log_shift)
        grad = gradW(X, G, W0, H0, log_shift=log_shift, l2=False)
        W1 = W0 - 1/gamma_w * grad
        loss2 = KLdiv_loss(X, G@W1, H0, log_shift=log_shift)
        W2 =proj_grad_step_w(X, G, W0, H0, gamma_w, log_shift=log_shift, safe=True, l2=False)
        loss3 = KLdiv_loss(X, G@W2, H0, log_shift=log_shift)
        assert( loss1 >= loss2) 
        assert(loss1 >= loss3)
        assert((W2 >= log_shift).all())

        maxit = 10
        loss_old = loss3
        grad = gradW(X, G, W2, H0, log_shift=log_shift, l2=False)
        n_grad_old = np.sum(grad**2)
        for _ in range(maxit):
            W2 = proj_grad_step_w(X, G, W2, H0, gamma_w, log_shift=log_shift, safe=True, l2=False)
            grad = gradW(X, G, W2, H0, log_shift=log_shift, l2=False)

            n_grad = np.sum(grad**2)
            loss = KLdiv_loss(X, G@W2, H0, log_shift=log_shift)
            assert loss<=loss_old
            assert n_grad<=n_grad_old
            loss_old = loss
            n_grad_old = n_grad



import numpy as np
from numpy.lib.function_base import kaiser
from snmfem.measures import mse, spectral_angle, KLdiv_loss, KLdiv, square_distance, find_min_MSE, find_min_angle, trace_xtLx, Frobenius_loss
import pytest
from snmfem.conf import log_shift
from snmfem.laplacian import create_laplacian_matrix

def base_loss(x_matr, d_matr, a_matr, eps=log_shift):
    """
    Evaluates the data-fitting part of the loss function.
    The d_matr argument is used for evaluation of temporary values of the d_matr that are not stored in self.d_matr
    """
    d_a = d_matr @ a_matr
    x_log = np.einsum("ij,ij", x_matr, np.log(d_a+eps))
    return d_a.sum() - x_log

def MSE_map(map1, map2):
    """
    Calculates the mean squared error between two 2D arrays. They have to have the same dimension.
    :map1: first array (np.array 2D)
    :map2: second array (np.array 2D)
    """
    tr_m1_m1 = np.einsum("ij,ij->", map1, map1)
    tr_m2_m2 = np.einsum("ij,ij->", map2, map2)
    tr_m1_m2 = np.trace(map1.T @ map2)
    return tr_m1_m1 - 2 * tr_m1_m2 + tr_m2_m2

def spectral_angle_simple(v1, v2):
    """
    Calculates the angle between two spectra. They have to have the same dimension.
    :v1: first spectrum (np.array 1D)
    :v2: second spectrum (np.array 1D)
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / np.pi

def test_mse():

    a = np.random.randn(10, 34)
    b = np.random.randn(10, 34)

    np.testing.assert_allclose(mse(a, b), MSE_map(a, b))

def test_distance () : 
    a = np.random.randn(10, 34)
    b = np.random.randn(10, 34)

    d_matr = []
    for a_vec in a :
        d_vec = []
        for b_vec in b :
            d_vec.append(mse(b_vec,a_vec))
        d_matr.append(d_vec)
    d_matr = np.array(d_matr)
    d_matr_2 = square_distance(a,b)
    np.testing.assert_allclose(d_matr,d_matr_2)

    d_matr_c = d_matr.copy()

    unique_mins= []
    unique_inds = []
    for vec in d_matr_c :
        unique_mins.append(np.min(vec))
        ind_min = np.argmin(vec)
        unique_inds.append(ind_min)
        d_matr_c[:,ind_min] = np.inf * np.ones(d_matr_c.shape[0])

    unique_mins = np.array(unique_mins)
    unique_inds = np.array(unique_inds)
    unique_mins_2 = find_min_MSE(a,b,unique=True) 
    _, unique_inds_2 = find_min_MSE(a,b,get_ind=True,unique=True)
    np.testing.assert_allclose(unique_mins,unique_mins_2)
    np.testing.assert_allclose(unique_inds, unique_inds_2)

    global_mins= []
    global_inds = []
    for vec in d_matr :
        global_mins.append(np.min(vec))
        global_inds.append(np.argmin(vec))

    global_mins = np.array(global_mins)
    global_inds = np.array(global_inds)
    global_mins_2 = find_min_MSE(a,b) 
    _, global_inds_2 = find_min_MSE(a,b,get_ind=True)
    np.testing.assert_allclose(global_mins,global_mins_2)
    np.testing.assert_allclose(global_inds,global_inds_2)





def test_spectral_angle():
    v1 = np.random.randn(10)
    v2 = np.random.randn(10)

    res1 = spectral_angle_simple(v1, v2)
    res2 = spectral_angle(v1, v2)

    np.testing.assert_allclose(res1, res2)

    np.testing.assert_allclose(spectral_angle(v1, v1), 0, atol=1e-5)

    v1 = np.array([1,0])
    v2 = np.array([0,1])

    np.testing.assert_allclose(spectral_angle(v1, v2), 90)


    v1 = np.array([1,0])
    v2 = np.array([-1,0])

    np.testing.assert_allclose(spectral_angle(v1, v2), 180)

    a = np.random.randn(5, 40)
    b = np.random.randn(5, 40)

    res = []
    for v1 in a:
        r = []
        for v2 in b:
            r.append(spectral_angle(v1, v2))
        res.append(np.array(r))
    res = np.array(res)

    np.testing.assert_allclose(spectral_angle(a, b), res)

    res_c = res.copy()

    unique_mins= []
    unique_inds = []
    for vec in res_c :
        unique_mins.append(np.min(vec))
        ind_min = np.argmin(vec)
        unique_inds.append(ind_min)
        res_c[:,ind_min] = np.inf * np.ones(res_c.shape[0])

    unique_mins = np.array(unique_mins)
    unique_inds = np.array(unique_inds)
    unique_mins_2 = find_min_angle(a,b,unique=True) 
    _, unique_inds_2 = find_min_angle(a,b,get_ind=True,unique=True)
    np.testing.assert_allclose(unique_mins,unique_mins_2)
    np.testing.assert_allclose(unique_inds,unique_inds_2)

    global_mins= []
    global_inds = []
    for vec in res :
        global_mins.append(np.min(vec))
        global_inds.append(np.argmin(vec))

    global_mins = np.array(global_mins)
    global_inds = np.array(global_inds)
    global_mins_2 = find_min_angle(a,b) 
    _, global_inds_2 = find_min_angle(a,b,get_ind=True)
    np.testing.assert_allclose(global_mins,global_mins_2)
    np.testing.assert_allclose(global_inds,global_inds_2)

def test_base_loss():
    l = 26
    k = 5
    p = 100

    A = np.random.rand(k,p)
    A = A/np.sum(A, axis=1, keepdims=1)
    
    D = np.random.rand(l,k)
    
    X = D @ A
    
    val = np.sum(X) - np.sum(X*np.log(X+log_shift))

    val2 = KLdiv_loss(X, D, A)
    val3 = base_loss(X, D, A)
    np.testing.assert_almost_equal(val3, val2)
    np.testing.assert_almost_equal(val, val2)

    np.testing.assert_almost_equal(KLdiv(X, D, A, average=True), 0)
    np.testing.assert_almost_equal(KLdiv(X, D, A), 0)

    # with a different value, the divergence should be bigger
    A = np.random.rand(k,p)
    A = A/np.sum(A, axis=1, keepdims=1)
    val2 = KLdiv_loss(X, D, A)
    assert(val2>val)
    val3 = KLdiv(X, D, A)
    np.testing.assert_almost_equal(val2 -val, val3 )


    A = np.random.rand(k,p)
    val2 = KLdiv_loss(X, D, A)
    assert(val2>val)
    
    val3 = KLdiv(X, D, A, average=False)
    val4 = KLdiv(X, D, A, average=True)
    np.testing.assert_almost_equal(val3, val4*X.shape[0]*X.shape[1] )

    A = 0.00001*np.random.rand(k,p)
    val2 = KLdiv_loss(X, D, A)
    assert(val2>val)

    A = np.zeros([k,p])
    val2 = KLdiv_loss(X, D, A)
    assert(val2>val)

    with pytest.raises(Exception):
        D = np.random.rand(l,k)
        A = np.random.randn(k,p)
        val2 = KLdiv_loss(X, D, A)

    with pytest.raises(Exception):
        A = np.random.rand(k,p)
        D = np.random.randn(l,k)
        val2 = KLdiv_loss(X, D, A)

def test_trace_xtLx():
    nx = 4
    ny = 6
    L = create_laplacian_matrix(nx, ny)
    x = np.ones([nx, ny]).flatten()
    np.testing.assert_allclose(trace_xtLx(L, x), 0)

    x = np.ones([nx, ny])
    x[2,3] = 2
    np.testing.assert_allclose(trace_xtLx(L, x.flatten()), 4)
    
    k = 10
    x = np.random.randn(nx*ny, k)
    r1 = trace_xtLx(L, x)
    r2 = 0
    for i in range(k):
        r2 = r2 + trace_xtLx(L, x[:,i])
    np.testing.assert_allclose(r1, r2)
    np.testing.assert_allclose(trace_xtLx(L, x), np.sum(np.diag(x.T @ (L @ x))))

    r3 = trace_xtLx(L, x, average=True)

    np.testing.assert_allclose(r1,r3*k*nx*ny)
    
def test_froebenius_loss():
    l = 26
    k = 5
    p = 100

    A = np.random.rand(k,p)
    A = A/np.sum(A, axis=1, keepdims=1)
    
    D = np.random.rand(l,k)
    
    X = D @ A
    
    np.testing.assert_almost_equal(Frobenius_loss(X, D, A), 0)

    D2 = np.random.rand(l,k)
    res1 = np.linalg.norm(D2 @ A -X,"fro")**2   
    res2 = Frobenius_loss(X, D2, A)

    np.testing.assert_almost_equal(res1, res2)

    res1 = Frobenius_loss(X, D2, A)
    res2 = Frobenius_loss(X, D2, A, average=True)
    np.testing.assert_almost_equal(res1/l/p, res2)

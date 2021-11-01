import numpy as np
from snmfem.conf import log_shift, dicotomy_tol
from sklearn.decomposition._nmf import _initialize_nmf as initialize_nmf 
from snmfem.laplacian import sigmaL

def dichotomy_simplex(num, denum, tol=dicotomy_tol, maxit=40):
    """
    Function to solve the num/(x+denum) -1 = 0 equation. Here, x is the Lagragian multiplier which is used to apply the simplex constraint.
    The first part consists in finding a and b such that num/(a+denum) -1 > 0 and num/(b+denum) -1  < 0. 
    The second part applies the dichotomy algorithm to solve the equation.
    """
    # The function has exactly one root at the right of the first singularity (the singularity at min(denum))
    
    # num = num.astype("float64")
    # denum = denum.astype("float64")

    # do some test

    assert((num>=0).all())
    assert((denum>=0).all())
    assert((np.sum(num, axis=0)>0).all())
    
    # Ideally we want to do this, but we have to exclude the case where num==0.
    # a = np.max(num/2 - denum, axis=0)
    if denum.shape[1]>1:
        a = []
        for n,d in zip(num.T, denum.T):
            m = n>0
            a.append(np.max(n[m]/2 - d[m]))
        a = np.array(a)
    else:
        d = denum[:,0]
        def max_masked(n):
            m = n>0
            return np.max(n[m]/2-d[m])
        a = np.apply_along_axis(max_masked, 0, num)
        
        
    # r = np.sum(num/denum, axis=0)
    # b = np.zeros(r.shape)
    # b[r>=1] = (len(num) * np.max(num, axis=0)/0.5 - np.min(denum, axis=0))[r>=1]
    
    b = len(num) * np.max(num, axis=0)/0.5 - np.min(denum, axis=0)

    def func(x):
        return np.sum(num / (x + denum), axis=0) - 1
    func_a = func(a)
    func_b = func(b)
    
    assert(np.sum(func_b>=0)==0)
    assert(np.sum(func_a<=0)==0)
    assert(np.sum(np.isnan(func_a))==0)
    assert(np.sum(np.isnan(func_b))==0)


    return dicotomy(a, b, func, maxit, tol)

def dichotomy_simplex_aq(a, b, minus_c, tol=dicotomy_tol, maxit=100):
    """
    Function to solve the dicotomy for the function:
    f(nu) = n_p * nu_k + 2a - sum_p sqrt ( (b_p + nu)^2 - 4 a c_p) + sum_p b_p

    The first part consists in finding nu_max and nu_min such that f(nu_max) > 0 and f(nu_min) < 0. 
    The second part applies the dichotomy algorithm to solve the equation.
    """
    # do some test
    assert(a>=0)
    assert((minus_c>=0).all())

    n_p = len(b) 
    nu_max = n_p * np.max(b**2/a+2*a+2*(b+ minus_c), axis=0) * 1.5 + 1e-3

    nu_min = - (2 * a + np.sum(b, axis=0))/ n_p  * 1.1 - 1e-3
    
    def func(x):
        return n_p * x + 2*a + np.sum( - np.sqrt( (b + x)**2 + 4*a*minus_c) + b, axis=0)
    func_max = func(nu_max)
    func_min = func(nu_min)
    
    assert(np.sum(func_min>=0)==0)
    assert(np.sum(func_max<=0)==0)
    assert(np.sum(np.isnan(func_max))==0)
    assert(np.sum(np.isnan(func_min))==0)

    return dicotomy(nu_max, nu_min, func, maxit, tol)     

def dicotomy(a, b, func, maxit, tol):
    """
    Dicotomy algorithm searching for func(x)=0.

    Inputs:
    * a: bound such that func(a) > 0
    * b: bound such that func(b) < 0
    * maxit: maximum number of iteration
    * tol: tolerance - the algorithm stops if |func(sol)| < tol
    
    This algorithm work for number or numpy array of any size.
    """
    # Dichotomy algorithm to solve the equation
    it = 0
    new = (a + b)/2
    func_new = func(new)
    while np.max(np.abs(func_new)) > tol:
        
        it=it+1
        func_a = func(a)
        # func_b = func(b)

        # if f(a)*f(new) <0 then f(new) < 0 --> store in b
        minus_bool = func_a * func_new <= 0
        
        # if f(a)*f(new) > 0 then f(new) > 0 --> store in a
        # plus_bool = func_a * func_new > 0
        plus_bool = np.logical_not(minus_bool)

        b[minus_bool] = new[minus_bool]
        a[plus_bool] = new[plus_bool]
        new = (a + b) / 2
        func_new = func(new)
        if it>=maxit:
            print("Dicotomy stopped for maximum number of iterations")
            break
    return new

def multiplicative_step_p(X, G, P, A, eps=log_shift, safe=True, l2=False, fixed_P = None):
    """
    Multiplicative step in P.
    """

    if safe:
        # Allow for very small negative values!
        assert(np.sum(A<-log_shift/2)==0)
        assert(np.sum(P<-log_shift/2)==0)
        assert(np.sum(G<-log_shift/2)==0)

    if l2:
        GG = G.T @ G
        AA = A @ A.T
        GGPAA = GG @ P @ AA

        GXA = G.T @ (X @ A.T)

        new_P = P / GGPAA * GXA
    else:
        GP = G @ P
        GPA = GP @ A
        # Split to debug timing...
        # term1 = G.T @ (X / (GPA + eps)) @ A.T
        op1 = X / (GPA + eps)
        
        mult1 = G.T @ op1
        term1 = (mult1 @ A.T)
        term2 = np.sum(G, axis=0,  keepdims=True).T @ np.sum(A, axis=1,  keepdims=True).T
        new_P = P / term2 * term1
    
    if fixed_P is None : 
        return new_P
    else : 
        new_P[fixed_P >= 0] = fixed_P[fixed_P >=0]
        return new_P

# import torch
# def multiplicative_step_p_torch(X, G, P, A, eps=log_shift):
#     """
#     Multiplicative step in P.
#     """

#     GP = G.matmul(P)
#     GPA = GP.matmul(A)
#     # Split to debug timing...
#     # term1 = G.T @ (X / (GPA + eps)) @ A.T
#     op1 = X / (GPA + eps)
    
#     mult1 = G.T.matmul(op1)
#     term1 = mult1.matmul(A.T)
#     term2 = (torch.sum(A, axis=1,  keepdims=True).matmul(torch.sum(G, axis=0,  keepdims=True))).T 
#     new_P = (P / term2 * term1)
#     return new_P

def multiplicative_step_a(X, G, P, A, force_simplex=True, mu=0, eps=log_shift, epsilon_reg=1, safe=True, dicotomy_tol=dicotomy_tol, lambda_L=0, L=None, l2=False, fixed_A_inds = None, sigmaL=sigmaL):
    """
    Multiplicative step in A.
    The main terms are calculated first.
    With mu_sparse = 0, the steps in A are calculated once. For mu_sparse != 0, the steps 
    in A are calculated first with particle regularization. Then only the entries allowed 
    by the mask are calculaed, without particle regularization. Note that mu can be passed
    as a vector to regularize the different phase of A differently.
    To calculate the regularized step, we make a linear approximation of the log.
    """
    if not(lambda_L==0):
        if L is None:
            raise ValueError("Please provide the laplacian")

    if not(fixed_A_inds is None) : 
        not_fixed_inds = [x for x in range(A.shape[1]) if not(x in fixed_A_inds)]    
        if not(lambda_L==0):
            AL = (A@L)[:,not_fixed_inds]
        new_A = A.copy()
        A = A[:,not_fixed_inds]
        X = X[:,not_fixed_inds]
    else : 
        if not(lambda_L==0):
            AL = A @ L

    if safe:
        # Allow for very small negative values!
        assert(np.sum(A<-log_shift/2)==0)
        assert(np.sum(P<-log_shift/2)==0)
        assert(np.sum(G<-log_shift/2)==0)

    GP = G @ P # Also called D
    
    if l2:
        PGGP = GP.T @ GP
        PGX = GP.T @ X
        num = A * PGX
        denum = PGGP @ A
    else:
        
        GPA = GP @ A
        # Split to debug timing...
        num = A * (GP.T @ (X / (GPA+eps)))
        # op1 = X / (GPA+eps)
        # op2 = GP.T @ op1
        # num = A * op2
        denum = np.sum(GP, axis=0, keepdims=True).T 

    if not(np.isscalar(mu) and mu==0):
        if len(np.shape(mu))==1:
            mu = np.expand_dims(mu, axis=1)
        denum = denum + mu / (A + epsilon_reg)
    if not(lambda_L==0):
        maxA = np.max(A)
        num = num + lambda_L * sigmaL * A * maxA
        denum = denum + lambda_L * sigmaL * maxA + lambda_L * AL 

    if force_simplex:
        nu = dichotomy_simplex(num, denum,dicotomy_tol)
    else:
        nu = 0
    if safe:
        assert np.sum(denum<0)==0
        assert np.sum(num<0)==0

    if fixed_A_inds is None : 
        new_A = num/(denum+nu)
    else : 
        new_A[:,not_fixed_inds] = num/(denum+nu)

    return new_A


def initialize_algorithms(X, G, P, A, n_components, init, random_state, force_simplex, fixed_A_inds = None):
    # Handle initialization

    if G is None : 
        skip_second = True
        # G = sparse.diags(np.ones(X.shape[0]).astype(X.dtype))        
        G = np.diag(np.ones(X.shape[0]).astype(X.dtype))

    # elif callable(G) : 
    #     assert not(model_params is None), "You need to input model_parameters"
    #     assert not(g_params is None), "You need to input g_parameters"
    #     G = G(model_params,g_params)
    #     skip_second = False

    else:
        skip_second = False

    if P is None:
        if A is None:
            D, A = initialize_nmf(X, n_components=n_components, init=init, random_state=random_state)
            # D, A = u.rescaled_DA(D,A)
            if force_simplex:
                scale = np.sum(A, axis=0, keepdims=True)
                A = np.nan_to_num(A/scale, nan = 1.0/A.shape[0] )
        D = np.abs(np.linalg.lstsq(A.T, X.T,rcond=None)[0].T)
        if skip_second:
            P = D
        else:
            # Divide in two parts the initial fitting, otherwise the bremsstrahlung (which has a low intensity) tends to be poorly learned
            # First fit the caracteristic Xrays, then subtract that contribution to obtain a rough estimate of the bremsstralung parameters
            Pcarac = (np.linalg.lstsq(G[:,:-2],D,rcond = None)[0]).clip(min = 0)
            Pbrem = (np.linalg.lstsq(G[:,-2:],D - G[:,:-2]@Pcarac,rcond = None)[0]).clip(min = 0)
            P = np.vstack((Pcarac, Pbrem))
            # P = np.abs(np.linalg.lstsq(G, D,rcond=None)[0])

    elif A is None:
        D = G @ P
        A = np.abs(np.linalg.lstsq(D, X, rcond=None)[0])
        if force_simplex:
            scale = np.sum(A, axis=0, keepdims=True)
            A = A/scale
    
    if not(fixed_A_inds is None) : 
        vec = np.zeros_like(A[:,0])
        vec[0] = 1
        fixed_A = np.tile(vec[:,np.newaxis],(len(fixed_A_inds),))
        A[:,fixed_A_inds] = fixed_A

    return G, P, A

def update_q(D, A, eps=log_shift):
    """Perform a Q step."""
    Atmp = np.expand_dims(A.T, axis=0)
    Dtmp = np.expand_dims(D, axis=1)
    Ntmp = np.expand_dims(D @ A, axis=2) 
    return Atmp * (Dtmp / (Ntmp+eps))
   
def multiplicative_step_pq(X, G, P, A, eps=log_shift, safe=True):
    """
    Multiplicative step in P using the PQ technique.

    This function does exactly the same as `multiplicative_step_p` and is probably slower.
    """

    if safe:
        # Allow for very small negative values!
        assert np.sum(A<-log_shift/2)==0
        assert np.sum(P<-log_shift/2)==0
        assert np.sum(G<-log_shift/2)==0

    GP = G @ P
    Q = update_q(GP, A, eps=log_shift)

    XQ = np.sum(np.expand_dims(X, axis=2) * Q, axis=1)

    term1 = G.T @ (XQ / (GP + eps)) 

    term2 = np.sum(G, axis=0,  keepdims=True).T @ np.sum(A, axis=1,  keepdims=True).T
    return P / term2 * term1

def multiplicative_step_aq(X, G, P, A, force_simplex=True, eps=log_shift, safe=True, dicotomy_tol=dicotomy_tol, lambda_L=0, L=None, sigmaL=sigmaL):
    """
    Multiplicative step in A.
    """
    if not lambda_L==0:
        if L is None:
            raise ValueError("Please provide the laplacian")

    if safe:
        # Allow for very small negative values!
        assert np.sum(A<-log_shift/2)==0
        assert np.sum(P<-log_shift/2)==0
        assert np.sum(G<-log_shift/2)==0

    GP = G @ P # Also called D
    GPA = GP @ A

    minus_c = A * (GP.T @ (X / (GPA+eps)))

    b = np.sum(GP, axis=0, keepdims=True).T 
    if not lambda_L==0 :
        b = b + lambda_L * A @ L - lambda_L * sigmaL  * A 
        a = lambda_L * sigmaL
        if force_simplex:
            nu = dichotomy_simplex_aq(a, b, minus_c)
            b = b + nu
        return (-b + np.sqrt(b**2 + 4* a *minus_c)) / (2*a)
    else: # We recover the classic case: multiplicative_step_a
        if force_simplex:
            nu = dichotomy_simplex(minus_c, b, dicotomy_tol)
            b = b + nu
        return minus_c / b

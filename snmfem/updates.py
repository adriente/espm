from re import A
import numpy as np
from snmfem.conf import log_shift, dicotomy_tol
from sklearn.decomposition._nmf import _initialize_nmf as initialize_nmf 
from snmfem.laplacian import sigmaL
import snmfem.utils as u
from scipy import sparse
# test

def dichotomy_simplex(num, denum, tol=dicotomy_tol, maxit=40):
    """
    Function to solve the num/(x+denum) -1 = 0 equation. Here, x is the Lagragian multiplier which is used to apply the simplex constraint.
    The first part consists in finding a and b such that num/(a+denum) -1 > 0 and num/(b+denum) -1  < 0. (line search)
    In this function, the largest root is found.
    The second part applies the dichotomy algorithm to solve the equation.
    In the future a vectorized version of Dekker Brent could be implemented.
    """
    # # The function has exactly one root at the right of the first singularity (the singularity at min(denum))
    num = num.astype("float64")
    denum = denum.astype("float64")

    # ind_min = np.argmax(num/denum, axis=0)
    # ind_min2 = np.argmin(denum, axis=0)
    # ind = np.arange(len(ind_min2))
    # amin1 = (num[ind_min, ind]/2-denum[ind_min, ind])
    # amin2 = (num[ind_min2, ind]/2- denum[ind_min2, ind])
    # a = np.maximum(amin1, amin2)
    alpha = 1
    i = 0
    a = np.max(num/(1+alpha) - denum, axis=0)
    while np.intersect1d(a,-denum).size > 0 :
        i+=1
        alpha/= 2
        a = np.max(num/(1+alpha) - denum, axis=0)
        if i ==20 :
            raise ValueError("Probably too many zeros in the data, the dichotomy will fail. Please retry with force_simplex = False")

    
    # r = np.sum(num/denum, axis=0)
    # b = np.zeros(r.shape)
    # b[r>=1] = (len(num) * np.max(num, axis=0)/0.5 - np.min(denum, axis=0))[r>=1]    
    b = (len(num) * np.max(num, axis=0)/0.5 - np.min(denum, axis=0))
    assert(np.sum((np.sum(num / (b + denum), axis=0) - 1)>=0)==0)
    assert(np.sum((np.sum(num / (a + denum), axis=0) - 1)<=0)==0)

    new = (a + b)/2

    # Dichotomy algorithm to solve the equation
    it = 0
    while (np.max(np.abs(np.sum(num / (new + denum), axis=0) - 1))) > tol:
        
        it=it+1
        # if f(a)*f(new) <0 then f(new) < 0 --> store in b
        minus_bool = (np.sum(num / (a + denum), axis=0) - 1) * (
            np.sum(num / (new + denum), axis=0) - 1
        ) <= 0
        # if f(a)*f(new) > 0 then f(new) > 0 --> store in a
        plus_bool = (np.sum(num / (a + denum), axis=0) - 1) * (
            np.sum(num / (new + denum), axis=0) - 1
        ) > 0

        b[minus_bool] = new[minus_bool]
        a[plus_bool] = new[plus_bool]
        new = (a + b) / 2
        if it>=maxit:
            print("Dicotomy stopped for maximum number of iterations")
            break
    return new


def multiplicative_step_p(X, G, P, A, eps=log_shift, safe=True, l2=False):
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

        return P / GGPAA * GXA
    else:
        GP = G @ P
        GPA = GP @ A
        # Split to debug timing...
        # term1 = G.T @ (X / (GPA + eps)) @ A.T
        op1 = X / (GPA + eps)
        
        mult1 = G.T @ op1
        term1 = (mult1 @ A.T)
        term2 = np.sum(G, axis=0,  keepdims=True).T @ np.sum(A, axis=1,  keepdims=True).T
        return P / term2 * term1

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


def multiplicative_step_a(X, G, P, A, force_simplex=True, mu=0, eps=log_shift, epsilon_reg=1, safe=True, dicotomy_tol=dicotomy_tol, lambda_L=0, L=None, l2=False):
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
        denum = denum + lambda_L * sigmaL * maxA + lambda_L * A @ L 

    if force_simplex:
        nu = dichotomy_simplex(num, denum,dicotomy_tol)
    else:
        nu = 0
    if safe:
        assert(np.sum(denum<0)==0)
        assert(np.sum(num<0)==0)

    return num/(denum + nu)


def initialize_algorithms(X, G, P, A, n_components, init, random_state, force_simplex):
    # Handle initialization
    if G is None : 
        skip_second = True
        # G = sparse.diags(np.ones(X.shape[0]).astype(X.dtype))        
        G = np.diag(np.ones(X.shape[0]).astype(X.dtype))
    else:
        skip_second = False

    if P is None:
        if A is None:
            D, A = initialize_nmf(X, n_components=n_components, init=init, random_state=random_state)
            # D, A = u.rescaled_DA(D,A)
            if force_simplex:
                scale = np.sum(A, axis=0, keepdims=True)
                A = A/scale 
        D = np.linalg.lstsq(A.T, X.T,rcond=None)[0].T
        if skip_second:
            P = D
        else:
            P = np.abs(np.linalg.lstsq(G, D,rcond=None)[0])

    elif A is None:
        D = G @ P
        A = np.abs(np.linalg.lstsq(D, X, rcond=None)[0])
        if force_simplex:
            scale = np.sum(A, axis=0, keepdims=True)
            A = A/scale
    return G, P, A

    # def initialize(self, x_matr):
    #     """
    #     Initialization of the data, matrices and parameters
    #     The data are flattened if necessary. The x-matr of SNMF has to be ExN, i.e. (number of energy channels) x (number of pixels).
    #     The a-matr is initialized at random unless init_a is specified.
    #     If a bremsstrahlung spectrum is specified, the b-matr update is deactivated (b_tol is set to 0) and B is set to 0.
    #     Otherwise, the b_matr is initialized through the values of c0, c1, c2, b1 and b2.
    #     The p-matr entries of the main phase are set through linear regression on the init_spectrum (if specified) or on the average spectrum.
    #     The other entries of the p_matr are initialized at random.
    #     """
    #     # Data pre-processing
    #     # store the original shape of the input data X
    #     self.x_shape = x_matr.shape
    #     # If necessary, flattens X to a Ex(NM) matrix, such that the columns hold the raw spectra
    #     if x_matr.ndim == 3:
    #         x_matr = x_matr.reshape(
    #             (self.x_shape[0] * self.x_shape[1], self.x_shape[2])
    #         ).T
    #         self.x_matr = x_matr.astype(np.float)
    #     else:
    #         self.x_matr = x_matr.astype(np.float)

    #     # Initialization of A
    #     if self.init_a is None:
    #         self.a_matr = np.random.rand(self.p_, self.x_matr.shape[1])
    #     else:
    #         self.a_matr = self.init_a

    #     # Initialization of B
    #     if self.bremsstrahlung:
    #         self.b_matr = np.zeros((self.g_matr.shape[0], self.p_))
    #         self.b_tol = 0
    #     else:
    #         # B is removed from the model

    #         self.b_matr = self.calc_b()

    #     # Initialization of p-matr
    #     # If the regularization is activated (mu_sparse != 0) it is important to correctly set the first phase so that the main phase is not penalized
    #     if self.init_p is None:
    #         # All phases are initialized at random and the first phase will be overwritten
    #         self.p_matr = np.random.rand(self.g_matr.shape[1], self.p_)
    #         # If a bremsstrahlung is specified, it is added to the g_matr and therefore should not be subtracted for the linear regression
    #         if self.bremsstrahlung:
    #             # Average spectrum initialization without b_matr model
    #             if self.init_spectrum is None:
    #                 avg_sp = np.average(self.x_matr, axis=1)
    #                 self.p_matr[:, 0] = (
    #                     np.linalg.inv(self.g_matr.T @ self.g_matr)
    #                     @ self.g_matr.T
    #                     @ avg_sp
    #                 ).clip(min=1e-8)
    #             # init_spectrum initialization without b_matr model
    #             else:
    #                 self.p_matr[:, 0] = (
    #                     np.linalg.inv(self.g_matr.T @ self.g_matr)
    #                     @ self.g_matr.T
    #                     @ self.init_spectrum
    #                 ).clip(min=1e-8)
    #         # If no bremsstrahlung spectrum is specified the b-matr is substracted from the linear regression to get correct values for p_matr (Linear regression on the gaussians only)
    #         else:
    #             # Average spectrum initialization with b_matr model
    #             if self.init_spectrum is None:
    #                 avg_sp = np.average(self.x_matr, axis=1)
    #                 self.p_matr[:, 0] = (
    #                     np.linalg.inv(self.g_matr.T @ self.g_matr)
    #                     @ self.g_matr.T
    #                     @ (avg_sp - self.b_matr[:, 0])
    #                 ).clip(min=1e-8)
    #             # init_spectrum initialization with b_matr model
    #             else:
    #                 self.p_matr[:, 0] = (
    #                     np.linalg.inv(self.g_matr.T @ self.g_matr)
    #                     @ self.g_matr.T
    #                     @ (self.init_spectrum - self.b_matr[:, 0])
    #                 ).clip(min=1e-8)
    #     else:
    #         # If specified the p_matr is used
    #         self.p_matr = self.init_p
    #     # The linear regression of p_matr are clipped to avoid negative values

    #     # Initalization of other internal variables.
    #     self.d_matr = self.g_matr @ self.p_matr + self.b_matr
    #     self.num_iterations = 0
    #     self.lambda_s = np.ones((self.x_matr.shape[1],))
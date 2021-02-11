import numpy as np
from snmfem.conf import log_shift, dicotomy_tol
from sklearn.decomposition._nmf import _initialize_nmf as initialize_nmf 

def dichotomy_simplex(num, denum, tol=dicotomy_tol):
    """
    Function to solve the num/(x+denum) -1 = 0 equation. Here, x is the Lagragian multiplier which is used to apply the simplex constraint.
    The first part consists in finding a and b such that num/(a+denum) -1 > 0 and num/(b+denum) -1  < 0. (line search)
    In this function, the largest root is found.
    The second part applies the dichotomy algorithm to solve the equation.
    In the future a vectorized version of Dekker Brent could be implemented.
    """
    # # The function has exactly one root at the right of the first singularity (the singularity at min(denum))
    # # So a and b are set to -min(denum) plus an offset.
    f_div = np.min(
        np.where(num != 0, denum, np.inf), axis=0
    )  # There are several min values in the case of particle regularization. In the case where there are exact 0 in num, the min of denum needs to be changed to the next nearest min.

    a_off = 100 * np.ones(num.shape[1])
    b_off = 0.01 * np.ones(num.shape[1])
    a = -f_div * np.ones(num.shape[1]) + a_off
    b = -f_div * np.ones(num.shape[1]) + b_off

    # r = np.sum(num/denum, axis=0)
    # ind_min = np.argmax(num/denum, axis=0)
    # ind_min2 = np.argmin(denum, axis=0)
    # ind = np.arange(len(ind_min))
    # bmin1 = num[ind_min, ind]-denum[ind_min, ind]
    # bmin2 = num[ind_min2, ind]-denum[ind_min2, ind]
    # a = r
    # b = np.maximum(bmin1, bmin2)
        
    # Search for a elements which give positive value
    constr = np.sum(num / (a + denum), axis=0) - 1
    while np.any(constr <= 0):
        # We exclude a elements which give positive values
        # We use <= because constr == 0 is problematic.
        constr_bool = constr <= 0
        # Reducing a will ensure that the function will diverge toward positive values
        a_off[constr_bool] /= 1.2
        a = -f_div * np.ones(num.shape[1]) + a_off
        constr = np.sum(num / (a + denum), axis=0) - 1

    # Search for b elements which give negative values
    constr = np.sum(num / (b + denum), axis=0) - 1
    while np.any(constr >= 0):
        # We exclude b elements which give negative values
        constr_bool = constr >= 0
        # increasing b will ensure that the function will converge towards negative values
        b_off[constr_bool] *= 1.2
        b = -f_div * np.ones(num.shape[1]) + b_off
        constr = np.sum(num / (b + denum), axis=0) - 1

    # Dichotomy algorithm to solve the equation
    while np.any(np.abs(b - a) > tol):
        new = (a + b) / 2
        # if f(a)*f(new) <0 then f(new) < 0 --> store in b
        minus_bool = (np.sum(num / (a + denum), axis=0) - 1) * (
            np.sum(num / (new + denum), axis=0) - 1
        ) < 0
        # if f(a)*f(new) > 0 then f(new) > 0 --> store in a
        plus_bool = (np.sum(num / (a + denum), axis=0) - 1) * (
            np.sum(num / (new + denum), axis=0) - 1
        ) > 0
        b[minus_bool] = new[minus_bool]
        a[plus_bool] = new[plus_bool]

    return (a + b) / 2


def multiplicative_step_p(X, G, P, A, eps=log_shift, safe=True):
    """
    Multiplicative step in P.
    """

    if safe:
        # Allow for very small negative values!
        assert(np.sum(A<-log_shift/2)==0)
        assert(np.sum(P<-log_shift/2)==0)
        assert(np.sum(G<-log_shift/2)==0)

    GP = G @ P
    GPA = GP @ A
    term1 = (G.T @ (X / (GPA + eps)) @ A.T)
    term2 = np.sum(G, axis=0,  keepdims=True).T @ np.sum(A, axis=1,  keepdims=True).T
    return P / term2 * term1



def multiplicative_step_a(X, G, P, A, force_simplex=True, mu=0, eps=log_shift, epsilon_reg=1, safe=True, dicotomy_tol=dicotomy_tol):
    """
    Multiplicative step in A.
    The main terms are calculated first.
    With mu_sparse = 0, the steps in A are calculated once. For mu_sparse != 0, the steps 
    in A are calculated first with particle regularization. Then only the entries allowed 
    by the mask are calculaed, without particle regularization. Note that mu can be passed
    as a vector to regularize the different phase of A differently.
    To calculate the regularized step, we make a linear approximation of the log.
    """

    if safe:
        # Allow for very small negative values!
        assert(np.sum(A<-log_shift/2)==0)
        assert(np.sum(P<-log_shift/2)==0)
        assert(np.sum(G<-log_shift/2)==0)

    GP = G @ P # Also called D
    GPA = GP @ A

    if not(np.isscalar(mu)) and len(np.shape(mu))==1:
        mu = np.expand_dims(mu, axis=1)

    num = A * (GP.T @ (X / (GPA+eps)))
    denum = np.sum(GP, axis=0, keepdims=True).T + mu / (A + epsilon_reg)

    if force_simplex:
        nu = dichotomy_simplex(num, denum,dicotomy_tol)
    else:
        nu = 0
    
    return num/(denum + nu)


def initialize_algorithms(X, G, P, A, n_components, init, random_state, force_simplex):
    # Handle initialization
    if P is None:
        if A is None:
            D, A = initialize_nmf(X, n_components=n_components, init=init, random_state=random_state)
            if force_simplex:
                scale = np.sum(A, axis=1, keepdims=True)
                A = A/scale
        
        D = np.linalg.lstsq(A.T, X.T)[0].T
        P = np.abs(np.linalg.lstsq(G, D)[0])

    elif A is None:
        D = G @ P
        A = np.abs(np.linalg.lstsq(D, X)[0])
        if force_simplex:
            scale = np.sum(A, axis=1, keepdims=True)
            A = A/scale
    return P, A

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
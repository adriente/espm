import numpy as np

from snmfem.updates import multiplicative_step_a, multiplicative_step_p, multiplicative_step_aq
from snmfem.measures import trace_xtLx
from snmfem.estimators import NMF
from snmfem.conf import log_shift, dicotomy_tol
from snmfem.laplacian import sigmaL

def smooth_l2_surrogate(At, L, A=None, sigmaL=sigmaL, lambda_L=1):
    tmp = (At @ L)
    t1 = np.sum(At * tmp)
    if not(A is None):
        t2 = np.sum(A * tmp)
        t3 = np.sum((At-A)**2)
    else:
        t2 = t1
        t3 = 0
    return lambda_L / 2 * (2*t2 -t1 + sigmaL * t3)

def diff_surrogate(At, A, L, sigmaL=sigmaL, lambda_L=1):
    b_inf = trace_xtLx(L, A.T) * lambda_L / 2
    b_supp = smooth_l2_surrogate(At, L=L, A=A, sigmaL=sigmaL, lambda_L=lambda_L)
    return b_supp - b_inf

class SmoothNMF(NMF):

    loss_names_ = NMF.loss_names_ + ["Lapl_reg_loss"]

    # args and kwargs are copied from the init to the super instead of capturing them in *args and **kwargs to be scikit-learn compliant.
    def __init__(self, lambda_L = 1.0, accelerate=False, linesearch=False, **kwargs):

        super().__init__( **kwargs)
        self.accelerate = accelerate
        self.lambda_L = lambda_L
        self.linesearch = linesearch
        if self.accelerate:
            assert np.max(np.array(self.mu))==0
            assert not self.l2
            self.sigmaL_ = sigmaL
            if self.linesearch:
                self.gamma = [self.sigmaL_]
                



    def _iteration(self, P, A):
        if self.accelerate:
            if self.linesearch:
                Aold = A.copy()            
            A = multiplicative_step_aq(self.X_, self.G_, P, A, force_simplex=self.force_simplex, eps=self.log_shift, safe=self.debug, dicotomy_tol=self.dicotomy_tol, lambda_L=self.lambda_L, L=self.L_, sigmaL=self.sigmaL_)
            P = multiplicative_step_p(self.X_, self.G_, P, A, eps=self.log_shift, safe=self.debug, l2=self.l2, fixed_P=self.fixed_P)
            if self.linesearch:
                d = diff_surrogate(Aold, A, L=self.L_, sigmaL=self.sigmaL_ )
                if d>0:
                    self.sigmaL_  = self.sigmaL_ / 1.2
                else:
                    self.sigmaL_  = self.sigmaL_ * 1.5
                self.gamma.append(self.sigmaL_ )
        else:
            A = multiplicative_step_a(self.X_, self.G_, P, A, force_simplex=self.force_simplex, mu=self.mu, eps=self.log_shift, epsilon_reg=self.epsilon_reg, safe=self.debug, dicotomy_tol=self.dicotomy_tol, lambda_L=self.lambda_L, L=self.L_, l2=self.l2, fixed_A_inds=self.fixed_A_inds)
            P = multiplicative_step_p(self.X_, self.G_, P, A, eps=self.log_shift, safe=self.debug, l2=self.l2)

        if callable(self.G) : 
            self.G_ = self.G(part_P = P[:-2,:],G = self.G_)
        return  P, A

    def loss(self, P, A, average=True, X = None):
        l1 = super().loss(P, A, average=average, X = X)
        l2 = self.lambda_L * trace_xtLx(self.L_, A.T, average=False)
        if average:
            l2 = l2 / self.GPA_numel_
    
        self.detailed_loss_.append(l2)
        return l1 + l2
    
    # def fit_transform(self, X, y=None, G=None, P=None, A=None, shape_2d=None, eval_print=10, true_D=None, true_A=None):
    #     self.shape_2d_ = shape_2d
    #     if not(self.shape_2d_ is None) :
    #         self.L_ = create_laplacian_matrix(*self.shape_2d_).astype(X.dtype)
    #     else : 
    #         self.L_ = np.diag(np.ones(X.shape[1])).astype(X.dtype)
    #     GP = super().fit_transform(X, y=None, G=G, P=P, A=A, shape_2d=shape_2d, eval_print=eval_print, true_D=true_D, true_A=true_A)
    #     return GP




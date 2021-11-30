import numpy as np

from esmpy.updates import multiplicative_step_h, multiplicative_step_w, multiplicative_step_hq
from esmpy.measures import trace_xtLx
from esmpy.estimators import NMF
from esmpy.conf import log_shift, dicotomy_tol
from esmpy.laplacian import sigmaL

def smooth_l2_surrogate(Ht, L, H=None, sigmaL=sigmaL, lambda_L=1):
    tmp = (Ht @ L)
    t1 = np.sum(Ht * tmp)
    if not(H is None):
        t2 = np.sum(H * tmp)
        t3 = np.sum((Ht-H)**2)
    else:
        t2 = t1
        t3 = 0
    return lambda_L / 2 * (2*t2 -t1 + sigmaL * t3)

def diff_surrogate(Ht, H, L, sigmaL=sigmaL, lambda_L=1):
    b_inf = trace_xtLx(L, H.T) * lambda_L / 2
    b_supp = smooth_l2_surrogate(Ht, L=L, H=H, sigmaL=sigmaL, lambda_L=lambda_L)
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
                



    def _iteration(self, W, H):
        if self.accelerate:
            if self.linesearch:
                Hold = H.copy()            
            H = multiplicative_step_hq(self.X_, self.G_, W, H, force_simplex=self.force_simplex, eps=self.log_shift, safe=self.debug, dicotomy_tol=self.dicotomy_tol, lambda_L=self.lambda_L, L=self.L_, sigmaL=self.sigmaL_)
            W = multiplicative_step_w(self.X_, self.G_, W, H, eps=self.log_shift, safe=self.debug, l2=self.l2, fixed_W=self.fixed_W)
            if self.linesearch:
                d = diff_surrogate(Hold, H, L=self.L_, sigmaL=self.sigmaL_ )
                if d>0:
                    self.sigmaL_  = self.sigmaL_ / 1.2
                else:
                    self.sigmaL_  = self.sigmaL_ * 1.5
                self.gamma.append(self.sigmaL_ )
        else:
            H = multiplicative_step_h(self.X_, self.G_, W, H, force_simplex=self.force_simplex, mu=self.mu, eps=self.log_shift, epsilon_reg=self.epsilon_reg, safe=self.debug, dicotomy_tol=self.dicotomy_tol, lambda_L=self.lambda_L, L=self.L_, l2=self.l2, fixed_H_inds=self.fixed_H_inds)
            W = multiplicative_step_w(self.X_, self.G_, W, H, eps=self.log_shift, safe=self.debug, l2=self.l2,fixed_W=self.fixed_W)

        if callable(self.G) : 
            self.G_ = self.G(part_W = W[:-2,:],G = self.G_)
        return  W, H

    def loss(self, W, H, average=True, X = None):
        l1 = super().loss(W, H, average=average, X = X)
        l2 = self.lambda_L * trace_xtLx(self.L_, H.T, average=False)
        if average:
            l2 = l2 / self.GWH_numel_
    
        self.detailed_loss_.append(l2)
        return l1 + l2
    

import numpy as np

from esmpy.updates import multiplicative_step_h, multiplicative_step_w, multiplicative_step_hq, multiplicative_step_w_checkerboard
from esmpy.measures import trace_xtLx, log_reg
from esmpy.estimators import NMFEstimator
from esmpy.laplacian import sigmaL
from esmpy.conf import dicotomy_tol

# from esmpy.measures import KL_loss_surrogate, KLdiv_loss, log_reg, log_surrogate


def smooth_l2_surrogate(Ht, L, H=None, sigmaL=sigmaL, lambda_L=1):
    HtTL = Ht @ L
    t1 = np.sum(HtTL * Ht)
    if H is None:
        t2 = t1
        t3 = 0
    else:
        t2 = np.sum(HtTL * H )
        t3 = np.sum((Ht-H)**2)
    return lambda_L / 2 * (2*t2 - t1 + sigmaL * t3)

def smooth_dgkl_surrogate(Ht, L, H=None, sigmaL=sigmaL, lambda_L=1):
    HtTL = Ht @ L
    t1 = np.sum(HtTL * Ht)
    
    def dgkl(p, q):
        return p * np.log(p/q) - p + q
    
    if H is None:
        t2 = t1
        t3 = 0
    else:
        t2 = np.sum(HtTL * H )
        maxH = np.max(H, axis=1)
        t3 = np.sum(maxH * np.sum(dgkl(Ht, H), axis=1))
    return lambda_L / 2 * (2*t2 - t1 + sigmaL * t3)

def diff_surrogate(Ht, H, L, sigmaL=sigmaL, lambda_L=1, dgkl=False):
    b_inf = trace_xtLx(L, H.T) * lambda_L / 2
    if dgkl:
        b_supp = smooth_dgkl_surrogate(Ht, L=L, H=H, sigmaL=sigmaL, lambda_L=lambda_L)
    else:
        b_supp = smooth_l2_surrogate(Ht, L=L, H=H, sigmaL=sigmaL, lambda_L=lambda_L)
    return b_supp - b_inf

class SmoothNMF(NMFEstimator):

    loss_names_ = NMFEstimator.loss_names_ + ["log_reg_loss"] + ["Lapl_reg_loss"]

    # args and kwargs are copied from the init to the super instead of capturing them in *args and **kwargs to be scikit-learn compliant.
    def __init__(self, lambda_L = 1.0, accelerate=False, linesearch=False, mu=0, epsilon_reg=1, algo_hq=False, 
                 force_simplex=True, dicotomy_tol=dicotomy_tol, **kwargs):

        super().__init__( **kwargs)
        self.accelerate = accelerate
        self.lambda_L = lambda_L
        self.linesearch = linesearch
        self.mu = mu
        self.epsilon_reg = epsilon_reg
        self.force_simplex = force_simplex
        self.dicotomy_tol = dicotomy_tol
        
        

        self.algo_hq = algo_hq
        if self.accelerate:
            assert np.max(np.array(self.mu))==0, "mu is not available for the accelerated algorithm."
            assert not self.l2
            self.sigmaL_ = sigmaL
            if self.linesearch:
                self.gamma = [self.sigmaL_]

        


    def _iteration_checkerboard(self, W, Hs, update_W=True):

        # KL_surr = KL_loss_surrogate(self.X_, W, H, H, eps=0)
        # log_surr = log_surrogate(H, H, mu=self.mu, epsilon=self.epsilon_reg)
        # print("loss before:", KL_surr, log_surr, log_surr+KL_surr)
        if self.linesearch:
            Holds = [Hs[i].copy() for i in range(len(Hs))]
        self.sigmaLs_ = []
        for i in range(4):
            if self.algo_hq:
                Hs[i] = multiplicative_step_hq(self.Xs_[i], self.G_, W, Hs[i], force_simplex=self.force_simplex, eps=self.log_shift, safe=self.debug, dicotomy_tol=self.dicotomy_tol, lambda_L=self.lambda_L, L=self.Ls_[i], sigmaL=self.sigmaLs_[i])
            else:
                Hs[i] = multiplicative_step_h(self.Xs_[i], self.G_, W, Hs[i], force_simplex=self.force_simplex, mu=self.mu, eps=self.log_shift, epsilon_reg=self.epsilon_reg, safe=self.debug, dicotomy_tol=self.dicotomy_tol, lambda_L=self.lambda_L, L=self.Ls_[i], l2=self.l2, fixed_H=self.fixed_H)
            if self.linesearch:
                d = diff_surrogate(Holds[i], Hs[i], L=self.Ls_[i], sigmaL=self.sigmaLs_[i], dgkl=not(self.algo_hq))
                if d>0:
                    self.sigmaLs_.append(self.sigmaLs_[i] / 1.2)
                else:
                    self.sigmaLs_.append(self.sigmaLs_[i] * 1.5)
        
        if self.linesearch:
            self.gamma.append(self.sigmaLs_)

        if update_W:
            W = multiplicative_step_w_checkerboard(self.Xs_, self.G_, W, Hs, eps=self.log_shift, safe=self.debug, l2=self.l2, fixed_W=self.fixed_W)
        
        # KL_surr = KL_loss_surrogate(self.X_, W, H, Hold, eps=0)
        # log_surr = log_surrogate(H, Hold, mu=self.mu, epsilon=self.epsilon_reg)
        # print("surrogate before:", KL_surr, log_surr, log_surr+KL_surr)
        # KL_surr = KL_loss_surrogate(self.X_, W, H, H, eps=0)
        # log_surr = log_surrogate(H, H, mu=self.mu, epsilon=self.epsilon_reg)
        # print("loss after:", KL_surr, log_surr, log_surr+KL_surr)

        if callable(self.G) : 
            self.G_ = self.G(part_W = W[:-2,:],G = self.G_)
        return  W, Hs

    def _iteration(self, W, H, update_W=True):

        # KL_surr = KL_loss_surrogate(self.X_, W, H, H, eps=0)
        # log_surr = log_surrogate(H, H, mu=self.mu, epsilon=self.epsilon_reg)
        # print("loss before:", KL_surr, log_surr, log_surr+KL_surr)

        if self.linesearch:
            Hold = H.copy()
        if self.algo_hq:
            H = multiplicative_step_hq(self.X_, self.G_, W, H, force_simplex=self.force_simplex, eps=self.log_shift, safe=self.debug, dicotomy_tol=self.dicotomy_tol, lambda_L=self.lambda_L, L=self.L_, sigmaL=self.sigmaL_)
        else:
            H = multiplicative_step_h(self.X_, self.G_, W, H, force_simplex=self.force_simplex, mu=self.mu, eps=self.log_shift, epsilon_reg=self.epsilon_reg, safe=self.debug, dicotomy_tol=self.dicotomy_tol, lambda_L=self.lambda_L, L=self.L_, l2=self.l2, fixed_H=self.fixed_H)

        if update_W:
            W = multiplicative_step_w(self.X_, self.G_, W, H, eps=self.log_shift, safe=self.debug, l2=self.l2, fixed_W=self.fixed_W)
        if self.linesearch:
            d = diff_surrogate(Hold, H, L=self.L_, sigmaL=self.sigmaL_, dgkl=not(self.algo_hq))
            if d>0:
                self.sigmaL_  = self.sigmaL_ / 1.2
            else:
                self.sigmaL_  = self.sigmaL_ * 1.5
            self.gamma.append(self.sigmaL_ )

        # KL_surr = KL_loss_surrogate(self.X_, W, H, Hold, eps=0)
        # log_surr = log_surrogate(H, Hold, mu=self.mu, epsilon=self.epsilon_reg)
        # print("surrogate before:", KL_surr, log_surr, log_surr+KL_surr)
        # KL_surr = KL_loss_surrogate(self.X_, W, H, H, eps=0)
        # log_surr = log_surrogate(H, H, mu=self.mu, epsilon=self.epsilon_reg)
        # print("loss after:", KL_surr, log_surr, log_surr+KL_surr)

        if callable(self.G) : 
            self.G_ = self.G(part_W = W[:-2,:],G = self.G_)
        return  W, H

    def loss(self, W, H, average=True, X = None):
        lkl = super().loss(W, H, average=average, X = X)
        
        reg = log_reg(H, self.mu, self.epsilon_reg, average=False)
        if average:
            reg = reg / self.GWH_numel_
        self.detailed_loss_.append(reg)

        l2 = 0.5 * self.lambda_L * trace_xtLx(self.L_, H.T, average=False)
        if average:
            l2 = l2 / self.GWH_numel_
        self.detailed_loss_.append(l2)

        return lkl + reg + l2

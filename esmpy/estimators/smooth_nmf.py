import numpy as np

from esmpy.estimators.updates import multiplicative_step_h, multiplicative_step_w, multiplicative_step_hq, proj_grad_step_h, proj_grad_step_w, gradH, gradW, estimate_Lipschitz_bound_h, estimate_Lipschitz_bound_w
from esmpy.measures import trace_xtLx, log_reg
from esmpy.estimators import NMFEstimator
from esmpy.conf import dicotomy_tol, sigmaL
from copy import deepcopy
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

# def smooth_l2_surrogate_alt(Ht, L, H=None, sigmaL=sigmaL, lambda_L=1):
#     HtTL = Ht @ L
#     t1 = np.sum(HtTL * Ht)
#     if H is None:
#         return lambda_L / 2 * t1
    
#     t2 = 2 * np.sum(HtTL * (H - Ht) )
#     t3 = sigmaL * np.sum((Ht-H)**2)
    
#     return lambda_L / 2 * (t1 + t2 + t3)

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

def diff_surrogate(Ht, H, L, sigmaL=sigmaL, lambda_L=1, algo="log_surrogate"):
    b_inf = trace_xtLx(L, H.T) * lambda_L / 2
    if algo=="log_surrogate":
        b_supp = smooth_dgkl_surrogate(Ht, L=L, H=H, sigmaL=sigmaL, lambda_L=lambda_L)
    elif algo== "l2_surrogate":
        b_supp = smooth_l2_surrogate(Ht, L=L, H=H, sigmaL=sigmaL, lambda_L=lambda_L)
    else: 
        raise "Unknown algorithm"
    return b_supp - b_inf



def quadratic_surrogate(x, xt, f_xt, gradf_xt, sigma):
    r"""Compute the quadratic surrogate function of :math:`f` at :math:`x^t`

    This function essentially computes:

    .. math::
        
        g(x,x^t) = f(x^t) + left< x - x^t , \nabla f (x^t) \right> + \sigma \| x - x^t \|_2^2 

    :param np.array x: variable
    :param np.array xt: variable
    :param np.array f_xt: function to be upper bounded
    :param np.array gradf_xt: function that compute the gradient
    :param float sigma: Lipschitz constant of the gradient

    :returns: the answer

    """
    return f_xt + np.sum((x-xt) * gradf_xt) + sigma * np.sum((x-xt)**2)



class SmoothNMF(NMFEstimator):

    loss_names_ = NMFEstimator.loss_names_ + ["log_reg_loss"] + ["Lapl_reg_loss"] + ["gamma"]

    # args and kwargs are copied from the init to the super instead of capturing them in *args and **kwargs to be scikit-learn compliant.
    def __init__(self, lambda_L = 1.0, linesearch=False, mu=0, epsilon_reg=1, algo="log_surrogate", 
                 force_simplex=True, dicotomy_tol=dicotomy_tol, gamma=None, **kwargs):

        super().__init__( **kwargs)
        self.lambda_L = lambda_L
        self.linesearch = linesearch
        self.mu = mu
        self.epsilon_reg = epsilon_reg
        self.force_simplex = force_simplex
        self.dicotomy_tol = dicotomy_tol
        assert algo in ["l2_surrogate", "log_surrogate", "projected_gradient"]
        self.algo = algo
        self.gamma = gamma

        if self.linesearch:
            assert not self.l2
            assert lambda_L > 0

        if self.algo=="l2_surrogate":
            assert not self.l2
        

    def fit_transform(self, X, y=None, W=None, H=None):
        if self.gamma is None:

            if self.algo in ["l2_surrogate", "log_surrogate"]:
                self.gamma_ = sigmaL
            else:
                gamma_W = estimate_Lipschitz_bound_w(self.log_shift, X, self.G, k=self.n_components)
                gamma_H = estimate_Lipschitz_bound_h(self.log_shift, X, self.G, k=self.n_components, lambda_L=self.lambda_L, mu=self.mu, epsilon_reg=self.epsilon_reg)
                self.gamma_ = [gamma_H, gamma_W]
        else:
            self.gamma_ = deepcopy(self.gamma)

        return super().fit_transform(X, y=y, W=W, H=H)

    def _iteration(self, W, H):

        # KL_surr = KL_loss_surrogate(self.X_, W, H, H, eps=0)
        # log_surr = log_surrogate(H, H, mu=self.mu, epsilon=self.epsilon_reg)
        # print("loss before:", KL_surr, log_surr, log_surr+KL_surr)

        # 1. Update for H
        if self.linesearch:
            Hold = H.copy()
        if self.algo=="l2_surrogate":
            H = multiplicative_step_hq(self.X_, self.G_, W, H, force_simplex=self.force_simplex, log_shift=self.log_shift, safe=self.debug, dicotomy_tol=self.dicotomy_tol, lambda_L=self.lambda_L, L=self.L_, sigmaL=self.gamma_, fixed_H=self.fixed_H)
        elif self.algo=="log_surrogate":
            H = multiplicative_step_h(self.X_, self.G_, W, H, force_simplex=self.force_simplex, mu=self.mu, log_shift=self.log_shift, epsilon_reg=self.epsilon_reg, safe=self.debug, dicotomy_tol=self.dicotomy_tol, lambda_L=self.lambda_L, L=self.L_, l2=self.l2, fixed_H=self.fixed_H, sigmaL=self.gamma_)
        elif self.algo=="projected_gradient":
            H = proj_grad_step_h(self.X_, self.G_, W, H, force_simplex=self.force_simplex, mu=self.mu, log_shift=self.log_shift, epsilon_reg=self.epsilon_reg, safe=self.debug, dicotomy_tol=self.dicotomy_tol, lambda_L=self.lambda_L, L=self.L_, l2=self.l2, fixed_H=self.fixed_H, gamma=self.gamma_[0])
        else:
            raise ValueError("Unknown algorithm")

        if self.linesearch:
            if self.algo in ["l2_surrogate", "log_surrogate"]:
                d = diff_surrogate(Hold, H, L=self.L_, sigmaL=self.gamma_, algo=self.algo)
                if d>0:
                    self.gamma_  = self.gamma_ / 1.05
                else:
                    self.gamma_  = self.gamma_ * 1.5
            else:
                gradf_xt = gradH(self.X_, self.G_, W, Hold, mu= self.mu, lambda_L=self.lambda_L, L=self.L_, epsilon_reg=self.epsilon_reg, log_shift=self.log_shift, safe=self.debug)
                f_xt = self.loss(W, Hold, X = self.X_, average=False)
                f_x = self.loss(W, H, X = self.X_, average=False)
                g_xxt = quadratic_surrogate(H, Hold, f_xt, gradf_xt, self.gamma_[0])
                d = g_xxt - f_x
                if d>0:
                    self.gamma_[0]  = self.gamma_[0] / 1.05
                else:
                    self.gamma_[0]  = self.gamma_[0] * 1.5

        # 2. Update for W
        if self.algo in ["l2_surrogate", "log_surrogate"]:
            W = multiplicative_step_w(self.X_, self.G_, W, H, log_shift=self.log_shift, safe=self.debug, l2=self.l2, fixed_W=self.fixed_W)
        else:
            if self.linesearch:
                Wold = W.copy()
            W = proj_grad_step_w(self.X_, self.G_, W, H, log_shift=self.log_shift, safe=self.debug, gamma=self.gamma_[1])
            if self.linesearch:
                gradf_xt = gradW(self.X_, self.G_, Wold, H, log_shift=self.log_shift, safe=self.debug)
                f_xt = self.loss(Wold, H, X = self.X_, average=False)
                f_x = self.loss(W, H, X = self.X_, average=False)
                g_xxt = quadratic_surrogate(W, Wold, f_xt, gradf_xt, self.gamma_[1])
                d = g_xxt - f_x
                if d>0:
                    self.gamma_[1]  = self.gamma_[1] / 1.05
                else:
                    self.gamma_[1]  = self.gamma_[1] * 1.5

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
        if isinstance(self.gamma_, list):
            self.detailed_loss_.append(self.gamma_[0])
        else:
            self.detailed_loss_.append(self.gamma_)

        return lkl + reg + l2
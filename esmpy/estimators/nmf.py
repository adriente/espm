
from esmpy.updates import multiplicative_step_h, multiplicative_step_w
from esmpy.measures import KLdiv, log_reg
from esmpy.conf import log_shift, dicotomy_tol
from esmpy.estimators import NMFEstimator


class NMF(NMFEstimator):
    
    loss_names_ = NMFEstimator.loss_names_ + ["log_reg_loss"]

    
    # args and kwargs are copied from the init to the super instead of capturing them in *args and **kwargs to be scikit-learn compliant.
    def __init__(self, log_shift=log_shift, mu=0, epsilon_reg=1, dicotomy_tol=dicotomy_tol,
                 **kwargs):

        super().__init__(**kwargs)
        self.mu = mu
        self.epsilon_reg = epsilon_reg
        self.log_shift = log_shift
        self.dicotomy_tol = dicotomy_tol

    def _iteration(self, W, H):
        H = multiplicative_step_h(self.X_, self.G_, W, H, force_simplex=self.force_simplex, mu=self.mu, eps=self.log_shift, epsilon_reg=self.epsilon_reg, safe=self.debug, dicotomy_tol=self.dicotomy_tol, l2=self.l2,fixed_H_inds=self.fixed_H_inds)
        W = multiplicative_step_w(self.X_, self.G_, W, H, eps=self.log_shift, safe=self.debug, l2=self.l2, fixed_W=self.fixed_W)
        if callable(self.G) : 
            self.G_ = self.G(part_W = W[:-2,:],G = self.G_)

        return  W, H

    def loss(self, W, H, average=True, X = None):
        lkl = super().loss(W, H, average=average, X = X)
        # GP = self.G_ @ P
        # kl = KLdiv(self.X_, GP, A, self.log_shift, safe=self.debug) 
        reg = log_reg(H, self.mu, self.epsilon_reg, average=False)
        if average:
            reg = reg / self.GWH_numel_
        self.detailed_loss_.append(reg)
        return lkl + reg




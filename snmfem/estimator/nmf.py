
from snmfem.updates import multiplicative_step_a, multiplicative_step_p
from snmfem.measures import KLdiv, log_reg
from snmfem.conf import log_shift, dicotomy_tol
from snmfem.estimator.base import NMFEstimator


class NMF(NMFEstimator):

    def __init__(self,*args, mu=0,
                 epsilon_reg=1, dicotomy_tol=dicotomy_tol,**kwargs):

        super().__init__(*args,**kwargs)
        self.mu = mu
        self.epsilon_reg = epsilon_reg
        self.log_shift = log_shift
        self.dicotomy_tol = dicotomy_tol

    def _iteration(self, P, A):
        A = multiplicative_step_a(self.X, self.G, P, A, force_simplex=self.force_simplex, mu=self.mu, eps=self.log_shift, epsilon_reg=self.epsilon_reg, safe=self.debug, dicotomy_tol=self.dicotomy_tol)
        P = multiplicative_step_p(self.X, self.G, P, A, eps=self.log_shift, safe=self.debug)

        return  P, A

    def loss(self, P, A):
        GP = self.G @ P
        kl = KLdiv(self.X, GP, A, self.log_shift, safe=self.debug) 
        reg = log_reg(A, self.mu, self.epsilon_reg)
        return kl + reg




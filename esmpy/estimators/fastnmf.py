from snmfem.updates import multiplicative_step_a, multiplicative_step_p
from snmfem.measures import KLdiv, log_reg
from snmfem.conf import log_shift, dicotomy_tol
from snmfem.estimators import NMFEstimator


class FastNMF(NMFEstimator):
    
    loss_names_ = NMFEstimator.loss_names_ + ["log_reg_loss"]

    
    # args and kwargs are copied from the init to the super instead of capturing them in *args and **kwargs to be scikit-learn compliant.
    def __init__(self, log_shift=log_shift, epsilon_reg=1, mu= 0.0, dicotomy_tol=dicotomy_tol, **kwargs):

        super().__init__( **kwargs )
        self.mu = mu
        self.epsilon_reg = epsilon_reg
        self.log_shift = log_shift
        self.dicotomy_tol = dicotomy_tol

    def _iteration(self, P, A):
        A = multiplicative_step_a(self.X_, self.G_, P, A, force_simplex=self.force_simplex, mu=self.mu, eps=self.log_shift, epsilon_reg=self.epsilon_reg, safe=self.debug, dicotomy_tol=self.dicotomy_tol, l2=self.l2)
        P = multiplicative_step_p(self.X_, self.G_, P, A, eps=self.log_shift, safe=self.debug, l2=self.l2)

        return  P, A

    def loss(self, P, A, average=True, X = None):
        lkl = super().loss(P, A, average=average, X = X)
        # GP = self.G_ @ P
        # kl = KLdiv(self.X_, GP, A, self.log_shift, safe=self.debug) 
        reg = log_reg(A, self.mu, self.epsilon_reg, average=True)
        if average:
            reg = reg / self.GPA_numel_
        self.detailed_loss_.append(reg)
        return lkl + reg

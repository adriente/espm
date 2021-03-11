
from snmfem.updates import multiplicative_step_a, multiplicative_step_p
from snmfem.measures import KLdiv, log_reg
from snmfem.conf import log_shift, dicotomy_tol
from snmfem.estimators import NMFEstimator
from snmfem.conf import log_shift


class SmoothNMF(NMFEstimator):
    # args and kwargs are copied from the init to the super instead of capturing them in *args and **kwargs to be scikit-learn compliant.
    def __init__(self, n_components=None, init='warn', tol=1e-4, max_iter=200,
                 random_state=None, verbose=1, log_shift=log_shift, debug=False,
                 force_simplex=True, mu=0, epsilon_reg=1, dicotomy_tol=dicotomy_tol
                 ):

        super().__init__( n_components=n_components, init=init, tol=tol, max_iter=max_iter,
                        random_state=random_state, verbose=verbose, log_shift=log_shift, debug=debug,
                        force_simplex=force_simplex
                        )
        self.mu = mu
        self.epsilon_reg = epsilon_reg
        self.log_shift = log_shift
        self.dicotomy_tol = dicotomy_tol

    def _iteration(self, P, A):
        A = multiplicative_step_a(self.X_, self.G_, P, A, force_simplex=self.force_simplex, mu=self.mu, eps=self.log_shift, epsilon_reg=self.epsilon_reg, safe=self.debug, dicotomy_tol=self.dicotomy_tol)
        P = multiplicative_step_p(self.X_, self.G_, P, A, eps=self.log_shift, safe=self.debug)

        return  P, A

    def loss(self, P, A):
        GP = self.G_ @ P
        kl = KLdiv(self.X_, GP, A, self.log_shift, safe=self.debug) 
        reg = log_reg(A, self.mu, self.epsilon_reg)
        return kl + reg




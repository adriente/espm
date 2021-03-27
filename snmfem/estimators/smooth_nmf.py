
from snmfem.updates import multiplicative_step_a, multiplicative_step_p
from snmfem.measures import trace_xtLx
from snmfem.estimators import NMF


class SmoothNMF(NMF):

    loss_names_ = NMF.loss_names_ + ["Laplacian reglarization"]

    # args and kwargs are copied from the init to the super instead of capturing them in *args and **kwargs to be scikit-learn compliant.
    def __init__(self, lambda_L=1, **kwargs):

        super().__init__(**kwargs)
        self.lambda_L = lambda_L

    def _iteration(self, P, A):
        A = multiplicative_step_a(self.X_, self.G_, P, A, force_simplex=self.force_simplex, mu=self.mu, eps=self.log_shift, epsilon_reg=self.epsilon_reg, safe=self.debug, dicotomy_tol=self.dicotomy_tol, lambda_L=self.lambda_L, L=self.L_, l2=self.l2)
        P = multiplicative_step_p(self.X_, self.G_, P, A, eps=self.log_shift, safe=self.debug, l2=self.l2)
        return  P, A

    def loss(self, P, A):
        l1 = super().loss(P, A)
        l2 = self.lambda_L * trace_xtLx(self.L_, A.T, average=True)
        self.detailed_loss_.append(l2)
        return l1 + l2




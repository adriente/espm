import numpy as np

from espm.estimators.updates import multiplicative_step_h, multiplicative_step_w, multiplicative_step_hq, proj_grad_step_h, proj_grad_step_w, gradH, gradW, estimate_Lipschitz_bound_h, estimate_Lipschitz_bound_w
from espm.measures import trace_xtLx, log_reg
from espm.estimators import NMFEstimator
from espm.estimators.surrogates import diff_surrogate, quadratic_surrogate
from espm.conf import dicotomy_tol, sigmaL
from copy import deepcopy
# from espm.measures import KL_loss_surrogate, KLdiv_loss, log_reg, log_surrogate



class SmoothNMF(NMFEstimator):
    r"""SmoothNMF - NMF with a smooth regularization term

    We encourage to read the example available in the documentation:    
    https://espm.readthedocs.io/en/latest/introduction/notebooks/toy-problem.html 
    
    The corresponding notebook is available on github:
    https://github.com/adriente/espm/blob/main/notebooks/toy-ML.ipynb


    The class `SmoothNMF` implements the regularized NMF algorithm. It solves problems of the form:

    .. math::

        \dot{W}, \dot{H} = \arg \min_{W \geq \epsilon, H \geq \epsilon} D_{GKL}(X || GWH) +
        \lambda_L tr(H \Delta H^\top) + \mu  \sum_{ij} \log(H_{ij} + \epsilon_{reg})

    where:
    
    - `D_{GKL}` is the Generalized KL divergence loss function defined as:

        .. math::

            D_{GKL}(X || Y) = \sum_{i,j} X_{ij} \log \frac{X_{ij}}{Y_{ij}} - X_{ij} + Y_{ij}

        See the documentation of the class :mod:`espm.estimators.NMFEstimator` for more details.
    
    - `\Delta` is the Laplacian operator (it can be created using the function `create_laplacian_matrix` from the `utils` module).
    
    - `\epsilon_{reg}` is the slope of the log regularization/sparsity at 0 (you probably want to leave this to 1).

    - `\lambda_L` is a regularization parameter, which encourages smoothness in the columns of `H`.
    
    - `\mu` is a regularization parameter, which is similar to an L1 sparsity penalty.

    The size of:

    - `X` is `(n, p)`
    - `W` is `(m, k)`
    - `H` is `(k, p)`
    - `G` is `(n, m)`

    The columns of the matrices `H` and `X` are assumed to be images, typically for the smoothness regularization.
    The parameter `shape_2d` defines the shape of the images, i.e., `shape_2d[0]*shape_2d[1] = p`.
    
    Parameters
    ----------
    lambda_L : float, default=1.0
        Regularization parameter for the smooth regularization term.
    linesearch : bool, default=False
        If True, use a line search to find the step size.
    mu : float, default=0
        Regularization parameter for the log regularization/sparsity term.
    epsilon_reg : float, default=1
        Slope of the log regularization/sparsity at 0.
    algo : str, default="log_surrogate"
        Algorithm to use for the smooth regularization term. Can be "log_surrogate", "l2_surrogate", or "projected_gradient".
    force_simplex : bool, default=True
        If True, force the solution to be in the simplex.
    dicotomy_tol : float, default=1e-3
        Tolerance for the dichotomy algorithm.
    gamma : float, default=None
        Initial value for the step size. If None, it is set to the Lipschitz constant of the gradient.
    **kwargs : dict
        Additional parameters for the `NMFEstimator` class.

    """        

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
        """Fit the model to the data X and returns the transformed data.

        The size of:

        * :math:`X` is :math:`(n, p)`,
        * :math:`W` is :math:`(m, k)`,
        * :math:`H` is :math:`(k, p)`,
        * :math:`G` is :math:`(n, m)`.    
    
        Parameters
        ----------
        X : array-like, shape (n, p)
            Data matrix to be decomposed
        y : Ignored
            Not used, present here for API consistency by convention.
        W : array-like, shape (m, k)
            If init='custom', it is used as initial guess for the solution.
        H : array-like, shape (k, p)
            If init='custom', it is used as initial guess for the solution.

        Returns
        -------
        GW : ndarrays
            Transformed data.
            
        """
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
        """Compute the loss function."""
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
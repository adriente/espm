import numpy as np
from espm.estimators.updates import multiplicative_step_h, multiplicative_step_w, multiplicative_step_hq, proj_grad_step_h, proj_grad_step_w, gradH, gradW, estimate_Lipschitz_bound_h, estimate_Lipschitz_bound_w
from espm.measures import trace_xtLx, log_reg
from espm.estimators import NMFEstimator
from espm.estimators.surrogates import diff_surrogate, quadratic_surrogate
from espm.conf import dicotomy_tol, sigmaL
from copy import deepcopy
from espm.conf import log_shift
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
    simplex_H : bool, default=False
        If True, force the solution of H to be in the simplex.
    simplex_W : bool, default=True
        If True, force the solution of W to be in the simplex.
    dicotomy_tol : float, default=1e-3
        Tolerance for the dichotomy algorithm.
    gamma : float, default=None
        Initial value for the step size. If None, it is set to the Lipschitz constant of the gradient.
    **kwargs : dict
        Additional parameters for the `NMFEstimator` class.

    """        

    loss_names_ = NMFEstimator.loss_names_ + ["log_reg_loss"] + ["Lapl_reg_loss"] + ["gamma"]

    # args and kwargs are copied from the init to the super instead of capturing them in *args and **kwargs to be scikit-learn compliant.
    def __init__(self,
                 lambda_L = 0.0,
                 linesearch=False,
                 mu=0,
                 epsilon_reg=1,
                 algo="log_surrogate",
                 dicotomy_tol=dicotomy_tol,
                 gamma=None,
                 n_components=2,
                 init=None,
                 tol=1e-4,
                 max_iter=200,
                 random_state=None,
                 verbose=1,
                 debug=False,
                 l2=False,
                 G=None,
                 shape_2d = None,
                 normalize = False,
                 log_shift=log_shift,
                 eval_print=10,
                 true_D = None,
                 true_H = None,
                 fixed_H = None,
                 fixed_W = None,
                 hspy_comp = False,
                 no_stop_criterion = False,
                 simplex_H=False,
                 simplex_W = True
                 ):

        super().__init__(n_components=n_components,
                         init=init,
                         tol=tol,
                         max_iter=max_iter,
                         random_state=random_state,
                         verbose=verbose,
                         debug=debug,
                         l2=l2,
                         G=G,
                         shape_2d = shape_2d,
                         normalize = normalize,
                         log_shift=log_shift,
                         eval_print=eval_print,
                         true_D = true_D,
                         true_H = true_H,
                         fixed_H = fixed_H,
                         fixed_W = fixed_W,
                         hspy_comp = hspy_comp,
                         no_stop_criterion = no_stop_criterion,
                         simplex_H=simplex_H,
                         simplex_W = simplex_W)
        self.lambda_L = lambda_L
        self.linesearch = linesearch
        self.mu = mu
        self.epsilon_reg = epsilon_reg
        self.dicotomy_tol = dicotomy_tol
        self.algo = algo
        self.gamma = gamma
        self.check_params()

    def check_params(self) : 
        """Check the parameters of the model."""
        # Type checking of the parameters
        # TODO: Add typing from the __init__ method. I don't know why it didn't work ...
        if not isinstance(self.lambda_L, (int, float)):
            print("The regularization parameter lambda_L must be a float or int")
            print("The regularization parameter lambda_L is set to 0.0")
            self.lambda_L = 0.0
        if not isinstance(self.linesearch, bool):
            print("The linesearch parameter must be a boolean")
            print("The linesearch parameter is set to False")
            self.linesearch = False
        if not isinstance(self.mu, (int, float, np.ndarray)):
            print("The regularization parameter mu must be a float, int or np.ndarray")
            print("The regularization parameter mu is set to 0")
            self.mu = 0
        if not isinstance(self.epsilon_reg, (int, float)):
            print("The regularization parameter epsilon_reg must be a float or int")
            print("The regularization parameter epsilon_reg is set to 1")
            self.epsilon_reg = 1
        if not isinstance(self.algo, str):
            print("The algorithm parameter must be a string")
            print("The algorithm is set to 'log_surrogate'")
            self.algo = "log_surrogate"
        if not isinstance(self.simplex_H, bool):
            print("The simplex_H parameter must be a boolean")
            print("The simplex_H parameter is set to False")
            self.simplex_H = False
        if not isinstance(self.simplex_W, bool):
            print("The simplex_W parameter must be a boolean")
            print("The simplex_W parameter is set to True")
            self.simplex_W = True
        if not isinstance(self.dicotomy_tol, (int, float)):
            print("The dicotomy_tol parameter must be a float or int")
            print("The dicotomy_tol parameter is set to 1e-3")
            self.dicotomy_tol = 1e-3
        if self.gamma is not None and not isinstance(self.gamma, (int, float, list)):
            print("The gamma parameter must be a float, int, or list")
            print("The gamma parameter is set to None")
            self.gamma = None
        if not isinstance(self.verbose, (bool, int)):
            print("The verbose parameter must be a boolean or int")
            print("The verbose parameter is set to 1")
            self.verbose = 1
        if not isinstance(self.debug, bool):
            print("The debug parameter must be a boolean")
            print("The debug parameter is set to False")
            self.debug = False
        if not isinstance(self.l2, bool):
            print("The l2 parameter must be a boolean")
            print("The l2 parameter is set to False")
            self.l2 = False
        if not isinstance(self.n_components, int):
            print("The n_components parameter must be an int")
            print("The n_components parameter is set to 2")
            self.n_components = 2
        # Value checking of the parameters
        if not(self.algo in ["l2_surrogate", "log_surrogate", "projected_gradient", "bmd"]) :
            print("The algorithm must be 'l2_surrogate', 'log_surrogate', 'bmd' or 'projected_gradient'")
            print("The algorithm is set to 'log_surrogate'")
            self.algo = "log_surrogate"
        if not(self.lambda_L >= 0) :
            print("The regularization parameter lambda_L must be non-negative")
            print("The regularization parameter lambda_L is set to 0")
            self.lambda_L = 0
        if not(self.epsilon_reg > 0.0) :
            print("The regularization parameter epsilon_reg must be positive")
            print("The regularization parameter epsilon_reg is set to 1")
            self.epsilon_reg = 1.0 
        if not(np.all(np.array(self.mu)>=0)) : 
            print("The regularization parameter mu must be non-negative")
            print("The regularization parameter mu is set to 0")
            self.mu = 0
        if not((self.simplex_H and not(self.simplex_W)) or (not(self.simplex_H) and self.simplex_W) or (not(self.simplex_H) and not(self.simplex_W))) :
            print("The simplex constraint must be applied to either W or H or none of them")
            print("The simplex constraint is applied to W and not to H")
            self.simplex_W = True
            self.simplex_H = False
        if self.linesearch:
            if self.l2 :
                print("The l2 parameter must be False when using linesearch")
                print("The l2 parameter is set to False")
                self.l2 = False 
            if not(self.lambda_L > 0) : 
                print("The regularization parameter lambda_L must be non-zero when using linesearch")
                print("The regularization parameter lambda_L is set to 1")
                self.lambda_L = 1

        if not(self.algo=="l2_surrogate") :
            if self.l2 : 
                print("The l2 parameter must be False when using the algorithm "+self.algo)
                print("The l2 parameter is set to False")
                self.l2 = False

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
        # To be remove in future versions. In this commented version below, G_ is called before intialisation which obviously causes issues.
        # I'll move it to the _iteration method. Adrien

        # if self.gamma is None:

        #     if self.algo in ["l2_surrogate", "log_surrogate"]:
        #         self.gamma_ = sigmaL
        #     else:
        #         gamma_W = estimate_Lipschitz_bound_w(self.log_shift, X, self.G_, k=self.n_components)
        #         gamma_H = estimate_Lipschitz_bound_h(self.log_shift, X, self.G_, k=self.n_components, lambda_L=self.lambda_L, mu=self.mu, epsilon_reg=self.epsilon_reg)
        #         self.gamma_ = [gamma_H, gamma_W]
        # else:
        #     self.gamma_ = deepcopy(self.gamma)

        self.gamma_ = None

        return super().fit_transform(X, y=y, W=W, H=H)

    def _iteration(self, W, H):

        # KL_surr = KL_loss_surrogate(self.X_, W, H, H, eps=0)
        # log_surr = log_surrogate(H, H, mu=self.mu, epsilon=self.epsilon_reg)
        # print("loss before:", KL_surr, log_surr, log_surr+KL_surr)

        if self.n_iter_ == 0:
            if self.gamma is None:

                if self.algo in ["l2_surrogate", "log_surrogate", "bmd"]:
                    self.gamma_ = sigmaL
                else:
                    gamma_W = estimate_Lipschitz_bound_w(self.log_shift, self.X_, self.G_, k=self.n_components)
                    gamma_H = estimate_Lipschitz_bound_h(self.log_shift,
                                                         self.X_,
                                                         self.G_,
                                                         k=self.n_components,
                                                         lambda_L=self.lambda_L,
                                                         mu=self.mu,
                                                         epsilon_reg=self.epsilon_reg)
                    self.gamma_ = [gamma_H, gamma_W]
            else:
                self.gamma_ = deepcopy(self.gamma)

        # 1. Update for H
        if self.linesearch:
            Hold = H.copy()
        if self.algo=="l2_surrogate":
            H = multiplicative_step_hq(self.X_,
                                       self.G_,
                                       W,
                                       H,
                                       simplex_H=self.simplex_H,
                                       log_shift=self.log_shift,
                                       safe=self.debug,
                                       dicotomy_tol=self.dicotomy_tol,
                                       lambda_L=self.lambda_L,
                                       L=self.L_,
                                       sigmaL=self.gamma_,
                                       fixed_H=self.fixed_H)
        elif self.algo=="log_surrogate":
            H = multiplicative_step_h(self.X_,
                                      self.G_,
                                      W,
                                      H,
                                      simplex_H=self.simplex_H,
                                      mu=self.mu,
                                      log_shift=self.log_shift,
                                      epsilon_reg=self.epsilon_reg,
                                      safe=self.debug,
                                      dicotomy_tol=self.dicotomy_tol,
                                      lambda_L=self.lambda_L,
                                      L=self.L_,
                                      l2=self.l2,
                                      fixed_H=self.fixed_H,
                                      sigmaL=self.gamma_)
        elif self.algo=="projected_gradient":
            H = proj_grad_step_h(self.X_,
                                 self.G_,
                                 W,
                                 H,
                                 simplex_H=self.simplex_H,
                                 mu=self.mu,
                                 log_shift=self.log_shift,
                                 epsilon_reg=self.epsilon_reg,
                                 safe=self.debug,
                                 dicotomy_tol=self.dicotomy_tol,
                                 lambda_L=self.lambda_L,
                                 L=self.L_,
                                 l2=self.l2,
                                 fixed_H=self.fixed_H,
                                 gamma=self.gamma_[0])
        elif self.algo=="bmd":
            H = multiplicative_step_h(self.X_,
                                      self.G_,
                                      W,
                                      H,
                                      simplex_H=self.simplex_H,
                                      mu=self.mu,
                                      log_shift=self.log_shift,
                                      epsilon_reg=self.epsilon_reg,
                                      safe=self.debug,
                                      dicotomy_tol=self.dicotomy_tol,
                                      lambda_L=self.lambda_L,
                                      L=self.L_,
                                      l2=self.l2,
                                      fixed_H=self.fixed_H,
                                      sigmaL=self.gamma_,
                                      use_bregman=True)
        else:
            raise ValueError("Unknown algorithm")

        if self.linesearch:
            if self.algo in ["l2_surrogate", "log_surrogate", "bmd"]:
                d = diff_surrogate(Hold, H, L=self.L_, sigmaL=self.gamma_, algo=self.algo)
                if d>0:
                    self.gamma_  = self.gamma_ / 1.05
                else:
                    self.gamma_  = self.gamma_ * 1.5
            else:
                gradf_xt = gradH(self.X_,
                                 self.G_,
                                 W,
                                 Hold,
                                 mu= self.mu,
                                 lambda_L=self.lambda_L,
                                 L=self.L_,
                                 epsilon_reg=self.epsilon_reg,
                                 log_shift=self.log_shift,
                                 safe=self.debug)
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
            W = multiplicative_step_w(self.X_,
                                      self.G_,
                                      W,
                                      H,
                                      log_shift=self.log_shift,
                                      safe=self.debug,
                                      l2=self.l2,
                                      simplex_W=self.simplex_W,
                                      fixed_W=self.fixed_W,
                                      physics_model=self.physics_model_)
        elif self.algo=="bmd":
            W = multiplicative_step_w(self.X_,
                                      self.G_,
                                      W,
                                      H,
                                      log_shift=self.log_shift,
                                      safe=self.debug,
                                      l2=self.l2,
                                      simplex_W=self.simplex_W,
                                      fixed_W=self.fixed_W,
                                      use_bregman=True,
                                      physics_model=self.physics_model_)
        else:
            if self.linesearch:
                Wold = W.copy()
            W = proj_grad_step_w(self.X_,
                                 self.G_,
                                 W,
                                 H,
                                 log_shift=self.log_shift,
                                 safe=self.debug,
                                 gamma=self.gamma_[1],
                                 simplex_W=self.simplex_W)
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
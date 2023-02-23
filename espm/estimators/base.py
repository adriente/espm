import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from espm.estimators.updates import initialize_algorithms
from espm.measures import KLdiv_loss, Frobenius_loss, find_min_angle, find_min_MSE
from espm.conf import log_shift
from espm.utils import rescaled_DH
import time
from abc import ABC, abstractmethod
from espm.utils import create_laplacian_matrix 
from scipy.sparse import lil_matrix


def normalization_factor (X, nc) : 
    m = np.mean(X)
    return nc/(m*X.shape[0])

class NMFEstimator(ABC, TransformerMixin, BaseEstimator):
    r""" Abstract class for NMF algorithms.

    This abstract class `espm.estimators.NMFEstimator` is used to implement the different NMF algorithms. It solves problems of the form:

    .. math::

        \dot{W}, \dot{H} = \arg \min_{W \geq \epsilon, H \geq \epsilon} \frac{1}{2} L(X, GWH) + R(W, H)

    where 
    :math:`X` is the data matrix, 
    :math:`G` is a matrix of known values, 
    :math:`W` and :math:`H` are the matrices to be learned, 
    :math:`R` is a regularization term and 
    :math:`L` is a loss function. that represent the generalized KL divergence. As a reminder, the generalized KL divergence is defined as:

    .. math::

        D_{GKL}(X || Y) = \sum_{i,j} X_{ij} \log \frac{X_{ij}}{Y_{ij}} - X_{ij} + Y_{ij}

    where :math:`Y = GWH`. Since :math:`X` does not depend on :math:`W` and :math:`H`, we obtain the loss function:

    .. math::

        L(X, Y) = - \sum_{i,j} X_{ij} \log \frac{GWH_{ij}} + GWH_{ij}

    The Generalized KL divergence has the advantage of being zero when :math:`X = Y`, which is not the case for our loss.
    Therefore, we shift the loss function by a constant :math:`C` such that it equals the Generalized KL divergence. 
    This constant is stored in the attribute :attr:`espm.estimators.NMFEstimator.const_KL_`.

    The loss function can also be selected to be the Frobenius norm. In this case, the loss function is:

    .. math::

        L(X, Y) = \frac{1}{2} \sum_{i,j} (X_{ij} - Y_{ij})^2

    While the code will work, it is not recommended to use the Frobenius norm as a loss function. This code is optimized for the KL divergence.

    The size of:

    * :math:`X` is :math:`(n, p)`,
    * :math:`W` is :math:`(m, k)`,
    * :math:`H` is :math:`(k, p)`,
    * :math:`G` is :math:`(n, m)`.

    The columns of the matrices :math:`H` and :math:`X` are assumed to be images. This is used typically for the smoothness regularization.
    The parameter `shape_2d` defines the shape of the images, i.e. `shape_2d[0]*shape_2d[1] = p`.

    Parameters
    ----------
    n_components : int, default=2
        Number of components, i.e. dimensionality of the latent space.
    init : str
        Method used to initialize the procedure. Default is None
        The method use the initialization of :mod:`sklearn.decomposition`. 
        It can be imported using:
        .. code-block::python
            >>> from sklearn.decomposition._nmf import _initialize_nmf
        
    tol : float, default=1e-4
        Tolerance of the stopping condition.
    max_iter : int, default=200
        Maximum number of iterations before timing out.
    random_state : int, RandomState instance, default=None
    verbose : int, default=1
        The verbosity level.
    debug : bool, default=False
        If True, the algorithm will log more and perform more checks.
    l2 : bool, default=False
        If True, the algorithm will use the l2 norm instead of the KL divergence.
    G : np.array, function or None, default=None
        If np.array, it is the known matrix of the data. 
        If function, it is a function that takes as input the data matrix and returns the known matrix (np.array). 
        If None, it is assumed that G is the identity matrix.
    shape_2d : tuple or None, default=None
        If not None, it is the image shape of the columns of the matrices  :math:`X` and  :math:`H`.
    normalize : bool, default=False
        If True, the algorithm will normalize the data matrix  :math:`X`.
    log_shift : float, default=1e-10
        Lower bound for W and H, i.e. :math:`\epsilon`.
    eval_print : int, default=10
        Number of iterations between each evaluation of the loss function.
    true_D : np.array or None, default=None
        Ground truth for the matrix :math:`GW`. Used for evaluation purposes.
    true_H : np.array or None, default=None
        Ground truth for the matrix :math:`H`. Used for evaluation purposes.
    fixed_H : np.array or None, default=None
        If not None, it fixes the non-zero values of the matrix :math:`H`.
    fixed_W : np.array or None, default=None
        If not None, it fixes the non-zero values of the matrix :math:`W`.
    hspy_comp : bool, default=False
        If True, the algorithm will use the format compatible with hyperspy.
        Use this option if you run the algorithm with the method decompositio in hyperspy.
        For example:
        .. code-block::python
            >>> est = SmoothNMF( n_components = 3, hspy_comp = True)
            >>> out = spim.decomposition(algorithm = est, return_info=True)
        
    """
    loss_names_ = ["KL_div_loss"]
    const_KL_ = None
    
    def __init__(self, n_components=2, init=None, tol=1e-4, max_iter=200,
                 random_state=None, verbose=1, debug=False,
                 l2=False,  G=None, shape_2d = None, normalize = False, log_shift=log_shift, 
                 eval_print=10, true_D = None, true_H = None, fixed_H = None, fixed_W = None, hspy_comp = False
                 ):
        self.n_components = n_components
        self.init = init
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.log_shift = log_shift
        self.debug = debug
        self.l2 = l2
        self.G = G
        self.shape_2d = shape_2d
        self.eval_print = eval_print
        self.true_D = true_D
        self.true_H = true_H
        self.fixed_H = fixed_H
        self.fixed_W = fixed_W
        self.hspy_comp = hspy_comp
        self.normalize = normalize

    def _more_tags(self):
        return {'requires_positive_X': True}

    @abstractmethod
    def _iteration(self,  W, H):
        pass
    

    def loss(self, W, H, average=True, X = None):
        """Loss function.

        Compute the loss function for the given matrices W and H.

        Parameters
        ----------
        W : np.array
            Matrix of shape (n, k)
        H : np.array
            Matrix of shape (k, p)
        average : bool, default=True
            If True, the loss is averaged over the number of elements of the matrices.
        X : np.array or None, default=None
            If not None, it is the data matrix. If None, it is assumed that the data matrix in `self.X_`.

        Returns
        -------
        loss_ : float
            Value of the loss function.

        """
        GW = self.G_ @ W
        if X is None : 
            X = self.X_

        assert(X.shape == (self.G_.shape[0],H.shape[1]))

        self.GWH_numel_ = self.G_.shape[0] * H.shape[1]
        
        if self.l2:
            loss_ = 0.5*Frobenius_loss(X, GW, H, average=False) 
        else:
            if self.const_KL_ is None:
                self.const_KL_ = np.sum(X*np.log(np.maximum(self.X_, self.log_shift))) - np.sum(X) 

            loss_ =  KLdiv_loss(X, GW, H, self.log_shift, average=False) + self.const_KL_
        if average:
            loss_ = loss_ / self.GWH_numel_
        self.detailed_loss_ = [loss_]
        return loss_

    def fit_transform(self, X, y=None, W=None, H=None):
        """Learn a NMF model for the data X and returns the transformed data.
        This is more efficient than calling fit followed by transform.

        The size of:

        * :math:`X` is :math:`(n, p)`,
        * :math:`W` is :math:`(m, k)`,
        * :math:`H` is :math:`(k, p)`,
        * :math:`G` is :math:`(n, m)`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n, p)
            Data matrix to be decomposed
        y : Ignored
            Not used, present here for API consistency by convention.
        W : array-like of shape (m, k)
            If specified, it is used as initial guess for the solution.
        H : array-like of shape (k, p)
            If specified, it is used as initial guess for the solution.

        Returns
        -------
        GW : ndarrays
            Transformed data.
        
        """
        if self.hspy_comp : 
            self.X_ = self._validate_data(X.T, dtype=[np.float64, np.float32])
        else : 
            self.X_ = self._validate_data(X, dtype=[np.float64, np.float32])

        if self.hspy_comp==False:
            try:
                import inspect
                curframe = inspect.currentframe()
                calframe = inspect.getouterframes(curframe, 2)
                if calframe[1][3]=="decomposition" and "hyperspy" in calframe[1][1]:
                    print("Are you calling the function decomposition from Hyperspy?\n" +
                        "If so, please set the compatibility argument 'hspy_comp' to True.\n\n" + 
                        "If this argument is not set correctly, the function will not work properly!!!")
            except:
                pass


        self.X_ = self.remove_zeros_lines(self.X_, self.log_shift)
        self.const_KL_ = None
        if self.normalize : 
            self.norm_factor_ = normalization_factor(self.X_,self.n_components)
            self.X_ = self.norm_factor_ * self.X_
        

        if callable(self.G): 
            G = self.G()
        else : 
            G = self.G
        
        self.G_, self.W_, self.H_ = initialize_algorithms(X = self.X_, G = G, W = W, H = H, n_components = self.n_components, init = self.init, random_state = self.random_state, force_simplex = self.force_simplex)

        if not(self.shape_2d is None) :
            self.L_ = create_laplacian_matrix(*self.shape_2d)
        else : 
            self.L_ =lil_matrix((self.X_.shape[1],self.X_.shape[1]),dtype=np.float32)
            self.L_.setdiag([1]*self.X_.shape[1])

        algo_start = time.time()
        # If mu_sparse != 0, this is the regularized step of the algorithm
        # Otherwise this is directly the data fitting step
        eval_before = np.inf
        eval_init = self.loss(self.W_, self.H_)
        self.n_iter_ = 0

        # if self.debug:
        self.losses_ = []
        self.rel_ = []
        self.detailed_losses_ = []
        if not(self.true_D is None) and not(self.true_H is None) : 
            if (self.true_D.shape[1] == self.n_components) and (self.true_H.shape[0] == self.n_components) : 
                self.angles_ = []
                self.mse_ = []
                self.true_losses_ = []
                true_DH = self.true_D @ self.true_H
            else : 
                print("The chosen number of components does not match the number of components of the provided truth. The ground truth will be ignored.")
        try:
            while True:
                # Take one step in A, P
                old_W, old_H = self.W_.copy(), self.H_.copy()
                
                self.W_, self.H_ = self._iteration(self.W_, self.H_ )
                eval_after = self.loss(self.W_, self.H_)
                self.n_iter_ +=1
                
                rel_W = np.max((self.W_ - old_W)/(self.W_ + self.tol*np.mean(self.W_) ))
                rel_H = np.max((self.H_ - old_H)/(self.H_ + self.tol*np.mean(self.H_) ))

                # store some information for assessing the convergence
                # for debugging purposes
                
                # We need to store this value as 
                #    loss = self.loss(self.P_,self.A_, X = true_DA )
                # might destroy it. Furthermore, saving the data before the if, might cause 
                # an error if the optimization is stoped with a keyboard interrupt. 
                detailed_loss_ = self.detailed_loss_

                if not(self.true_D is None) and not(self.true_H is None) :
                    if (self.true_D.shape[1] == self.n_components) and (self.true_H.shape[0] == self.n_components) : 
                        if self.force_simplex:
                            W, H = self.W_, self.H_ 
                        else:
                            W, H = rescaled_DH(self.W_, self.H_ )
                        GW = self.G_ @ W
                        angles = find_min_angle(self.true_D.T,GW.T, unique=True)
                        mse = find_min_MSE(self.true_H, H,unique=True)
                        loss = self.loss(self.W_,H, X = true_DH )
                        self.angles_.append(angles)
                        self.mse_.append(mse)
                        self.true_losses_.append(loss)
                
                self.losses_.append(eval_after)
                self.detailed_losses_.append(detailed_loss_)
                self.rel_.append([rel_W,rel_H])
                              
                # check convergence criterions
                if self.n_iter_ >= self.max_iter:
                    print("exits because max_iteration was reached")
                    break

                # If there is no regularization the algorithm stops with this criterion
                # Otherwise it goes to the data fitting step
                elif max(rel_H,rel_W) < self.tol:
                    print(
                        "exits because of relative change rel_A {} or rel_P {} < tol ".format(
                            rel_H,rel_W
                        )
                    )
                    break
                elif abs((eval_before - eval_after)/eval_init) < self.tol:
                    print(
                        "exits because of relative change < tol: {}".format(
                            (eval_before - eval_after)/min(eval_before, eval_after)
                        )
                    )
                    break

                elif np.isnan(eval_after):
                    print("exit because of the presence of NaN")
                    break

                elif (eval_before - eval_after) < 0:
                    if hasattr(self, "accelerate"):
                        if not self.accelerate:
                            print("exit because of negative decrease {}: {}, {}".format((eval_before - eval_after), eval_before, eval_after))
                            break
                    else:
                        print("exit because of negative decrease {}: {}, {}".format((eval_before - eval_after), eval_before, eval_after))
                        break
                
                if self.verbose > 0 and np.mod(self.n_iter_, self.eval_print) == 0:
                    print(
                        f"It {self.n_iter_} / {self.max_iter}: loss {eval_after:3e},  {self.n_iter_/(time.time()-algo_start):0.3f} it/s",
                    )
                    pass
                eval_before = eval_after
        except KeyboardInterrupt:
            pass
        
        if not(self.force_simplex):
            self.W_, self.H_ = rescaled_DH(self.W_, self.H_ )
        
        algo_time = time.time() - algo_start
        print(
            f"Stopped after {self.n_iter_} iterations in {algo_time//60} minutes "
            f"and {np.round(algo_time) % 60} seconds."
        )
        self.reconstruction_err_ = self.loss(self.W_, self.H_)

        if self.normalize : 
            self.W_ = self.W_ / self.norm_factor_
        
        GW = self.G_ @ self.W_
        self.n_components_ = self.H_.shape[0]
        
        if self.hspy_comp : 
            self.components_ = GW.T
            return self.H_.T
        else : 
            self.components_ = self.H_
            return GW

    def fit(self, X, y=None, **params):
        """ Learn a NMF model for the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed
        y : Ignored
        params : dict
            Parameters passed to the `fit_transform` method.
        

        Returns
        -------
        self
            The model.

        """
        self.fit_transform(X, **params)
        return self

    # def transform(self, X):
    #     """Transform the data X according to the fitted NMF model.
    #     Parameters
    #     ----------
    #     X : {array-like, sparse matrix} of shape (n_samples, n_features)
    #         Data matrix to be transformed by the model.
    #     Returns
    #     -------
    #     P : ndarray of shape (n_samples, n_components)
    #         Transformed data.
    #     """
    #     check_is_fitted(self)
    #     X = self._validate_data(X, accept_sparse=('csr', 'csc'),
    #                             dtype=[np.float64, np.float32],
    #                             reset=False)

    #     return self.P_

    def inverse_transform(self, W):
        """ Transform data back to its original space.

        Parameters
        ----------
        W : {ndarray, sparse matrix} of shape (n_samples, n_components)
            Transformed data matrix.
        
        Returns
        -------
        
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Data matrix of original shape.
        
        """
        check_is_fitted(self)
        return self.G_ @ W @ self.H_
    
    def get_losses(self):
        
        if not(self.true_D is None) and not(self.true_H is None) :
            mse_list = []
            angles_list = []
            for i in range(self.n_components) :
                angles_list.append("ang_p{}".format(i))
                mse_list.append("mse_p{}".format(i))
            names = ["full_loss"] + self.loss_names_ + ["rel_W","rel_H"] + angles_list + mse_list + ["true_KL_loss"]

            dt_list = []
            for elt in names : 
                dt_list.append((elt,"float64"))
            dt = np.dtype(dt_list)

            tup_list = []
            for i in range(len(self.losses_)) : 
                tup_list.append((self.losses_[i],) + tuple(self.detailed_losses_[i]) + tuple(self.rel_[i]) \
                    + tuple(self.angles_[i]) + tuple(self.mse_[i]) + (self.true_losses_[i],) )
            
            array = np.array(tup_list,dtype=dt)
        
        #Bon j'ai copié ca comme un bourrin. Il y a moyen de faire mieux.
        else : 
            names = ["full_loss"] + self.loss_names_ + ["rel_W","rel_H"]
            dt_list = []
            for elt in names : 
                dt_list.append((elt,"float64"))
            dt = np.dtype(dt_list)

            tup_list = []
            for i in range(len(self.losses_)) : 
                tup_list.append((self.losses_[i],) + tuple(self.detailed_losses_[i]) + tuple(self.rel_[i]))
            
            array = np.array(tup_list,dtype=dt)

        return array

    def remove_zeros_lines (self, X, epsilon) : 
        if np.all(X >= 0) : 
            new_X = X.copy()
            sum_cols = X.sum(axis = 0)
            sum_rows = X.sum(axis = 1)
            new_X[:,np.where(sum_cols == 0)] = epsilon
            new_X[np.where(sum_rows == 0),:] = epsilon
            return new_X
        else : 
            raise ValueError("There are negative values in X")


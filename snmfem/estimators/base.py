import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from snmfem.updates import initialize_algorithms
from snmfem.measures import KLdiv_loss, KLdiv, Frobenius_loss, find_min_angle, find_min_MSE
from snmfem.conf import log_shift
from snmfem.utils import rescaled_DA
import time
from abc import ABC, abstractmethod 



class NMFEstimator(ABC, TransformerMixin, BaseEstimator):
    
    loss_names_ = ["KL_div_loss"]
    const_KL_ = None
    
    def __init__(self, n_components=None, init='warn', tol=1e-4, max_iter=200,
                 random_state=None, verbose=1, log_shift=log_shift, debug=False,
                 force_simplex=True, skip_G=False, l2=False,
                 ):
        self.n_components = n_components
        self.init = init
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.log_shift = log_shift
        self.debug = debug
        self.force_simplex= force_simplex
        self.skip_G = skip_G
        self.l2 = l2

    def _more_tags(self):
        return {'requires_positive_X': True}

    @abstractmethod
    def _iteration(self,  P, A):
        pass
    

    def loss(self, P, A, average=True, X = None):
        GP = self.G_ @ P
        if X is None : 
            X = self.X_

        assert(X.shape == (self.G_.shape[0],A.shape[1]))

        self.GPA_numel_ = self.G_.shape[0] * A.shape[1]
        
        if self.l2:
            loss = Frobenius_loss(X, GP, A, average=False) 
        else:
            if self.const_KL_ is None:
                self.const_KL_ = np.sum(X*np.log(self.X_+ self.log_shift)) - np.sum(X) 

            loss =  KLdiv_loss(X, GP, A, self.log_shift, safe=self.debug, average=False) + self.const_KL_
        if average:
            loss = loss / self.GPA_numel_
        self.detailed_loss_ = [loss]
        return loss

    def fit_transform(self, X, y=None, G=None, P=None, A=None, shape_2d = None, eval_print=10, true_D = None, true_A = None):
        """Learn a NMF model for the data X and returns the transformed data.
        This is more efficient than calling fit followed by transform.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed
        P : array-like of shape (n_samples, n_components)
            If specified, it is used as initial guess for the solution.
        A : array-like of shape (n_components, n_features)
            If specified, it is used as initial guess for the solution.
        Returns
        -------
        P, A : ndarrays
        """
        self.X_ = self._validate_data(X, dtype=[np.float64, np.float32])
        self.const_KL_ = None
        
        if self.skip_G:
            G = None
        self.G_, self.P_, self.A_ = initialize_algorithms(self.X_, G, P, A, self.n_components, self.init, self.random_state, self.force_simplex)
        
        self.shape_2d_ = shape_2d
        if not(self.shape_2d_ is None) :
            self.L_ = create_laplacian_matrix(*self.shape_2d_)
        else : 
            self.L_ =lil_matrix((self.X_shape[1],self.X_shape[1]),dtype=np.float32)
            self.L_.setdiag([1]*self.X_shape[1])

        algo_start = time.time()
        # If mu_sparse != 0, this is the regularized step of the algorithm
        # Otherwise this is directly the data fitting step
        eval_before = np.inf
        eval_init = self.loss(self.P_, self.A_)
        self.n_iter_ = 0

        # if self.debug:
        self.losses_ = []
        self.rel_ = []
        self.detailed_losses_ = []
        self.true_D_ = true_D
        self.true_A_ = true_A
        if not(true_D is None) and not(true_A is None) : 
            self.angles_ = []
            self.mse_ = []
            self.true_losses_ = []
            true_DA = true_D @ true_A

        try:
            while True:
                # Take one step in A, P
                old_P, old_A = self.P_.copy(), self.A_.copy()
                
                self.P_, self.A_ = self._iteration( self.P_, self.A_ )
                eval_after = self.loss(self.P_, self.A_)
                self.n_iter_ +=1
                
                rel_P = np.max((self.P_ - old_P)/(self.P_ + self.tol*np.mean(self.P_) ))
                rel_A = np.max((self.A_ - old_A)/(self.A_ + self.tol*np.mean(self.A_) ))

                # store some information for assessing the convergence
                # for debugging purposes
                
                # We need to store this value as 
                #    loss = self.loss(self.P_,self.A_, X = true_DA )
                # might destroy it. Furthermore, saving the data before the if, might cause 
                # an error if the optimization is stoped with a keyboard interrupt. 
                detailed_loss_ = self.detailed_loss_

                if not(true_D is None) and not(true_A is None) :
                    if self.force_simplex:
                        P, A = self.P_, self.A_ 
                    else:
                        P, A = rescaled_DA(self.P_, self.A_ )
                    GP = self.G_ @ P
                    angles = find_min_angle(true_D.T,GP.T, unique=True)
                    mse = find_min_MSE(true_A, A,unique=True)
                    loss = self.loss(self.P_,A, X = true_DA )
                    self.angles_.append(angles)
                    self.mse_.append(mse)
                    self.true_losses_.append(loss)
                
                self.losses_.append(eval_after)
                self.detailed_losses_.append(detailed_loss_)
                self.rel_.append([rel_P,rel_A])
                              
                # check convergence criterions
                if self.n_iter_ >= self.max_iter:
                    print("exits because max_iteration was reached")
                    break

                # If there is no regularization the algorithm stops with this criterion
                # Otherwise it goes to the data fitting step
                elif max(rel_A,rel_P) < self.tol:
                    print(
                        "exits because of relative change rel_A {} or rel_P {} < tol ".format(
                            rel_A,rel_P
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
                    print("exit because of negative decrease")
                    break
                
                if self.verbose > 0 and np.mod(self.n_iter_, eval_print) == 0:
                    print(
                        f"It {self.n_iter_} / {self.max_iter}: loss {eval_after:0.3f},  {self.n_iter_/(time.time()-algo_start):0.3f} it/s",
                    )
                eval_before = eval_after
        except KeyboardInterrupt:
            pass
        
        if not(self.force_simplex):
            self.P_, self.A_ = rescaled_DA(self.P_, self.A_ )
        
        algo_time = time.time() - algo_start
        print(
            f"Stopped after {self.n_iter_} iterations in {algo_time//60} minutes "
            f"and {np.round(algo_time) % 60} seconds."
        )
        self.reconstruction_err_ = self.loss(self.P_, self.A_)

        self.n_components_ = self.A_.shape[0]
        self.components_ = self.A_

        GP = self.G_ @ self.P_

        return GP

    def fit(self, X, y=None, **params):
        """Learn a NMF model for the data X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed
        y : Ignored
        Returns
        -------
        self
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

    def inverse_transform(self, P):
        """Transform data back to its original space.
        Parameters
        ----------
        P : {ndarray, sparse matrix} of shape (n_samples, n_components)
            Transformed data matrix.
        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Data matrix of original shape.
        .. versionadded:: 0.18
        """
        check_is_fitted(self)
        return self.G_ @ P @ self.A_
    
    def get_losses(self):
        
        if not(self.true_D_ is None) and not(self.true_A_ is None) :
            mse_list = []
            angles_list = []
            for i in range(self.n_components) :
                angles_list.append("ang_p{}".format(i))
                mse_list.append("mse_p{}".format(i))
            names = ["full_loss"] + self.loss_names_ + ["rel_P","rel_A"] + angles_list + mse_list + ["true_KL_loss"]

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
            names = ["full_loss"] + self.loss_names_ + ["rel_P","rel_A"]
            dt_list = []
            for elt in names : 
                dt_list.append((elt,"float64"))
            dt = np.dtype(dt_list)

            tup_list = []
            for i in range(len(self.losses_)) : 
                tup_list.append((self.losses_[i],) + tuple(self.detailed_losses_[i]) + tuple(self.rel_[i]))
            
            array = np.array(tup_list,dtype=dt)

        return array
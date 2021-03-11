import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from snmfem.updates import initialize_algorithms
from snmfem.measures import KLdiv
from snmfem.conf import log_shift
import time
from abc import ABC, abstractmethod 



class NMFEstimator(ABC, TransformerMixin, BaseEstimator):

    def __init__(self, n_components=None, init='warn', tol=1e-4, max_iter=200,
                 random_state=None, verbose=1, log_shift=log_shift, debug=False,
                 force_simplex=True
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
        

    def _more_tags(self):
        return {'requires_positive_X': True}

    @abstractmethod
    def _iteration(self,  P, A):
        pass

    def loss(self, P, A):
        GP = self.G_ @ P
        kl = KLdiv(self.X_, GP, A, self.log_shift, safe=self.debug) 
        return kl

    def fit_transform(self, X, y=None, G=None, P=None, A=None, eval_print=10):
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

        self.G_, self.P_, self.A_ = initialize_algorithms(self.X_, G, P, A, self.n_components, self.init, self.random_state, self.force_simplex)
        

        algo_start = time.time()
        # If mu_sparse != 0, this is the regularized step of the algorithm
        # Otherwise this is directly the data fitting step
        eval_before = np.inf
        self.n_iter_ = 0
        if self.debug:
            self.losses = []
        try:
            while True:
                start = time.time()
                # Take one step in A, P
                self.P_, self.A_ = self._iteration( self.P_, self.A_ )
                eval_after = self.loss(self.P_, self.A_)
                self.n_iter_ +=1


                # store some information for assessing the convergence
                # for debugging purposes
                if self.debug:
                    self.losses.append(eval_after)


                # check convergence criterions
                if self.n_iter_ >= self.max_iter:
                    print("exits because max_iteration was reached")
                    break

                # If there is no regularization the algorithm stops with this criterion
                # Otherwise it goes to the data fitting step
                elif abs(eval_before - eval_after) < self.tol:
                    print(
                        "exits because of relative change < tol: {}".format(
                            eval_before - eval_after
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
                        f"It {self.n_iter_} / {self.max_iter}: loss {eval_after:0.3f},  {time.time()-start:0.3f} s/it",
                    )
                eval_before = eval_after
        except KeyboardInterrupt:
            pass

        algo_time = time.time() - algo_start
        print(
            f"Stopped after {self.n_iter_} iterations in {algo_time//60} minutes "
            f"and {np.round(algo_time) % 60} seconds."
        )

        self.reconstruction_err_ = KLdiv(self.X_, self.G_ @ self.P_, self.A_, self.log_shift, safe=self.debug) 

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
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from esmpy.updates import initialize_algorithms
from esmpy.measures import KLdiv_loss, Frobenius_loss, find_min_angle, find_min_MSE
from esmpy.conf import log_shift
from esmpy.utils import rescaled_DH
import time
from abc import ABC, abstractmethod
from esmpy.laplacian import create_laplacian_matrix 
from scipy.sparse import lil_matrix


def normalization_factor (X, nc) : 
    m = np.mean(X)
    return nc/(m*X.shape[0])

class NMFEstimator(ABC, TransformerMixin, BaseEstimator):
    
    loss_names_ = ["KL_div_loss"]
    const_KL_ = None
    
    def __init__(self, n_components=2, init='warn', tol=1e-4, max_iter=200,
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
        GW = self.G_ @ W
        if X is None : 
            X = self.X_

        assert(X.shape == (self.G_.shape[0],H.shape[1]))

        self.GWH_numel_ = self.G_.shape[0] * H.shape[1]
        
        if self.l2:
            loss = Frobenius_loss(X, GW, H, average=False) 
        else:
            if self.const_KL_ is None:
                self.const_KL_ = np.sum(X*np.log(self.X_+ self.log_shift)) - np.sum(X) 

            loss =  KLdiv_loss(X, GW, H, self.log_shift, safe=self.debug, average=False) + self.const_KL_
        if average:
            loss = loss / self.GWH_numel_
        self.detailed_loss_ = [loss]
        return loss

    def fit_transform(self, X, y=None, W=None, H=None, n_pixel_side=0):
        """Learn a NMF model for the data X and returns the transformed data.
        This is more efficient than calling fit followed by transform.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed
        W : array-like of shape (n_samples, n_components)
            If specified, it is used as initial guess for the solution.
        H : array-like of shape (n_components, n_features)
            If specified, it is used as initial guess for the solution.
        Returns
        -------
        W, H : ndarrays
        """
        print(self.n_components)
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
            Hs, Ws = [], []
            Hs.append(self.H_.copy())
            Ws.append(self.W_.copy())
            while True:
                # Take one step in A, P
                old_W, old_H = self.W_.copy(), self.H_.copy()
                
                self.W_, self.H_ = self._iteration(self.W_, self.H_)
                Hs.append(self.H_.copy())
                Ws.append(self.W_.copy())
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
                        f"It {self.n_iter_} / {self.max_iter}: loss {eval_after:0.3f},  {self.n_iter_/(time.time()-algo_start):0.3f} it/s",
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
            return self.H_.T, Hs, Ws
        else : 
            self.components_ = self.H_
            return GW, Hs, Ws

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

    def inverse_transform(self, W):
        """Transform data back to its original space.
        Parameters
        ----------
        W : {ndarray, sparse matrix} of shape (n_samples, n_components)
            Transformed data matrix.
        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Data matrix of original shape.
        .. versionadded:: 0.18
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


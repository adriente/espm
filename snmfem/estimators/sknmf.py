from sklearn.decomposition import NMF
from snmfem.utils import rescaled_DA, arg_helper

import numpy as np

default_dict = {
    "n_components" : None,
    "init" : "warn",
    "solver" : "cd",
    "beta_loss" : "frobenius",
    "tol" : 0.0001,
    "max_iter" : 200,
    "random_state" : None,
    "alpha" : 0.0,
    # "alpha_W" : 0.0,
    # "alpha_H" : "same",
    "l1_ratio" : 0.0,
    "verbose" : 0,
    "shuffle" : False,
    "regularization" : "both" 
}

class SKNMF(NMF):
    """Small compatibility wrapper for sckit-NMF"""
    
    def __init__(self, *args, **kwargs):
        filled_kwargs = arg_helper(kwargs,default_dict)
        super().__init__(
            *args,
            n_components = filled_kwargs["n_components"],
            init = filled_kwargs["init"],
            solver = filled_kwargs["solver"],
            beta_loss = filled_kwargs["beta_loss"],
            tol = filled_kwargs["tol"],
            max_iter = filled_kwargs["max_iter"],
            random_state = filled_kwargs["random_state"],
            alpha = filled_kwargs["alpha"],
            # alpha_W = filled_kwargs["alpha_W"],
            # alpha_H = filled_kwargs["alpha_H"],
            l1_ratio = filled_kwargs["l1_ratio"],
            verbose = filled_kwargs["verbose"],
            shuffle = filled_kwargs["shuffle"],
            regularization = filled_kwargs["regularization"]
            )
        self.G_ = None
        self.P_ = None
        self.A_ = None
        
    def  fit_transform(self, X, y=None, P=None, A=None):
        W = super().fit_transform(X, y=y, W=P, H=A)
        H = self.components_
        GP, A = rescaled_DA(W,H)
        self.G_ = np.eye(W.shape[0])
        self.P_ = GP
        self.A_ = A
        return W
    
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
    
    def get_losses(self):
        return None
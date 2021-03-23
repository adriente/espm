from sklearn.decomposition import NMF
from snmfem.utils import rescaled_DA

import numpy as np

class SKNMF(NMF):
    """Small compatibility wrapper for sckit-NMF"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.G_ = None
        self.P_ = None
        self.A_ = None
        
    def  fit_transform(self, X, y=None, G=None, P=None, A=None):
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
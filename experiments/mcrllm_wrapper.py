from mcrllm import mcrllm
import numpy as np

class MCRLLM() : 

    def __init__(self, n_components=None, init='Kmeans', tol=1e-4, max_iter=200,verbose=1,mcr_method = True, hspy_comp = True) :
        self.n_components = n_components
        self.init = init
        self.max_iter = max_iter
        self.verbose = verbose
        self.mcr_method = mcr_method
        self.hspy_comp = hspy_comp

    def fit_transform(self, X, y=None, P=None, A=None) : 
        if self.hspy_comp :
            Xt = X
        else : 
            Xt = X.T
        if self.mcr_method : 
            method = "variable"
        else : 
            method = ""
        mcr = mcrllm(Xt,self.n_components,self.init,self.max_iter,method)
        GP, A = mcr.S.T, mcr.C.T
        
        if self.hspy_comp : 
            self.components_ = GP.T
            return A.T
        else : 
            self.components_ = A
            return GP

    def fit(self, X, y=None, **params) : 
        self.fit_transform(X, **params)
        return self

    def get_losses(self):
        # To be implemented eventually
        return None

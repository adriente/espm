from snmfem.estimators.mcrllm.mcrllm import mcrllm
import numpy as np

class MCRLLM() : 

    def __init__(self, n_components=None, init='warn', tol=1e-4, max_iter=200,verbose=1,mcr_method = True) :
        self.n_components = n_components
        self.init = init
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.mcr_method = mcr_method

    def fit_transform(self, X, y=None, G=None, P=None, A=None, shape_2d = None,true_D = None, true_A = None) : 
        Xt = X.T
        if self.mcr_method : 
            method = "variable"
        else : 
            method = ""
        mcr = mcrllm(Xt,self.n_components,self.init,self.max_iter,method)
        GP, A = mcr.S.T, mcr.C.T
        self.G_ = np.eye(mcr.S.shape[1])
        self.P_ = GP
        self.A_ = A
        return GP

    def fit(self, X, y=None, **params) : 
        self.fit_transform(X, **params)
        return self

    def get_losses(self):
        return None

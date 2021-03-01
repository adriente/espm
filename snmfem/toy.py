import numpy as np
import snmfem.conf as conf
from pathlib import Path

class ToyModel () :
    def __init__(self,l = 25, k = 3, p = 100, c = 10, n_poisson=200, force_simplex=True,**kwargs) : 
        self.A = np.random.rand(k,p)
        if force_simplex:
            self.A = self.A/np.sum(self.A, axis=0, keepdims=True)
    
        self.G = np.random.rand(l,c)
        self.P = np.random.rand(c,k)
        GP = self.G @ self.P
        self.g_matr = self.G
        self.densities = np.ones((k,))
        self.n_poisson = n_poisson

        self.X = GP @ self.A

        self.Xdot = 1/n_poisson * np.random.poisson(n_poisson *self.X)

    def generate_g_matr(self,**kwargs) :
        pass

    def save(self,filename) :
        d = {}  # dictionary of everything we would like to save
        d["X"] = self.X
        d["Xdot"] = self.Xdot
        d["phases"] = (self.G @ self.P).T #Transpose to make have the same shape as generate_data.py
        d["densities"] = self.densities
        d["weights"] = self.A
        d["N"] = self.n_poisson
        np.savez(conf.DATASETS_PATH / Path(filename), **d)

# This should be done somewhere else I believe...
# if __name__ == "__main__" : 
#     t = ToyModel()
#     t.save("aspim038_toy.npz")

        

import numpy as np
import snmfem.conf as conf
from pathlib import Path
from snmfem.models import PhysicalModel

class Toy(PhysicalModel) :
    def __init__(self, *args,seed = 0,**kwargs) : 
        super().__init__(*args, **kwargs)
        self.seed = seed
        try : 
            self.c = self.params_dict["c"]
        except KeyError : 
            self.c = self.x.shape[0]
        try : 
            self.k = self.params_dict["k"]
        except KeyError : 
            print("You need to define a number of phases for GP.")
            
        np.random.seed(self.seed)
        self.P = np.random.rand(self.c, self.k)
        self.phases = None

    def generate_g_matr(self,**kwargs) : 
        np.random.seed(self.seed)
        self.G = np.random.rand(self.x.shape[0], self.c)

    def generate_spectrum(self) :
        np.random.seed(self.seed)
        self.spectrum = self.G @ np.random.rand(self.c,1)

    def generate_phases(self) : 
        self.phases = (self.G @ self.P).T
        

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
        self.k = None

    def generate_g_matr(self,**kwargs) : 
        np.random.seed(self.seed)
        self.G = np.random.rand(self.x.shape[0], self.c)

    def generate_phases(self, phases_parameters) : 
        self.k = len(phases_parameters)
        self.generate_g_matr()
        self.P = np.random.rand(self.c, self.k)
        self.phases = (self.G @ self.P).T
        

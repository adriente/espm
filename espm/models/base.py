r"""
Models abstract class
---------------------

Its main purpose is to read databases and set energy parameters. 
"""

from abc import ABC, abstractmethod
from pathlib import Path
import json
import espm.conf as conf
import numpy as np
from typing import Optional

class Model(ABC):
    """Abstract class for models."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate_g_matr (self, g_parameters) :
        pass

    @abstractmethod
    def generate_phases (self, phase_parameters) :
        pass



class ToyModel(Model):
    r"""Toy model.
    
    Simple model with a random G matrix and phases.

    The G matrix is generated with a random number of gaussians per column.
    The phases are generated using a vector drawn from a Laplacian distribution and multiplied by the G matrix.
    
    Parameters
    ----------
    L : int
        Length of the phases.
    C : int
        Number of possible components in the phases.
    K : int
        Number of phases.
    seed : int, optional
        Seed for the random number generator.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> from espm.models import ToyModel
        >>> import matplotlib.pyplot as plt
        >>> model = ToyModel(L=200, C=15, K=3, seed=0)
        >>> model.generate_g_matr()
        >>> model.generate_phases()
        >>> print(model.G.shape)
        (200, 15)
        >>> print(model.phases.shape)
        (200, 3)
        >>> plt.plot(model.phases)


    """
    def __init__(self, L: int=200, C: int=15, K: int=3, seed: Optional[int]=None) -> None:
        super().__init__()
        self.L = L
        self.C = C
        self.K = K
        self.seed = seed
        self.G = None
        self.phases = None
    
    def generate_g_matr (self, *args, **kwargs) -> None:
        """Generate G matrix.

        Parameters
        ----------
        args : ignored
        kwargs : ignored

        Returns
        -------
        None

        """

        if self.G is not None :
            return
        
        np.random.seed(seed=self.seed)
        n_el = 45
        n_gauss = np.random.randint(2, 5,[self.C])
        l = np.arange(0, 1, 1/self.L)
        mu_gauss = np.random.rand(n_el)
        sigma_gauss = 1/n_el + np.abs(np.random.randn(n_el))/n_el/5

        G = np.zeros([self.L,self.C])

        def gauss(x, mu, sigma):
            # return np.exp(-(x-mu)**2/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
            return np.exp(-(x-mu)**2/(2*sigma**2))

        for i, c in enumerate(n_gauss):
            inds = np.random.choice(n_el, size=[c] , replace=False)
            for ind in inds:
                w = 0.1+0.9*np.random.rand()
                G[:,i] += w * gauss(l, mu_gauss[ind], sigma_gauss[ind])
        self.G = G

    def generate_phases (self, *args, **kwargs) -> None:
        """Generate phases.

        Parameters
        ----------
        args : ignored
        kwargs : ignored

        Returns
        -------
        None

        """

        if self.phases is not None :
            return
        Wdot = np.abs(np.random.laplace(size=[self.C, self.K]))
        self.Wdot = Wdot / np.mean(Wdot)/self.L

        self.generate_g_matr()
        self.phases = self.G @ self.Wdot
    

class PhysicalModel(Model) :
    """Abstract class of the models"""
    def __init__(self, e_offset, e_size, e_scale, params_dict,db_name="default_xrays.json", E0 = 200, **kwargs) :
        super().__init__()
        self.x = self.build_energy_scale(e_offset, e_size, e_scale)
        self.params_dict = params_dict
        if db_name is None :
            self.db_dict = {}
            self.db_mdata = {}
        else :
            self.db_dict = self.extract_DB(db_name)
            self.db_mdata = self.extract_DB_mdata(db_name)
        self.bkgd_in_G = False
        self.spectrum = np.zeros_like(self.x)
        self.G = np.diag(np.ones_like(self.x))
        self.phases = None
        self.E0 = E0


    def extract_DB (self,db_name) :
        r"""
        Read the cross-sections from the database

        Parameters
        ----------
        db_name : 
            :string: Name of the json database file

        Returns
        -------
        data
            :dict: A dictionnary containing the cross-sections in the database
        """
        db_path = conf.DB_PATH / Path(db_name)
        with open(db_path,"r") as f :
            json_dict = json.load(f)
        return json_dict["table"]

    def extract_DB_mdata (self,db_name) :
        r"""
        Read the metadata of the database

        Parameters
        ----------
        db_name : 
            :string: Name of the json database file

        Returns
        -------
        data
            :dict: A dictionnary containing the metadata related to the database
        """
        db_path = conf.DB_PATH / Path(db_name)
        with open(db_path,"r") as f :
            json_dict = json.load(f)
        return json_dict["metadata"]

    def build_energy_scale(self,e_offset, e_size, e_scale) :
        r"""
        Build an array corresponding to a linear energy scale.

        Parameters
        ----------
        e_offset
            :float: Offset of the energy scale
        e_size 
            :int: Number of channels of the detector
        e_scale : 
            :float: Scale in keV/channel

        Returns
        -------
        energy scale
            :np.array 1D: Linear energy scale based on the given parameters of shape (e_size,)
        """
        return np.linspace(e_offset,e_offset+e_size*e_scale,num=e_size)


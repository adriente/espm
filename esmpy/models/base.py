r"""
Models abstract class
---------------------

Its main purpose is to read databases and set energy parameters. 
"""

from abc import ABC, abstractmethod
from pathlib import Path
import json
import esmpy.conf as conf
import numpy as np

class Model(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate_g_matr (self, g_parameters) :
        pass

    @abstractmethod
    def generate_phases (self, phase_parameters) :
        pass


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


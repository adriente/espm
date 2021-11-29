from abc import ABC, abstractmethod
from pathlib import Path
import json
import snmfem.conf as conf
import numpy as np

class PhysicalModel(ABC) :

    def __init__(self, e_offset, e_size, e_scale, params_dict,db_name="default_xrays.json", E0 = 200, **kwargs) :
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

    @abstractmethod
    def generate_g_matr (self, g_parameters) :
        pass

    @abstractmethod
    def generate_phases (self, phase_parameters) :
        pass

    def extract_DB (self,db_name) :
        db_path = conf.DB_PATH / Path(db_name)
        with open(db_path,"r") as f :
            json_dict = json.load(f)
        return json_dict["table"]

    def extract_DB_mdata (self,db_name) :
        db_path = conf.DB_PATH / Path(db_name)
        with open(db_path,"r") as f :
            json_dict = json.load(f)
        return json_dict["metadata"]

    def build_energy_scale(self,e_offset, e_size, e_scale) :
        return np.linspace(e_offset,e_offset+e_size*e_scale,num=e_size)


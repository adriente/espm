from abc import ABC, abstractmethod
import glob

class EMModel (ABC) :

    def __init__(self,DB_PATH, e_offset, e_size, e_scale, elts_dict, params_dict) :
        self.x = self.build_energy_scale(e_offset, e_size, e_scale)
        self.elts_dict = elts_dict
        self.params_dict = params_dict


    @abstractmethod
    def generate_spectrum() :
        pass

    @abstractmethod
    def generate_g_matr () :
        pass

    @staticmethod
    def extract_DB (DB_PATH) :
        json_filenames = glob.glob(DB_PATH + "*.json")

    @staticmethod
    def build_energy_scale(e_offset, e_size, e_scale) :
        return np.linspace(e_offset,e_offset+e_size*e_scale,num=e_size)

    # Eventually it could be interesting to make a decompose function that would give the different components of a model.
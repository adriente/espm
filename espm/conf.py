from pathlib import Path
import hyperspy.misc.eds.ffast_mac as macs

# Path of the base
BASE_PATH = Path(__file__).parent

# Path of the db
DB_PATH = BASE_PATH / Path("tables/")

NUMBER_PERIODIC_TABLE = DB_PATH / Path("periodic_table_number.json")

SYMBOLS_PERIODIC_TABLE = DB_PATH / Path("periodic_table_symbols.json")

SIEGBAHN_TO_IUPAC = DB_PATH / Path("siegbahn_to_iupac.json")

DEFAULT_SDD_EFF = "SDD_efficiency.txt"

# Path of the generated datasets
DATASETS_PATH = BASE_PATH.parent / Path("generated_datasets")
# Ensure that the folder DATASETS_PATH exists
DATASETS_PATH.mkdir(exist_ok=True, parents=True)


HSPY_MAC = macs.ffast_mac # Tabulated absorption coefficient in Hyperspy

DEFAULT_MISC_PARAMS = {
    "data_folder" : "default_synth_data",
    "shape_2d" : (80,80),
    "N" : 100,
    "densities" : [2.33,19.3],
    "model" : "EDXS",
    "seed" : 0
}

DEFAULT_EDXS_PARAMS = {
    "e_offset" : 0.200,
    "e_size" : 1980,
    "e_scale" : 0.01,
    "width_slope" : 0.01,
    "width_intercept" : 0.065,
    "db_name" : "200keV_xrays.json",
    "E0" : 200,
    "params_dict" : {
        "Det" : DEFAULT_SDD_EFF,
        "Abs" : {
            "thickness" : 100.0e-7,
            "toa" : 22,
            "density" : None,
            "atomic_fraction" : False
            }
    }
}

log_shift = 1e-14
dicotomy_tol = 1e-5
seed_max = 4294967295
sigmaL = 8
maxit_dichotomy = 100

from pathlib import Path

# Path of the base
BASE_PATH = Path(__file__).parent

# Path of the db
DB_PATH = BASE_PATH / Path("Data/")

# Path of the generated datasets
DATASETS_PATH = BASE_PATH.parent / Path("generated_datasets")
# Ensure that the folder DATASETS_PATH exists
DATASETS_PATH.mkdir(exist_ok=True, parents=True)

RESULTS_PATH = BASE_PATH.parent / Path("results")
# Ensure that the folder DATASETS_PATH exists
RESULTS_PATH.mkdir(exist_ok=True, parents=True)

SCRIPT_CONFIG_PATH = BASE_PATH.parent / Path("scripts/config/")

log_shift = 1e-14
dicotomy_tol = 1e-4
seed_max = 4294967295


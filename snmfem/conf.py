from pathlib import Path

# Path of the base
BASE_PATH = Path(__file__).parent

# Path of the db
DB_PATH = BASE_PATH / Path("Data/simple_xrays_threshold.json")

# Path of the generated datasets
DATASETS_PATH = BASE_PATH.parent / Path("generated_datasets")
# Ensure that the folder DATASETS_PATH exists
DATASETS_PATH.mkdir(exist_ok=True, parents=True)



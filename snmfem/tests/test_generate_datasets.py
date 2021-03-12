from snmfem.datasets import generate_edxs_dataset, generate_toy_dataset
from snmfem.conf import DATASETS_PATH
from pathlib import Path
import shutil


def test_generate_edxs_dataset():
    
    folder = DATASETS_PATH / Path("test")
    generate_edxs_dataset(folder=folder, seeds=[0], N=100)
    shutil.rmtree(folder)

    folder = DATASETS_PATH / Path("test")
    generate_edxs_dataset(folder=folder, seeds=[0, 2], N=58)
    shutil.rmtree(folder)

def test_generate_toy_dataset():
    
    folder = DATASETS_PATH / Path("test")
    generate_toy_dataset(folder=folder, seeds=[0], shape_2D = (15, 15), k = 5, N = 200, laplacian=True)    
    shutil.rmtree(folder)

    folder = DATASETS_PATH / Path("test")
    generate_toy_dataset(folder=folder, seeds=[0, 4, 6], shape_2D = (8, 25), k = 2, N = 25, laplacian=False)    
    shutil.rmtree(folder)

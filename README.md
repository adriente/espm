# This README needs an update

# Simplex non-negative matrix factorization for Electron Microscopy
This project contains the code to create artificial Energy dispersive X-ray spectroscopy (EDXS) data and to perform hyperspectral unmixing on EDXS spectrum images.


## Installation
You can install this package using
```
pip install cython
pip install .
```

## Installation as a developer (What you should do Adrien)
After setting up your virtual environnement, simply run 
```
pip install cython
pip install -e ".[testing]"
```

You can then simply run the tests with 
```
pytest snmfem
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

## Getting started with the repo
Generate the synthetic dataset. Run the script:
```
python scripts/generate_synthetic_dataset.py
```

## Running the algorithm
Fill a json file with the relevant input parameters such as dataset name, hyperparameters values, etc ... (see the template in scritps/config/).
Then you can run the algorithm running :
```
python experiments.py json_filename
```
It will produce two ouputs : 
* A .npz file with the G, P and A matrices
* A .json file containing the scores obtained based on the ground truth (and informations on the input used)

For now it is supported for NMF only and the user needs to provide artificial data with ground truths

# TO BE UPDATED!

## Repo structure
* The implementation of the SNMF algorithm can be found in `Code/snmf.py`
* If you want to use this you should have a look at the `Code/analyze_data.ipynb` notebook for some example code.
* You can create data using the `Code/generate_data.py` with the help of `Code\artificial_data.ipynb`

## Meaning of the matrices in the 'SNMF algorithm'
* `X` is the data we try to model, shoul be passed to the `fit` method as a (height, width, num_channels) matrix, but is internally processed as a (height * width, num_channels) matrix, you can find an example matrix in `Code/Data/aspim036_N150_2ptcls_brstlg.hspy` this is a cropped TEM image
* `G` is a matrix of the gaussians of which our reconstructed spectrum will be composed, so this is a (num_channels, num_possible_gaussians) matrix, this matrix can be constructed from the `xrays_V2.json` file by the `Gaussians.create_matrix` method in the `utils.py` file
* `P` is the matrix of weights assigned to each gaussian, for each spectrum: this is a (num_gaussians, num_spectra) matrix and can be interpreted as the portion of each gaussian in each specrum
* `B` represents the continuum X-rays corresponding to each phase and is the product of self-absorption, bremsstrahlung and detector efficiency, it is thus a (num_channels, num_spectra) matrix.
* `G * P + B` represent the actual reconstructed spectra
* `A` is the 'activations' matrix, this is internally stored as a (height * width, num_spectra) matrix, but should be interpreted as a (num_spectra, height, width) matrix: it represents the portion of each spectra, at every pixel

## Cross validation
To obtain sparser solutions a regularization is available in the SNMF code. To find automatically the best value for the strength of that regularization, it possible to use `Code/cross_validation.py` with the help of `Code/analyze_data.ipynb` to perform a grid search on the hyperparameters of the algorithm.

## Todo
* Integrate xraylib to generate a more accurate `xrays.json`
* Integrate a database to calculate absorption coefficients
* Integrate detector artefacts 

# This README needs an update

# Simplex non-negative matrix factorization for Electron Microscopy
This project contains the code to create artificial Energy dispersive X-ray spectroscopy (EDXS) data and to perform hyperspectral unmixing on EDXS spectrum images.


## Installation
You can install this package using
```
pip install cython
pip install .
```

If you want to develop, please use the option
```
pip install -e ".[dev]" 
```

# Set up the virtual env

To be done

## Getting started with the repo
Generate the synthetic dataset. Run the script:
```
python experiments/generate_synthetic_dataset.py
```

## Running the algorithm
Fill a json file with the relevant input parameters such as dataset name, hyperparameters values, etc ... (see the template in scritps/config/). They are two types of configuration files:
1. Dataset files
2. Experiment files

To run an experiment, you can use 
```
python scripts/synthetic_exp.py json_filename
```
For example the following line will execute the experiment with the NMF method and the toy dataset.
```
python scripts/synthetic_exp.py exp_NMF_Toy.json
```
Since the Toy dataset has 10 samples, it will run 10 different experiment. To only run one experiment, you can use
```
python scripts/synthetic_exp.py exp_NMF_Toy.json True 3
```
This will run the experiment with the sample 3. Here, the argument `True` set the saving of the matrices G,P and A.

The experiment script produces two ouputs : 
* A .npz file with the G, P and A matrices (one file for each sample if the second argument is set to `True`)
* A .json file containing the scores obtained based on the ground truth (and informations on the input used). One file is used for all samples.


For now it is supported for NMF only and the user needs to provide artificial data with ground truths

## Todo
* Integrate xraylib to generate a more accurate `xrays.json`
* Integrate a database to calculate absorption coefficients
* Integrate detector artefacts 

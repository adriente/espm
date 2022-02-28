# This README needs an update

# Simplex non-negative matrix factorization for Electron Microscopy
This project contains the code to create artificial Energy dispersive X-ray spectroscopy (EDXS) data and to perform hyperspectral unmixing on EDXS spectrum images.


## Installation
Be sure to have all files by initialazing the submodule. Not necessary anymore
```
git submodule update --init --recursive
```

Then, you can install this package using
```
pip install cython
pip install .
```

If you want to develop, please use the option
```
pip install ."testing" 
```

# Set up the virtual env

To be done

# Modifying hyperspy 

Once your virtual env has started search for the folder where hyperspy is installed : 
```
pipenv --venv
```

Go to this location + ``/lib/python3.X/site-packages/hyperspy`` and open the ``hyperspy_extension.yaml``.
In that file add the following lines : 
```
  EDXSsnmfem:
    signal_type: "EDXSsnmfem"
    signal_dimension: 1
    dtype : real
    lazy: False
    module: snmfem.datasets.spim
```


## Installation as a developer (What you should do Adrien)
After setting up your virtual environnement, simply run 
```
git submodule update --init --recursive
pip install cython
pip install -e ".[testing]"
```

You can then simply run the tests with 
```
pytest esmpy
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

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

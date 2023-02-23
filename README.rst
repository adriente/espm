espm: The Electron Spectro-Microscopy Python Library
=====================================================

.. image:: https://readthedocs.org/projects/espm/badge/?version=latest
    :target: https://espm.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

This library contains algorithms to perform non-negative matrix factorization with 
diverse regularisation (e.g. Laplacian or L1) and contraints (e.g. simplex).

It is specialized for Electron Microscopy applications. It contains code to create artificial 
Energy dispersive X-ray spectroscopy (EDXS) data and to perform hyperspectral unmixing on 
EDXS spectrum images.

Installation
------------

You can install this package from PyPi using::

    $ pip install espm

If you want to develop, please use the option::

    $ git clone https://github.com/adriente/espm.git
    $ cd espm
    $ pip install cython
    $ pip install -e ".[dev]" 

Getting started
---------------
Generate the synthetic dataset. Run the script::

    $ python experiments/generate_synthetic_dataset.py


Documentation
-------------

The documentation is available at https://espm.readthedocs.io/en/latest/

You can get started with the following notebooks:

* https://espm.readthedocs.io/en/latest/introduction/notebooks/api.html
* https://espm.readthedocs.io/en/latest/introduction/notebooks/toy-problem.html

TODOs
-----

Here is a list of things that we need to do before the version 0.2.0, which will be the first
official release of the library. The code is already available on github at the following address:  
https://github.com/adriente/espm.git 
A draft of the documentation is available at: https://espm.readthedocs.io/en/latest/

* Update the line 40 of `doc/introduction/index.rst` (@Adrien)
* Make some doc for the dataset module (just the minimum) (@Adrien)
    - Nati: I started doint it. I suggest to put all the function that create samples in dataset.samples.
* Toy dataset: create model class, change outputs, adapts function (@Nati) Done!
* Separate the spectral and spacial parts (@Adrien)
    - Move generate_EDXS_phases to models
    - Create a modules for weights (Nati: I started doing it)
* Complete the general doc in `doc/introduction/index.rst` (@Adrien)
* Clarify the code for the estimator: remove the L2 loss (@Nati) Done!
* Add the general problem that we solve in the doc (@Nati) Done!
* Update the ML notebook with more explanations (@Nati) Done!
* Check that the doc is somehow understanable and sufficiently complete (@Sebastian)
* Add the reference to the dataset paper (@Nati) 
* add the reference to the algorithm paper (@Nati) 

@Adrien: Check the code in espm.weights. If you copy this code, you can generate figures for the doc. 
It seems that you need to have the title `Examples` for the system to work. `Example` does not work!

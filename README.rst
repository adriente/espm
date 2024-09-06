espm: The Electron Spectro-Microscopy Python Library
=====================================================

.. image:: https://readthedocs.org/projects/espm/badge/?version=latest
    :target: https://espm.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

The espm package is designed for simulation and physics-guided NMF decomposition of hyperspectral data.
Even though the package is mainly centered around electron spectro-microscopy applications, custom models can be implemented for other type of data.
Currently espm supports the simulation and analysis of simultaneous scanning transmission electron microscopy and energy dispersive X-ray spectroscopy (STEM / EDXS). 
In future implementation, we will try to extend the package to support electron energy loss spectroscopy (EELS).

This library is integrated as much as possible in the `hyperspy <https://hyperspy.org>` and `scikit-learn <https://scikit-learn.org>` frameworks.

The main components of the package are:
- The simulation of STEM-EDXS datasets using :mod:`espm.datasets` which combines :mod:`espm.weights` for the simulation of spatial distributions and :mod:`èspm.models` for the simulation of spectra.
- The hyperspectral unmixing of STEM-EDXS spectrum images using :mod:`espm.estimators`. This module contains algorithms to perform non-negative matrix factorization with diverse regularisation (e.g. Laplacian or L1) and contraints (e.g. simplex).
- The :mod:`espm.models` module can also be used to perform a physics-guided decomposition of STEM-EDXS datasets.

Installation
------------

You can install this package from PyPi using::

    $ pip install espm

If you want to develop, please use the option::

    $ git clone https://github.com/adriente/espm.git
    $ cd espm
    $ pip install cython
    $ pip install -e ."[dev]" 

If you get issues regarding pandoc when using `make doc`, you can install it using::

    $ sudo apt-get install pandoc

or::
    
    $ conda install pandoc

Recommended Installation
------------------------

We recommend to install the package in a virtual environment using conda::

    $ conda create -n espm python=3.11
    $ conda activate espm
    $ pip install espm

We also recommend that you install the following packages in the same environment::

    $ pip install pyqt5
    $ pip install hyperspy_gui_traitsui
    $ pip install hyperspy_gui_ipywidgets

It is especially useful for the interactive plotting in the notebooks.

Getting started
---------------
Try the api.ipynb notebook in the `notebooks` folder.


Documentation
-------------

The documentation is available at https://espm.readthedocs.io/en/latest/

You can get started with the following notebooks:

* Simulate STEM-EDXS data : https://espm.readthedocs.io/en/latest/introduction/notebooks/generate_data.ipynb
* Physics-guided decomposition (ESpM-NMF) STEM-EDXS data : https://espm.readthedocs.io/en/latest/introduction/notebooks/api.html
* Tests of the ESpM-NMF with a toy dataset : https://espm.readthedocs.io/en/latest/introduction/notebooks/toy-problem.html

CITING
------

If you use this library, please cite the following paper::

    @article{teurtrie2023espm,
    title={espm: A Python library for the simulation of STEM-EDXS datasets},
    author={Teurtrie, Adrien and Perraudin, Nathana{\"e}l and Holvoet, Thomas and Chen, Hui and Alexander, Duncan TL and Obozinski, Guillaume and H{\'e}bert, C{\'e}cile},
    journal={Ultramicroscopy},
    pages={113719},
    year={2023},
    publisher={Elsevier}
    }

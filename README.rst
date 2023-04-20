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
Try the api.ipynb notebook in the `notebooks` folder.


Documentation
-------------

The documentation is available at https://espm.readthedocs.io/en/latest/

You can get started with the following notebooks:

* https://espm.readthedocs.io/en/latest/introduction/notebooks/api.html
* https://espm.readthedocs.io/en/latest/introduction/notebooks/toy-problem.html

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

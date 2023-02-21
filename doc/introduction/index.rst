=========================
Introduction to the espm
=========================

This tutorial will show you the basic operations of the toolbox. After
installing the package with pip, start by opening a python shell, e.g.
a Jupyter notebook, and import `espm`. The package `espm` is built on
top of the `hyperspy` and the `scikit-learn` packages. 

The `hyperspy` package is a Python library for multidimensional data analysis.
It provides the base framework to handles our data. The `scikit-learn` package
is a Python library for machine learning. It provides the base framework to
for the Non Negative Matrix Factorization (NMF) algorithms develeoped in this
package.

.. plot::
    :context: reset

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import hyperspy.api as hs
    >>> import espm

Datasets
--------

Let us start by creating some data to play with this tutorial. We can generating 
artificial datasets using the following lines:

.. plot::
    :context: close-figs

    >>> import espm.datasets as ds
    >>> ds.generate_built_in_datasets(seeds_range=5)
    >>> spim = ds.load_particules(sample = 0)
    >>> spim.change_dtype('float64')

Here the object `spim` is of the :class:`hyperspy._signals.signal1d.Signal1D`.
This object has different useful attributes and methods. For example, 
@ADRIEN --- summarize here some of them

Factorization
-------------

Taking the non-negative matrix factorization (NMF) is done with the following:

.. plot::
    :context: close-figs
    
    >>> out = spim.decomposition(3)
    >>> spim.plot_decomposition_loadings(3)
    >>> spim.plot_decomposition_factors(3)

It will use the algorithm developped in this contribution_.

.. _link: https://link-to-the-paper.com


.. note::
    Please see the review article `espm : a Python library for the simulation 
    of STEM-EDXS datasets` for an overview of
    the methods this package leverages.



=========================
List of example notebooks
=========================

To go deeper, we invite you to consult the following notebooks.

.. nbgallery::
   notebooks/api
   notebooks/generate_data
   notebooks/toy-problem



   

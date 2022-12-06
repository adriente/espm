=========================
Introduction to the esmpy
=========================

This tutorial will show you the basic operations of the toolbox. After
installing the package with pip, start by opening a python shell, e.g.
a Jupyter notebook, and import the PyGSP. We will also need NumPy to create
matrices and arrays.

.. plot::
    :context: reset

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import hyperspi.api as hs

We then set default plotting parameters. We're using the ``matplotlib`` backend
to embed plots in this tutorial. 

.. plot::
    :context: close-figs

    >>> plotting.BACKEND = 'matplotlib'
    >>> plt.rcParams['figure.figsize'] = (10, 5)

Datasets
--------

We are going to need some data to play with this tutorial. We can generating 
artificial datasets using the following lines:



.. plot::
    :context: close-figs
    >>> import esmpy.datasets as ds
    >>> ds.generate_built_in_datasets(seeds_range=5)
    >>> spim = ds.load_particules(sample = 0)

Here the object `spim` is of the :class:`hyperspy._signals.signal1d.Signal1D`.

.. plot::
    :context: close-figs
    >>> out = spim.decomposition(3)
    >>> spim.plot_decomposition_factors(3)

This is a link_.

.. _link: https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)

.. note::
    Please see the review article `esmpy : a Python library for the simulation 
    of STEM-EDXS datasets`_ for an overview of
    the methods this package leverages.

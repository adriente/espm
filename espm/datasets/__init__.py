r"""
The :mod:`espm.datasets` module implements the functions that combines the spatial distributions generated from :mod:`espm.weights` and the spectra generated from :mod:`espm.models` into a 3D dataset.
This part of the espm package manages the integration into the hyperspy framework : 
- the datasets and their metadata are stored as hyperspy signals (.hspy).
- the :mod:`espm.eds_spim` module implements the :class:`EDS_espm` class, which is a subclass of the :class:`hyperspy.signals.Signal1D` class.

Using the :class:`EDS_espm` class, the user can easily use most of the hyperspy functionalities (e.g. plotting, fitting, decomposition, etc.) as well as the espm functionalites on their experimental and simulated data.
    
.. note::

    For now espm supports only the signals modeled as EDS data but we aim at implementing the signals corresponding to EELS data too.

"""

from espm.datasets.base import generate_dataset
from espm.datasets.eds_spim import EDS_espm
from espm.datasets.built_in_EDXS_datasets import *
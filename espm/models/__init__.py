r"""
The :mod:`espm.models` module implements the model objects that are used to simulate features of the data (e.g. in EDXS it corresponds to spectra).
The models are also used for the creation of the G matrix, which can be later used for the decomposition of data using NMF.

.. note::

    For now espm supports only the modelling of EDS data but we aim 
    at supporting the modelling of EELS data too.

"""

from espm.models.base import PhysicalModel, ToyModel
from espm.models.edxs import EDXS
r"""
The :mod:`espm.models` module implements the physical modelling 
that is used both for the creation of the G matrix and for the 
simulation of data.

.. note::

    For now espm supports only the modelling of EDS data but we aim 
    at supporting the modelling of EELS data too.

"""

from espm.models.base import PhysicalModel, ToyModel
from espm.models.edxs import EDXS
r"""
The :mod:`pyesm.models` module implements the physical modelling that is used both for the creation of the G matrix and for the simulation of data.

.. note::

    For now pyesm supports only the modelling of EDS data but we will aim at supporting the modelling of EELS data too.

"""

from pyesm.models.base import PhysicalModel
from pyesm.models.edxs import EDXS
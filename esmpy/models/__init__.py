r"""
The :mod:`esmpy.models` module implements the physical modelling that is used both for the creation of the G matrix and for the simulation of data.

..note:: 
    For now esmpy supports only the modelling of EDS data but we will aim at supporting the modelling of EELS data too.

"""

from esmpy.models.base import PhysicalModel
from esmpy.models.edxs import EDXS
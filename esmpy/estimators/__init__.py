# -*- coding: utf-8 -*-

r""" NMF Estimators.

The :mod:`esmpy.estimators` module implements different NMF algorithms.

The class :mod:`esmpy.estimators` is an abstract class for all NMF algorithms. It implements the fit and transform methods.
The fit method is implemented in the abstract class and calls the `_iteration` method which is implemented in the child classes. 
The transform method is implemented in the child classes.

"""
from esmpy.estimators.base import NMFEstimator
from esmpy.estimators.smooth_nmf import SmoothNMF
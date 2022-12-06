# -*- coding: utf-8 -*-

"""
Test suite for the docstrings of the esmpy package.
"""

import os
import doctest


def gen_recursive_file(root, ext):
    for root, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(ext):
                yield os.path.join(root, name)


def test_docstrings(root, ext, setup=None):
    files = list(gen_recursive_file(root, ext))
    return doctest.DocFileSuite(*files, setUp=setup, tearDown=teardown,
                                module_relative=False)


def setup(doctest):
    import numpy
    import esmpy
    import hyperspy.api
    doctest.globs = {
        'utils': esmpy.utils,
        'np': numpy,
        'hs': hyperspy.api
    }


def teardown(doctest):
    """Close matplotlib figures to avoid warning and save memory."""
    import esmpy
    esmpy.utils.close_all()

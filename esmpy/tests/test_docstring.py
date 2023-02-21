# -*- coding: utf-8 -*-

"""
Test suite for the docstrings of the pyesm package.
"""

import os
import doctest


def gen_recursive_file(root, ext):
    for root, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(ext):
                yield os.path.join(root, name)


def func_test_docstrings(root, ext, setup=None):
    files = list(gen_recursive_file(root, ext))
    return doctest.DocFileSuite(*files, setUp=setup, tearDown=teardown,
                                module_relative=False)


def setup(doctest):
    import numpy
    import hyperspy.api
    doctest.globs = {
        'np': numpy,
        'hs': hyperspy.api
    }


def teardown(doctest):
    """Close matplotlib figures to avoid warning and save memory."""
    from pyesm.utils import close_all
    close_all()


def test_docstrings_pyesm():
    # Docstrings from API reference.
    func_test_docstrings('pyesm', '.py', setup)


def test_docstrings_rst():
    # Docstrings from tutorials. No setup to not forget imports.
    func_test_docstrings('.', '.rst')



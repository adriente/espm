============
Contributing
============

Contributions are welcome, and they are greatly appreciated! The development of
this package takes place on `GitHub <https://github.com/adriente/espm>`_.
Issues, bugs, and feature requests should be reported `there
<https://github.com/adriente/espm/issues>`_.
Code and documentation can be improved by submitting a `pull request
<https://github.com/adriente/espm/pulls>`_. Please add documentation and
tests for any new code.

The package can be set up (ideally in a fresh virtual environment) for local
development with the following::

    $ git clone https://github.com/adriente/espm.git
    $ pip install -e ".[dev]"

You can improve or add functionality in the ``espm`` folder, along with
corresponding unit tests in ``espm/tests/test_*.py`` (with reasonable
coverage).
If you have a nice example to demonstrate the use of the introduced
functionality, please consider adding a notebook in ``doc/introduction/notebooks``.
In general, this can be done as a relative symbolic link to a notebook in the
``notebooks`` folder.

Update ``README.rst`` and ``CHANGELOG.rst`` if applicable.

After making any change, please run the tests, and build the
documentation with the following ::

    $ make test
    $ make doc

To iterate faster, you can partially run the test suite, at various degrees of
granularity, as follows::

   $ pytest espm.test.test_utils.py
   $ pytest espm.test.test_estimators.py

Making a release
----------------

#. Update the version number and release date in ``setup.py``,
   ``espm/__init__.py`` and ``CHANGELOG.rst``.
#. Create a git tag with ``git tag -a v0.2.0 -m "espm v0.2.0"``.
#. Push the tag to GitHub with ``git push --tag``. The tag should now
   appear in the releases and tags tab.
#. `Create a release <https://github.com/adriente/esmpy/releases/new>`_ on
   GitHub and select the created tag. 
#. Build the distribution with ``make dist`` and check that the
   ``dist/espm-0.2.0.tar.gz`` source archive contains all required files. The
   binary wheel should be found as ``dist/espm-0.2.0.py3-none-any.whl``.
#. Test the upload and installation process::

    $ twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    $ pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple emspy

#. Build and upload the distribution to the real PyPI with ``make release``.


Repository organization
-----------------------

::

  LICENSE.txt         Project license
  *.rst               Important documentation
  Makefile            Targets for make
  setup.py            Meta information about package (published on PyPI)
  .gitignore          Files ignored by the git revision control system
  .travis.yml         Defines testing on Travis continuous integration

  espm/              Contains the modules (the actual toolbox implementation)
   __init.py__        Load modules at package import
   *.py               One file per module
  espm/tables/       Contains data tables for the EDXS and EDSS models
   __init.py__        Load modules at package import
   *.json             data_files for the package  
   *.txt              data_files for the package  

  espm/tests/        Contains the test suites (will be distributed to end user)
   __init.py__        Load modules at package import
   test_*.py          One test suite per module
   test_docstrings.py Test the examples in the docstrings (reference doc)

  doc/                Package documentation
   conf.py            Sphinx configuration
   index.rst          Documentation entry page
   *.rst              Include doc files from root directory

  doc/reference/      Reference documentation
   index.rst          Reference entry page
   *.rst              Only directives, the actual doc is alongside the code

  doc/introduction/
   index.rst          Tutorials entry page
   *.rst              One file per tutorial

  doc/introduction/notebooks/
   *.ipynb            One notebook per tutorial
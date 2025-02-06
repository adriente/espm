=========
Changelog
=========

All notable changes to this project will be documented in this file.
The format is based on `Keep a Changelog <https://keepachangelog.com>`_
and this project adheres to `Semantic Versioning <https://semver.org>`_.


0.1.4 (2023-02-22)
------------------

First inofficial release

* First version of the package on PyPi
* All main functionality tested
* Basic documentation


0.2.0 (2023-04-20)
------------------

Second inofficial release

* All important part of the documentation are present
* This is a test before the first official release 1.0.0

0.2.1 (2023-04-21)
------------------

Third inofficial release

* Test the final process for the first official release 1.0.0

1.0.0 (2023-04-21)
------------------

First official release

* All important part of the documentation are present
* All main functionality tested

1.0.1 (2023-06-12)
------------------

First patch release

* New notebook to reproduce the results of the paper
* Fixing some minor bugs
* Adding fixed stoichiometries in the G matrix

1.1.0 (2024-04-19)
------------------

First major update

In the following we refer to an instance of `espm.datasets.EDSespm` as `spim`.

Conceptual changes : 

* In ESpM-NMF 

Syntax changes :

* In `spim.build_G()`, the keyword argument to separate high and low energy lines in the G matrix is now `elements_dict` instead of `reference_elt`. The dictionary allows atomic number or chemical symbol notation.
* When calling `espm.estimators.SmoothNMF`, prefer the use of `spim.model` instead of `spim.G` to pass the EDXS modelling.
* The metadata of the `spim` can be set in two ways : 
    * with the set functions `spim.set_analysis_parameters` that replace the existing metadata.
    * with the add functions `spim.add_analysis_parameters` that do not replace the existing metadata.

Conceptual changes :

* The `espm.estimators.SmoothNMF` object can take G as a keyword argument. The argument accepts either a `numpy.ndarray` (that can be called with `spim.G`) or an instance of `espm.models.EDXS` (that can be called with `spim.model`).
* When calling, for example, `spim.build_G(elements_dict = {'Fe' : 4.0})` the Fe lines are split between high and low energy lines. Now, in the simplex constraint and in the quantification, the low energy lines are ignored.

New features :

* The `espm.datasets.EDSespm.estimate_best_binning` can be used to estimate the best binning factor to apply on the data before performing the ESpM-NMF decompostion. Use the output (`bb`) of this function in `spim.rebin(scale = bb )` to apply the binning.
* An alternative init for the ESpM-NMF decompostion can be activated by executing `spim.custom_init = True`. It works only when using `G = spim.model` in `espm.estimators.SmoothNMF`.
* The `spim.print_concentration_report` was improved thanks to the `prettytable` package. Statistical errors on the quantifications are now displayed.

1.1.1 (2024-04-25)
------------------

Patch of the 1.1.0

* Fixing the version of traits, hyperspy 2.0.0 is not compatible with traits 6.0.0 and above.

1.1.2 (2024-09-09)
------------------

Patch of the 1.1.1

* Switching back to traits 6.0.0 and above that is now compatible with hyperspy 2.0.0 and above.
* Update of the notebooks to be compatible with the new version of the package.
* Adding a filter to the generation of the G matrix so that lines outside the energ range are correctly ignored.

1.1.3 (2025-02-05)
------------------

Patch of the 1.1.2

Major changes : 

* The `espm.datasets.EDSespm` now inherits from the `exspy.signals.EDSTEMSpectrum` enabling all the corresponding functionalities.
* EDXS model : It is possible to add ignored elements such as Cu (`ignored_elements = ['Cu']`), so that it is fitted independently of the mass-thickness. It helps the fitting process for stray X-rays from the sample environment such as the Cu grid.
* Fitting of 1D spectra : Added functionalities to fit and quantify 1D spectra.
* Energy windows on which the bremsstrahlung is fitted can be set with the `select_background_windows` gui method. It also displays the fitted background.

Syntax changes :

* The methods setting the metadata that are relevant for modelling are now functionning like exspy. The microscope parameters are set using exspy's `set_microscope_parameters` method. The metadata of the `espm.datasets.EDSespm` object can be set using the `set_analysis_parameters` method.
* Currently, the beta_tilt has to be set using `data_object.metadata.Acquisition_instrument.TEM.Stage.beta_tilt = value`. 

Technical changes : 

* Update of the installation process by using pyproject.toml
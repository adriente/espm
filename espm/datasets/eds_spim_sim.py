from espm.datasets.eds_spim import EDSespm

class EDSespmSimulated(EDSespm) : 

    _signal_type = "EDS_espm_Simulated"

    def __init__ (self,*args,**kwargs) : 
        super().__init__(*args,**kwargs)
        self._phases = None
        self._maps = None
        self._maps_2d = None
        self._Xdot = None

    ##############
    # Properties #
    ##############

    def _set_default_analysis_params(self) :
        # TODO : make them fetch preferences from the user 
        super()._set_default_analysis_params()
        md = self.metadata
        md.Signal.signal_type = "EDS_espm_Simulated"

    @property
    def maps (self) :
        r"""
        Ground truth of the spatial distribution of the phases in the form of a 3D array of shape (shape_2d[0],shape_2d[1],n_phases), if available.
        """ 
        if self._maps is None : 
            self._maps = self.build_ground_truth()[1]
        return self._maps

    @property
    def phases (self) : 
        r"""
        Ground truth of the spectra of the phases in the form of a 2D array of shape (n_phases,n_features), if available.
        """
        if self._phases is None : 
            self._phases = self.build_ground_truth()[0]
        return self._phases

    @property
    def maps_2d (self) : 
        r"""
        Ground truth of the spatial distribution of the phases in the form of a 2D array of shape (shape_2d[0]*shape_2d[1],n_phases), if available.
        """
        if self._maps_2d is None : 
            self._maps_2d = self.build_ground_truth(reshape = False)[1]
        return self._maps_2d
    
    
    @property
    def Xdot (self) : 
        r"""
        The ground truth in the form of a 3D array of shape (shape_2d[0],shape_2d[1],n_features), if available.
        """
        if self._Xdot is None : 
            try : 
                self._Xdot = self.phases @ self.maps
            except AttributeError : 
                print("This dataset contains no ground truth. Nothing was done.")
        return self._Xdot
    
    #################
    # Decomposition #
    #################

    def decomposition(
        self,
        normalize_poissonian_noise=True,
        navigation_mask=1.0,
        closing=True,
        *args,
        **kwargs,
    ):
        """Apply a decomposition to a dataset with a choice of algorithms.

        The results are stored in ``self.learning_results``.

        Read more in the :ref:`User Guide <mva.decomposition>`.

        Parameters
        ----------
        normalize_poissonian_noise : bool, default True
            If True, scale the signal to normalize Poissonian noise using
            the approach described in [*]_.
        navigation_mask : None or float or boolean numpy array, default 1.0
            The navigation locations marked as True are not used in the
            decomposition. If float is given the vacuum_mask method is used to
            generate a mask with the float value as threshold.
        closing: bool, default True
            If true, applied a morphologic closing to the mask obtained by
            vacuum_mask.
        algorithm : {"SVD", "MLPCA", "sklearn_pca", "NMF", "sparse_pca", "mini_batch_sparse_pca", "RPCA", "ORPCA", "ORNMF", custom object}, default "SVD"
            The decomposition algorithm to use. If algorithm is an object,
            it must implement a ``fit_transform()`` method or ``fit()`` and
            ``transform()`` methods, in the same manner as a scikit-learn estimator.
        output_dimension : None or int
            Number of components to keep/calculate.
            Default is None, i.e. ``min(data.shape)``.
        centre : {None, "navigation", "signal"}, default None
            * If None, the data is not centered prior to decomposition.
            * If "navigation", the data is centered along the navigation axis.
              Only used by the "SVD" algorithm.
            * If "signal", the data is centered along the signal axis.
              Only used by the "SVD" algorithm.
        auto_transpose : bool, default True
            If True, automatically transposes the data to boost performance.
            Only used by the "SVD" algorithm.
        signal_mask : boolean numpy array
            The signal locations marked as True are not used in the
            decomposition.
        var_array : numpy array
            Array of variance for the maximum likelihood PCA algorithm.
            Only used by the "MLPCA" algorithm.
        var_func : None or function or numpy array, default None
            * If None, ignored
            * If function, applies the function to the data to obtain ``var_array``.
              Only used by the "MLPCA" algorithm.
            * If numpy array, creates ``var_array`` by applying a polynomial function
              defined by the array of coefficients to the data. Only used by
              the "MLPCA" algorithm.
        reproject : {None, "signal", "navigation", "both"}, default None
            If not None, the results of the decomposition will be projected in
            the selected masked area.
        return_info: bool, default False
            The result of the decomposition is stored internally. However,
            some algorithms generate some extra information that is not
            stored. If True, return any extra information if available.
            In the case of sklearn.decomposition objects, this includes the
            sklearn Estimator object.
        print_info : bool, default True
            If True, print information about the decomposition being performed.
            In the case of sklearn.decomposition objects, this includes the
            values of all arguments of the chosen sklearn algorithm.
        svd_solver : {"auto", "full", "arpack", "randomized"}, default "auto"
            If auto:
                The solver is selected by a default policy based on `data.shape` and
                `output_dimension`: if the input data is larger than 500x500 and the
                number of components to extract is lower than 80% of the smallest
                dimension of the data, then the more efficient "randomized"
                method is enabled. Otherwise the exact full SVD is computed and
                optionally truncated afterwards.
            If full:
                run exact SVD, calling the standard LAPACK solver via
                :py:func:`scipy.linalg.svd`, and select the components by postprocessing
            If arpack:
                use truncated SVD, calling ARPACK solver via
                :py:func:`scipy.sparse.linalg.svds`. It requires strictly
                `0 < output_dimension < min(data.shape)`
            If randomized:
                use truncated SVD, calling :py:func:`sklearn.utils.extmath.randomized_svd`
                to estimate a limited number of components
        copy : bool, default True
            * If True, stores a copy of the data before any pre-treatments
              such as normalization in ``s._data_before_treatments``. The original
              data can then be restored by calling ``s.undo_treatments()``.
            * If False, no copy is made. This can be beneficial for memory
              usage, but care must be taken since data will be overwritten.
        **kwargs : extra keyword arguments
            Any keyword arguments are passed to the decomposition algorithm.


        Examples
        --------
        >>> s = exspy.data.EDS_TEM_FePt_nanoparticles()
        >>> si = hs.stack([s]*3)
        >>> si.change_dtype(float)
        >>> si.decomposition()

        See also
        --------
        vacuum_mask

        References
        ----------
        .. [*] M. Keenan and P. Kotula, "Accounting for Poisson noise
           in the multivariate analysis of ToF-SIMS spectrum images", Surf.
           Interface Anal 36(3) (2004): 203-212.
        """
        super().decomposition(
            normalize_poissonian_noise=normalize_poissonian_noise,
            navigation_mask=navigation_mask,
            *args,
            **kwargs,
        )
        self.set_signal_type("EDS_espm_Simulated")
    
    ######################################
    # Modelling and simulation functions #
    ######################################

    def build_ground_truth(self,reshape = True) : 
        r"""
        Get the ground truth stored in the metadata of the :class:`EDSespm` object, if available. The reshape arguments can be used to get the ground truth in a form easier to use for machine learning algorithms.

        Parameters
        ----------
        reshape : bool, optional
            If False, the ground truth is returned in the form of a 3D array of shape (shape_2d[0],shape_2d[1],n_phases) and a 2D array of shape (n_phases,n_features).
        
        Returns
        -------
        phases : numpy.ndarray
            The ground truth of the spectra of the phases.
        weights : numpy.ndarray
            The ground truth of the spatial distribution of the phases.
        """
        if "phases" in self.metadata.Truth.Data : 
            phases = self.metadata.Truth.Data.phases
            weights = self.metadata.Truth.Data.weights
            if reshape : 
                phases = phases.T
                weights = weights.reshape((weights.shape[0]*weights.shape[1], weights.shape[2])).T
        else : 
            raise AttributeError("There is no ground truth contained in this dataset")
        return phases, weights
    

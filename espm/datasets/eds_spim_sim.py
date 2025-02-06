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
    

import itertools as itrtls
import numpy as np
import batches as bat


class CrossVal:
    """
    Cross validation class used to determine optimal values of the hyperparameters of the tested algorithm.
    """

    def __init__(self, algo, hypars_dict):
        """
        Creates an instance of cross_validation
        :algo: Instance of the algorithm that is tested (object)
        :hypars_dict: dictionnary of values used for the grid search of hyperparameters. ({"name":[list of values]})
        :results: Stores the values of the results in a dict ({"name values",fitness})
        """
        self.algo = algo
        self.hypars_dict = hypars_dict
        self.results = None

    def hypars_mesh(self):
        """
        Makes a mesh of hyper parameters out of the specified lists
        """
        # develops {"name": [values]} into [{"name" : value}]
        lists_params = []
        for param_name in self.hypars_dict.keys():
            param_range = []
            for elt in self.hypars_dict[param_name]:
                param_range.append({param_name: elt})
            lists_params.append(param_range)
        # makes a mesh of tuples [({"name" : value},...),...]
        params_mesh = list(itrtls.product(*lists_params))
        return params_mesh

    def tuple_to_dict(self, tup):
        """
        Transforms a tuple of dicts into one dict
        """
        res_dict = {}
        for elt in tup:
            res_dict.update(elt)
        return res_dict

    def fitness(self, x, adj_model):
        """
        Absolute difference between a model and the fitted data
        """
        return np.sum(np.abs(adj_model - x))

    def grid_search(self, X, method="checker", n=4):
        """
        Computes the algorithm for each hyperparameters in the mesh and for each batches.
        Returns a dict {"param name" : value, ... , "fitness" : fitness}
        :X: data (np.ndarray)
        :method: batches preparation method (string)
        :n: Optional, number of batches (int) not meaningful for each batches method
        """
        # Prepare the batches according to the selected method
        batches = bat.Batches(X, method=method, n=n)
        batches.prepare_batches()
        results = []

        for hypar_tup in self.hypars_mesh():
            # Sets the hyperparameters for the algorithm
            combined_dict = self.tuple_to_dict(hypar_tup)
            self.algo.set_hypars(**combined_dict)
            fitness = 0

            for i in range(len(batches.batches)):
                # A model is calculated for one batch
                self.algo.fit(batches.batches[i])
                # This model is applied to the other batch
                # spectra are kept constant, the abundances are determined with the apply model method
                adj_model = batches.apply_model(self.algo, i)
                # The resulting fitness will be the sum over the batches
                fitness += self.fitness(batches.batches[i], adj_model)
                # Re-use the previous iteration as initial values for the next one
                # This function has to be updated to take b_matr into account
                self.algo.set_init_pars(init_a=self.algo.a, init_p=self.algo.p)
            combined_dict.update({"fitness": fitness})
            results.append(combined_dict)
        self.results = results

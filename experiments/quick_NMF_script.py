from espm.conf import DATASETS_PATH, BASE_PATH
from pathlib import Path
import hyperspy.api as hs
import numpy as np
import sys

def run(seed) : 
    path1 = BASE_PATH.parent / Path("experiments/71GPa_experimental_data.hspy")
    sexp = hs.load(str(path1))
    from sklearn.decomposition import NMF
    model = NMF(n_components=3, init='nndsvdar', random_state=seed, beta_loss="kullback-leibler",solver = "mu", tol = 1e-7, max_iter=30000)

    sexp.change_dtype("float64")
    sexp.decomposition(algorithm = model)

    f = sexp.get_decomposition_factors()
    l = sexp.get_decomposition_loadings()

    d = {}
    d["fe0"] = f.data[0]
    d["fe1"] = f.data[1]
    d["fe2"] = f.data[2]
    d["le0"] = l.data[0]
    d["le1"] = l.data[1]
    d["le2"] = l.data[2]

    np.savez("nmf_71Gpa_exp_{}.npz".format(seed),**d)

if __name__ == "__main__" : 
    print(sys.argv[1:])
    seed =  sys.argv[1:]
    run(seed)


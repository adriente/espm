import hyperspy.api as hs
import numpy as np
from pathlib import Path
from espm.utils import rescaled_DH
from experiments import results_string
import numpy.lib.recfunctions as rfn
from espm import measures
import sys
import conf

def metrics_statistics (k,metrics_summary,n_samples) : 
    names_a = []
    names_m = []
    formats = (k)*["float64"]
    for i in range(k) : 
        names_a.append("angle_p{}".format(i))
        names_m.append("mse_p{}".format(i))
    
    angles_array = np.zeros((n_samples,),dtype = {"names" : names_a, "formats" : formats})
    mse_array = np.zeros((n_samples,),dtype = {"names" : names_m, "formats" : formats})
    for j,metrics in enumerate(metrics_summary) :
        for i in range(k) : 
            key_a = "angle_p{}".format(i)
            key_m = "mse_p{}".format(i)
            angles_array[key_a][j] = metrics[0][metrics[2][i]]
            mse_array[key_m][j] = metrics[1][i]

    warn_array = np.zeros((n_samples,),dtype={"names" : ["warning"], "formats" : ["bool"]})
    for j,metrics in enumerate(metrics_summary) : 
        warn_array["warning"][j] = metrics[3]

    return rfn.merge_arrays((angles_array,mse_array,warn_array), flatten = True, usemask = False)



def run_batch (k, folder, init, output, random_state) : 

    n_samples = 6
    k = int(k)
    random_state = int(random_state)
    txt = output + ".txt"
    npz = output + ".npz"

    d = {}
    
    metrics_list = []
    for i in range(n_samples) :
        print("starting sample {}".format(i)) 
        file = Path(folder) / Path("sample_{}.hspy".format(i))
        spim = hs.load(str(file))
        true_spectra = spim.phases.T
        true_maps = spim.maps
        shape_2d = spim.shape_2d
        spim.decomposition(False,algorithm = "NMF", max_iter = 50000, tol = 1e-9, solver = "mu", beta_loss = "kullback-leibler", output_dimension = k,print_info= True, init = init, random_state = random_state)
        factors = spim.get_decomposition_factors().data.T
        loadings = spim.get_decomposition_loadings().data.reshape((k,shape_2d[0]*shape_2d[1]))
        r_factors, r_loadings = rescaled_DH(factors,loadings)
        metrics_list.append(measures.find_min_config(true_maps,true_spectra,r_loadings,r_factors.T))

        d["D_{}".format(i)] = r_factors
        d["H_{}".format(i)] = r_loadings

    summary = metrics_statistics(k,metrics_list,n_samples)

    with open(txt,"a") as f : 
        f.write(results_string({"name" : "NMF"},summary))

    filename = conf.RESULTS_PATH / Path(npz)
    np.savez(filename, **d)

if __name__ == "__main__" : 
    print(sys.argv[1:])
    k, folder, init, output, random_state =  sys.argv[1:]
    run_batch(k, folder, init, output, random_state)
import hyperspy.api as hs
import numpy as np
from pathlib import Path
from snmfem.utils import rescaled_DA
from snmfem.experiments import compute_metrics, results_string
import numpy.lib.recfunctions as rfn
import sys

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

    return rfn.merge_arrays((angles_array,mse_array), flatten = True, usemask = False)



def run_batch (k,true_filename, folder, init, output, random_state) : 
    true_spim = hs.load(true_filename)
    true_spectra = true_spim.phases_2d
    true_maps = true_spim.weights
    shape_2d = true_spim.shape_2d

    n_samples = 3
    k = int(k)
    random_state = int(random_state)
    
    metrics_list = []
    for i in range(n_samples) :
        print("starting sample {}".format(i)) 
        file = Path(folder) / Path("sample_{}.hspy".format(i))
        spim = hs.load(str(file))
        spim.decomposition(True,algorithm = "NMF", max_iter = 50000, tol = 1e-9, solver = "mu", beta_loss = "kullback-leibler", output_dimension = k,print_info= True, init = init, random_state = random_state)
        factors = spim.get_decomposition_factors().data.T
        loadings = spim.get_decomposition_loadings().data.reshape((k,shape_2d[0]*shape_2d[1]))
        r_factors, r_loadings = rescaled_DA(factors,loadings)
        metrics_list.append(compute_metrics(true_spectra,true_maps,r_factors,r_loadings))

    summary = metrics_statistics(k,metrics_list,n_samples)

    with open(output,"a") as f : 
        f.write(results_string({"name" : "NMF"},summary))

if __name__ == "__main__" : 
    print(sys.argv[1:])
    k,true_filename, folder, init, output, random_state =  sys.argv[1:]
    run_batch(k,true_filename, folder, init, output, random_state)
import numpy as np


def spectral_angle(v1, v2):
    """
    Calculates the angle between two spectra. They have to have the same dimension.
    :v1: (np.array 1D) first spectrum 
    :v2: (np.array 1D) second spectrum 
    """
    v1_u = v1/np.linalg.norm(v1)
    v2_u = v2/np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))*180/np.pi


def mse(map1,map2) :
    """ Mean square error.

    Calculates the mean squared error between two 2D arrays. They have to have the same dimension.
    :map1: (np.array 2D) first array 
    :map2: (np.array 2D) second array 
    """
    return np.sum((map1-map2)**2)

# This function will find the best matching endmember for each true spectrum. 
# This is useful since the A and P matrice are initialized at random. 
# This function works but can probably greatly improved
def find_min_angle (list_true_vectors,list_algo_vectors) :
    # This function calculates all the possible angles between endmembers and true spectra
    # For each true spectrum a best matching endmember is found
    # The function returns the angles of the corresponding pairs
    copy_algo_vectors=list_algo_vectors.copy()
    size=list_algo_vectors[0].shape
    ordered_angles=[]
    for i in range(len(list_true_vectors)) :
        list_angles=[]
        for j in range(len(list_algo_vectors)) :
            list_angles.append(spectral_angle(list_true_vectors[i],list_algo_vectors[j]))
        ind_min=np.argmin(np.array(list_angles))
        list_algo_vectors[ind_min]=1e28*np.ones(size)
        ordered_angles.append(spectral_angle(list_true_vectors[i],copy_algo_vectors[ind_min]))
    return ordered_angles

# This function works but can probably greatly improved
def find_min_MSE (list_true_maps,list_algo_maps) :
    # This function calculates all the possible MSE between abundances and true maps
    # For each true map a best matching abundance is found
    # The function returns the MSE of the corresponding pairs
    copy_algo_maps=list_algo_maps.copy()
    size=list_algo_maps[0].shape
    ordered_maps=[]
    for i in range(len(list_true_maps)) :
        list_maps=[]
        for j in range(len(list_algo_maps)) :
            list_maps.append(mse(list_true_maps[i],list_algo_maps[j]))
        ind_min=np.argmin(np.array(list_maps))
        list_algo_maps[ind_min]=1e28*np.ones(size)
        ordered_maps.append(mse(list_true_maps[i],copy_algo_maps[ind_min]))
    return ordered_maps

# This function gives the residuals between the model determined by snmf and the data that were fitted
def residuals (data,model) :
    X_sum=data.sum(axis=0).sum(axis=0)
    model_sum=model.get_phase_map(0).sum()*model.get_phase_spectrum(0)+model.get_phase_map(1).sum()*model.get_phase_spectrum(1)+model.get_phase_map(2).sum()*model.get_phase_spectrum(2)
    return X_sum-model_sum

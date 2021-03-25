import numpy as np
from H_MCRLLM import half_mcrllm

def use_half_MCR(X):

    # Load all the hierarchical data
    S_H = np.load('data_S_H.npy')
    
    # Load the reference spectra coming from the PCA (step 2)
    S = np.load('data_PCA_endmembers.npy')
    nb_pure = S.shape[0]
    
    # Center and scale to unit variance
    S_H_mean = np.mean(S_H,axis=0)
    S_H_std = np.std(S_H,axis=0)
    S_H = (S_H - S_H_mean)/S_H_std
    
    # Remove (center and scale to unit variance)
    S = S * S_H_std + S_H_mean
        
    decomp = half_mcrllm(X , nb_pure , init = S)
    C = decomp.C
    
    return C,S
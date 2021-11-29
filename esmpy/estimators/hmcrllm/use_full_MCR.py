import numpy as np
from H_MCRLLM import mcrllm

def use_full_MCR(X,nb_iter):

    # Load all the hierarchical data
    S_H = np.load('data_H.npy')
    
    # Load the reference spectra coming from the PCA (step 2)
    S = np.load('data_PCA_select.npy')
    nb_pure = S.shape[0]
    
    # Center and scale to unit variance
    S_H_mean = np.mean(S_H,axis=0)
    S_H_std = np.std(S_H,axis=0)
    S_H = (S_H - S_H_mean)/S_H_std
    
    # Remove (center and scale to unit variance)
    S = S * S_H_std + S_H_mean
        
    decomp = mcrllm(X, nb_pure, init = S, nb_iter = nb_iter)
    C,S = decomp.C , decomp.S

    
    return C,S
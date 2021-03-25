# PCA of h_MCR results - choose spectra

import numpy as np
from sklearn import decomposition
from H_MCRLLM.pointselector import SelectFromCollection, Score_plot_3D
import matplotlib.pyplot as plt

def select_with_PCA(S_H):

    print('Hierarchical spectra: ',S_H.shape)
        
    # Center and scale to unit variance
    S_H_mean = np.mean(S_H,axis=0)
    S_H_std = np.std(S_H,axis=0)
    S_H = (S_H - S_H_mean)/S_H_std
    
    #PCA
    # Number of PCA components
    nb_pc = int(input('Number of principal components for PCA (Should be equal or superior to you number of chemical species) :  '))
    pca = decomposition.PCA(n_components=nb_pc)
    pca.fit(S_H)
    
    Spca = pca.transform(S_H)  # spectra expressed in PCA scores (t)
    
    
    a = input('Do you want to see a 3D plot of the PCA scores (y/n) ? ')
    
    plt.ion()
    
    if a == 'y':
        
        unsatisfied = True
        
        while unsatisfied: 
            
            plot = Score_plot_3D(Spca)
        
            while True:
            
                if plot.accepted == True:
                    unsatisfied = False
                    break
                    
                plt.pause(1)  
    
    plt.ioff()
    
    # Number of reference spectra to be found in PCA score space
    print('\n####################\n')
    print('\nCreate reference spectra based on the hierarchical spectra obtained.')
    print('To do so:')
    print('1. You will be asked to select as many spectra as you have chemical species ')
    print('2. Each time, you will to choose PCA scores to plot with (usually t1 and t2) ')
    print('3. Circle with your mouse 1 or more spectra to be combined into a reference spectra')
    print("4. Accept the spectra presented with 'enter' or re-do the selection until satisfication")
    
    plt.ion()
    
    nb_phases = int(input('Number of chemical species in your dataset : '))
    
    Final_spectra = np.zeros([nb_phases,len(S_H_mean)])
    
    
    for i in range(nb_phases):
        
        Sselect = SelectFromCollection(Spca, S_H, S_H_mean, S_H_std)
    
        while True:
        
            if Sselect.accepted == True:
                a = Sselect.ind
                Select = np.mean(S_H[a,:],axis=0)
                #Stored spectra are centered and scaled. They will be corrected later
                Final_spectra[i,:] = Select
                break
        
            plt.pause(1)
        
    
    plt.ioff()
    
    return Final_spectra
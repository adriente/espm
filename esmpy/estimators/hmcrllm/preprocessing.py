# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:08:44 2020

@author: Yannick
"""
import numpy as np


def preprocess_data(xraw):
    
    x_sum = np.sum(xraw, axis=1)
    xcopy = np.copy(xraw)
    check_pix = False
    check_level = False
    deletedpix = 0
    deletedLevels = 0
    
    
    
    
    #we take out every pixels which did not receive any count. We will add them back at the end 
    
    if np.any(x_sum == 0):
        
        print('One or more pixel(s) did not receive any count so they were taken out for the calculation')
            
        check_pix = True
        deletedpix = np.where(x_sum == 0)[0]
        xcopy = np.delete(xraw, deletedpix, axis = 0)
        x_sum = np.delete(x_sum , deletedpix)
        
    
    
    
    #Avoid errors, if an energy level has 0 counts on all pixels, it causes log(0) and the level is meaningless, so we take it out
    
    sum_level = np.sum(xcopy , axis = 0)
    
    if np.any(sum_level == 0):
        
        print('One or more level of energy did not have any count on it so they were taken out for calculation')
        check_level = True
        
        deletedLevels = np.where(sum_level == 0)[0]
        
        xcopy = np.delete(xcopy , deletedLevels , axis = 1 )
        


    
    #We finally normalize each spectra so their sum is zero
        
    x_sum = np.array([np.sum(xcopy, axis=1)]).T
    x = xcopy / x_sum
    
  
    return x , x_sum , deletedpix , deletedLevels , check_pix , check_level
    






def final_process(C, S , deletedpix , deletedLevels, check_pix, check_level):
    
    
    nb_c = np.shape(C)[1]
    
    # We add back the pixels we took out initially. The concentrations are all set to 0

    if check_pix == True:
        
        nb_pix_ori = np.shape(C)[0] + len(deletedpix)
        
        
        C_final = np.zeros([nb_pix_ori, nb_c])
        
        counter = 0
            
        
        for i in range(nb_pix_ori):
            
            if np.any(deletedpix == i):
                
                C_final[i,:] = 0
                counter += 1
            
            else: 
                C_final[i,:] = C[i-counter,:]
                
    
    else:
        C_final = C
    
    
    
    
    #Insert deleted levels back 
    
    if check_level == True:
        
        nb_level_ori = np.shape(S)[1] + len(deletedLevels)
        
        S_final = np.zeros([nb_c , nb_level_ori] )

        counter = 0
            
        
        for i in range(nb_level_ori):
            
            if (i-counter) >= (np.shape(S)[1] - 1) :
                
                S_final[:,i] = S[:, np.shape(S)[1] - 1]
            
            
            elif np.any(deletedLevels == i):
                
                S_final[:,i] = (S[:,i-counter-1] + S[:,i-counter+1])/2
                counter += 1
                
            
            else: 
                S_final[:,i] = S[:,i-counter]
                
        
    else:
        S_final = S
                
        
        
        
    return C_final,S_final
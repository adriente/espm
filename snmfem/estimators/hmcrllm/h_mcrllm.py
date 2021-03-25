# -*- coding: utf-8 -*-
"""
@auteurs : Hugo Caussan, Louis-Philippe Baillargeon and Yannick Poulin-G. 
"""
import numpy as np
from H_MCRLLM import *
from H_MCRLLM.preprocessing import preprocess_data
from H_MCRLLM.mcrllm import * 

class h_MCR_LLM:
    
         
    @classmethod
    def h_mcr_llm(cls, nb_i, min_pixels, max_pixels, Max_level, Xraw):
        
        print('\nBeginning hierarchical initialization\n')
        
        X , _ , _ , _ , _ , _ = preprocess_data(Xraw)
        
        all_s = [] # All spectras
        all_c = [] # All concentrations
        all_p = [] # All pixels (ID)
        positions = np.int64(np.linspace(0,X.shape[0]-1, X.shape[0]).T)
        
        #print("\n### LEVEL 0 ###")
        
        decomp = mcrllm(X, nb_c = 2 , init = 'Kmeans' , nb_iter = nb_i )
        S = decomp.S
        C = decomp.C
        print(C.shape)
        print(S.shape)

        all_s.append(S[0,:])
        all_s.append(S[1,:])
        
        all_c.append(C[:,0])
        all_c.append(C[:,1])
                
        #separate pixel according to higest contribution
        all_p.append(positions[C[:,0]>0.5])
        all_p.append(positions[C[:,1]>0.5])
        
        all_s_final = []
        all_c_final = []
        all_p_final = []
        
        
        for level in range(Max_level):
            
                print("\n\n### LEVEL " + str(level+1) + " ###")
                all_s_loop = []
                all_c_loop = []
                all_p_loop = []
                for i in range (len(all_p)):
                    
                    if all_p[i].shape[0] >= max_pixels: # Si le jeu i contient assez de spectre, on le sépare en deux
                        
                        decomp = mcrllm(X[all_p[i],:], nb_c = 2 , init = 'Kmeans' , nb_iter = nb_i)
                        S = decomp.S
                        C = decomp.C
                        
                        
                        # Si les deux séparations de MCR contiennent assez de spectre, on peut les garder pour le prochain niveau
                        if all_p[i][C[:,0]>0.5].shape[0] > min_pixels:
                            all_s_loop.append(S[0,:])
                            all_c_loop.append(C[:,0])
                            all_p_loop.append(all_p[i][C[:,0]>0.5])
                            
                         
                        if all_p[i][C[:,1]>0.5].shape[0] > min_pixels:
                            all_s_loop.append(S[1,:])
                            all_c_loop.append(C[:,1])
                            all_p_loop.append(all_p[i][C[:,1]>0.5])
                            
                        
                    else: # Si le jeu i contient une quantité admissible de spectre, on le garde 
                        
                        all_s_final.append(all_s[i])
                        all_c_final.append(all_c[i])
                        all_p_final.append(all_p[i])
    
                  
                all_s = np.copy(all_s_loop)
                all_c = np.copy(all_c_loop)
                all_p = np.copy(all_p_loop)
                
                
            
#                print("Loop :")
#                for set in all_p:
#                    print(set.shape[0], end = ' ')
#                print("\nFinal :")
#                for set in all_p_final:
#                    print(set.shape[0], end = ' ')
                
        
        all_s_final = np.array(all_s_final)
        all_p_final = np.array(all_p_final)
        
        
        print('\nHierarchical initialization completed')
        
        if all_s.shape[0] == 0:
            return all_s_final
        if all_s_final.shape[0] == 0:
            return all_s
        
        return np.vstack((all_s, all_s_final))
    
    

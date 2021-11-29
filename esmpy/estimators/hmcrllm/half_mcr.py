# DEMI MCR LLM

"""
Coded by Jeffrey Byrns
Adapted by Ryan Gosselin
"""

# import packages
import numpy as np
from scipy.optimize import minimize
from functools import partial
from H_MCRLLM.preprocessing import preprocess_data,final_process



from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import pysptools.eea.eea
import math
from tqdm import tqdm


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 



"""All intitialisation methods to extract the endmembers"""
################################################################################################################################

class KmeansInit: #Kmeans initialisation
    @classmethod
    def initialisation(cls, x, nb_c, n_init=10):

        s = KMeans(n_clusters=nb_c, n_init = n_init).fit(x).cluster_centers_

        return s

################################################################################################################################

class MBKmeansInit: #Mini Batch Kmeans (plus rapide mais moins robuste)
    @classmethod
    def initialisation(cls, x, nb_c):
        
        ctr_k = MiniBatchKMeans(n_clusters = nb_c).fit(x)
        s = ctr_k.cluster_centers_
        
        return s
################################################################################################################################

class NFindrInit:
    @classmethod
    def initialisation(cls, x, nb_c):
        size0 = x.shape[0]
        size1 = x.shape[1]
        xr = np.reshape(x, (1, size0, size1) ) # NFindr travaille avec des jeux de données 3D...
        s = pysptools.eea.NFINDR.extract(cls, xr, nb_c)
        
        return s

################################################################################################################################
   
class RobustNFindrInit:
    @classmethod
    def initialisation(cls, x, nb_c, fraction = 0.1, min_spectra = 50, nb_i = 50): # Initialisaiton KMeans + NMF / Voir avec Ryan Gosselin pour les explications et évolutions
        """
        fraction : Fraction du jeu de données par échantillon
        min_spectra : minimum de spectre pour garder un échantillong
        nb_i : nombre d'échantillons à créer
        """
        
        
        def km(x, nb_c):
            k = KMeans(n_clusters=nb_c).fit(x)
            IDX = k.labels_
            C = k.cluster_centers_
            return IDX, C
        
        
        s1 = x.shape[0]
        
        fX = math.ceil(s1*fraction)

        
        BESTC = np.array(())
       
        DETC = 0
        for i in tqdm(range(nb_i)):
            
            randomVector = np.random.choice(s1, fX, replace = False)# Create random vector with unique values
            sampX = x[randomVector,:]#Pick a sample in x according to the randomVector
            
            #Run Kmeans
            IDX, C = km(sampX, nb_c)
            
            #Check Number of pixels in each kmeans centroid
            U, nbU = np.unique(IDX, return_counts=True);
            
            
            if min(nbU) > min_spectra: #Do not keep this bootstrap if too few pixels fall in a category
                a = np.zeros((nb_c,1)) + 1 #Start NMF
                C1 = np.column_stack((a, C))
                CC = C1@C1.T
                detc = np.linalg.det(CC)
                if detc > DETC:
                    DETC = detc
                    BESTC = np.copy(C)
                    #print(nbU)
                                           
        return BESTC
    
################################################################################################################################

class RobustNFindrV2Init:
    @classmethod
    def initialisation(cls, x, nb_c, fraction = 0.1, min_spectra = 50, nb_i = 50):
        """
        fraction : Fraction du jeu de données par échantillon
        min_spectra : minimum de spectre pour garder un échantillong
        nb_i : nombre d'échantillons à créer
        """
        
        def km(x, nb_c):
            k = KMeans(n_clusters=nb_c).fit(x)
            IDX = k.labels_
            C = k.cluster_centers_
            return IDX, C
        
        s1 = x.shape[0]
        f = fraction # Fraction to keep in each bootstrap
        fX = math.ceil(s1*f)

        allS = np.array(())


        for i in tqdm(range(nb_i)):
            randomVector = np.random.choice(s1, fX, replace = False)# Create random vector with unique values
            sampX = x[randomVector,:]#Pick a sample in x according to the randomVector
    
        #Run Kmeans
            IDX, C = km(sampX, nb_c)
        
            #Check Number of pixels in each kmeans centroid
            U, nbU = np.unique(IDX, return_counts=True);

            #print(nbU)
            if min(nbU) > min_spectra: #Do not keep this bootstrap if too few pixels fall in a category
                try:
                    allS = np.vstack((allS, C));
                except ValueError:
                    allS = np.copy(C)
        
        size0 = allS.shape[0]
        size1 = allS.shape[1]
        allS = np.reshape(allS, (1, size0, size1) ) # NFindr travaille avec des jeux de données 3D...
        s = pysptools.eea.NFINDR.extract(cls, allS, nb_c)
                    
        return s
    
################################################################################################################################

class AtgpInit: # Automatic Target Generation Process
    @classmethod
    def initialisation(cls, x, nb_c):
        
        s = pysptools.eea.eea.ATGP(x, nb_c)
        
        return s[0]
    
################################################################################################################################
    
class FippiInit: # Fast Iterative Pixel Purity Index
    @classmethod
    def initialisation(cls, x, nb_c):
        
        t = pysptools.eea.eea.FIPPI(x, q = nb_c)
        
        s = t[0]
        s = s[:nb_c, :]
        
        return s
  
################################################################################################################################
    
class PpiInit: # Pixel Purity Index
    @classmethod
    def initialisation(cls, x, nb_c):
        numSkewers = 10000
        s = pysptools.eea.eea.PPI(x, nb_c, numSkewers)
        
        return s[0]   
 
    
################################################################################################################################
class nKmeansInit:    
    @classmethod
    def initialisation(cls, x, nb_c, n = 15): 
        
        """Sometimes it's necessary to run Kmeans for more component than we want, to get the expected spectras, this version runs
        the initialisation for nb_c + n components, and keep the nb_c centroids containing the most pixels"""

        
        nb_ci = nb_c + n 
        
        init = KMeans(nb_ci).fit(x)
        s = init.cluster_centers_
        lab = init.labels_
        
        U, nbU = np.unique(lab, return_counts=True);# nbU is the number of pixels in each centroid
        
        ind = nbU.argsort()[-nb_c:] # Get the indices of the nb_c centroids containing the most pixels
        
        s = s[ind,:] # Keep only the nb_c spectra
        
        
        return s


#########################################################################################################################################################


class half_mcrllm:

       
    def __init__(self,xraw, nb_c , init):
        
        # We don't use the number of iterations, it is just to accomodate the GUI
        
        self.Xraw = xraw
        self.nb_c = nb_c
        
    
        self.X ,self.Xsum,self.deletedpix, self.deletedLevels, self.check_pix , self.check_level = preprocess_data(self.Xraw)
        
        if self.check_pix:
            self.Xraw = np.delete(self.Xraw , self.deletedpix , axis = 0)
        
        if self.check_level:
            self.Xraw = np.delete(self.Xraw , self.deletedLevels , axis = 1)
        
        
        self.define_initial_spectra(init)
        
        c_pred = self.X @ self.S.T @ np.linalg.inv(self.S @ self.S.T)
        
        c = self.C_plm(self.S, self.Xraw, nb_c, c_pred)
        
        self.C = c
        
        self.C,self.S = final_process(self.C ,self.S, self.deletedpix , self.deletedLevels , self.check_pix , self.check_level )
    
    
    
    
    
    def C_plm(self, s, xraw, nb_c, c_pred):
        #initialize C

        [nb_pix,nb_lev] = np.shape(xraw)
        c_new = np.zeros((nb_pix,nb_c))
        


        # on calcule les concentrations optimales pour chaque pixel par maximum likelihood 
        for pix in range(nb_pix):

                x_sum = np.sum(xraw[pix,:])      #total des counts 
                sraw = s*x_sum
                
                c_new[pix,:] = self.pyPLM(nb_c, sraw, xraw[pix,:], c_pred[pix,:])
                
                
         # avoid errors (this part should not be necessary)
        c_new[np.isnan(c_new)] = 1/nb_c
        c_new[np.isinf(c_new)] = 1/nb_c
        c_new[c_new<0] = 0
        c_sum1 = np.array([np.sum(c_new,axis=1)])
        c_sum =c_sum1.T@np.ones((1,np.size(c_new,axis =1)))
        c_new = c_new/c_sum

        return c_new
    
    
    
    def pyPLM(self, nb_c, sraw, xrawPix, c_old):
        

        # sum of every value is equal to 1
        def con_one(c_old):
            return 1-sum(c_old) 
        

        # all values are positive
        bnds = ((0.0, 1.0),) * nb_c
        

        cons = [{'type': 'eq', 'fun': con_one}]
        
        
        
        
        def regressLLPoisson(sraw,  xrawPix, c_pred):
            
            
            
            #compute prediction of counts
            yPred = c_pred @ sraw
            
            # avoid errors, should not be necessary
            yPred[yPred < 1/(10000*len(yPred))] = 1/(10000*len(yPred))
            
            
            
            
            logLik = -np.sum(xrawPix*np.log(yPred)-yPred)
            
            
            return (logLik)
        
        
        
        def jacobians(nb_c, xrawPix, sraw, c_pred):

            #compute prediction of counts
            yPred = c_pred @ sraw
            
            
            #compute jacobians
            jacC = np.zeros(nb_c)
            
            for phase in range(nb_c):
                
                jacC[phase] = -np.sum(((xrawPix*sraw[phase,:])/yPred)-sraw[phase,:])
                
            return(jacC) 
        

   
                
        # Run the minimizer    
        results = minimize(partial(regressLLPoisson, sraw,  xrawPix), c_old, method='SLSQP', bounds=bnds, constraints=cons, jac = partial(jacobians, nb_c, xrawPix, sraw))
        results = results.x
        results = np.asarray(results)
        


        c_new = results.reshape(int(len(results) / nb_c), nb_c)
        
        
        return c_new
    
    
    

    def define_initial_spectra(self,init):
        
        if type(init) == type(''):
            
            if init == 'Kmeans':
                print('Initializing with {}'.format(init))
                self.Sini = KmeansInit.initialisation(self.X,self.nb_c)
                self.S = self.Sini.copy()
                
            elif init == 'MBKmeans':
                print('Initializing with {}'.format(init))
                self.Sini = MBKmeansInit.initialisation(self.X,self.nb_c)
                self.S = self.Sini.copy()
            
            
            elif init == 'NFindr':
                print('Initializing with {}'.format(init))
                self.Sini = NFindrInit.initialisation(self.X,self.nb_c)
                self.S = self.Sini.copy()
            
            elif init == 'RobustNFindr':
                print('Initializing with {}'.format(init))
                self.Sini = RobustNFindrInit.initialisation(self.X,self.nb_c)
                self.S = self.Sini.copy()
                
            elif init == 'ATGP':
                print('Initializing with {}'.format(init))
                self.Sini = AtgpInit.initialisation(self.X,self.nb_c)
                self.S = self.Sini.copy()
               
            elif init == 'FIPPI':
                print('Initializing with {}'.format(init))
                self.Sini = FippiInit.initialisation(self.X,self.nb_c)
                self.S = self.Sini.copy()
                
            elif init == 'nKmeansInit':
                print('Initializing with {}'.format(init))
                self.Sini = nKmeansInit.initialisation(self.X,self.nb_c)
                self.S = self.Sini.copy()
        
        elif type(init) == type(np.array([1])):
            print('Initizaling with given spectra')
            self.S = init
            
        else:
            raise('Initialization method not found')
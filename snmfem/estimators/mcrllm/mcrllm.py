

import numpy as np
from scipy.optimize import minimize
from functools import partial
import matplotlib.pyplot as plt


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



#################################################################################################################################    


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

    
##############################################################################################################################################    


class mcrllm:
    
    
    def __init__(self,Xraw,nb_c,init,nb_iter=1,method="variable",fact_ini=.5):
        
        # Save Xraw and normalize data
        self.Xraw = Xraw
        self.X ,self.Xsum,self.deletedpix, self.deletedLevels, self.check_pix , self.check_level = preprocess_data(self.Xraw)
        
        if self.check_pix:
            self.Xraw = np.delete(self.Xraw , self.deletedpix , axis = 0)
        
        if self.check_level:
            self.Xraw = np.delete(self.Xraw , self.deletedLevels , axis = 1)
        
        
        self.pix,self.var = np.shape(self.X)
        self.nb_c = nb_c
        self.method = method 
        self.expvar = np.inf
        self.fact_ini = fact_ini
        
        # History initialization
        self.allC = []
        self.allS = []
        self.allphi = []
        
        
        self.C = np.ones([self.pix,self.nb_c])/nb_c
        

            
        # Initialization
        self.define_initial_spectra(init)
        
        
        
        for iteration in range(nb_iter):
            print("Iteration {:.0f}".format(len(self.allS)+1))

            
            self.C_plm()
            self.S_plm()
            
            
        self.C,self.S = final_process(self.C ,self.S, self.deletedpix , self.deletedLevels , self.check_pix , self.check_level )
        
        
     
    
    def C_plm(self):
        
        c_new = np.zeros((self.pix,self.nb_c))
        

        # on calcule les concentrations optimales pour chaque pixel par maximum likelihood 
        for pix in range(self.pix):
            sraw = self.S*self.Xsum[pix]
            c_new[pix,:] = self.pyPLM(sraw, self.Xraw[pix,:], self.C[pix,:])
                
                
        # avoid errors (this part should not be necessary)
        c_new[np.isnan(c_new)] = 1/self.nb_c
        c_new[np.isinf(c_new)] = 1/self.nb_c
        c_new[c_new<0] = 0
        c_sum1 = np.array([np.sum(c_new,axis=1)]).T
        c_new = c_new/c_sum1

        self.C = c_new.copy()
        self.allC.append( c_new.copy() )
    
            
        
        
    
    def Sphi(self,phi,h):
    
        C_m = self.C**phi
            
        S = np.linalg.inv(C_m[h,:].T@C_m[h,:])@C_m[h,:].T@self.X[h,:]
        S[S<1e-15] = 1e-15
        S = S/np.array([np.sum(S,axis=1)]).T
        
        return S
    
    
    
    
    def S_plm(self):
        
        
        h = np.random.permutation(len(self.X))
        phi_optimal = 1
        
        if self.method == "variable":
            allMSE = []
            all_phis = np.arange(.1,10.1,.1)
            
            for phi in all_phis:
                S = self.Sphi(phi,h)
                allMSE.append(np.sum( (S-self.S)**2 ))
                
            phi_optimal = all_phis[np.argmin(allMSE)]
            self.S = self.Sphi(phi_optimal,h)
            
            
                    
        else: # Standard
            
            self.S =  self.Sphi(phi_optimal,h)
            
            
        self.allS.append( self.S.copy() )
        self.allphi.append(phi_optimal)
        
        
        
    
    
    def pyPLM(self, sraw, xrawPix, c_old):
        

        # sum of every value is equal to 1
        def con_one(c_old):
            return 1-sum(c_old) 
        
        
        
        def regressLLPoisson(sraw,  xrawPix, c_pred):
            
            #compute prediction of counts
            yPred = c_pred @ sraw
            nb_lev = len(yPred) #
            # avoid errors, should (may?) not be necessary
            yPred[yPred < 1/10000] = 1/10000
            logLik = -np.sum(xrawPix*np.log(yPred)-yPred)
            return (logLik)
        
        
        
        def jacobians(nb_c, xrawPix, sraw, c_pred):

            #compute prediction of counts
            yPred = c_pred @ sraw
            # avoid errors, should (may?) not be necessary
            yPred[yPred < 1/10000] = 1/10000
            
            #compute jacobians
            jacC = np.zeros(nb_c)
            
            for phase in range(nb_c): 
                jacC[phase] = -np.sum(((xrawPix*sraw[phase,:])/yPred)-sraw[phase,:])    
            return(jacC) 
        
        
        
        # all values are positive
        bnds = ((0.0, 1.0),) * self.nb_c
        cons = [{'type': 'eq', 'fun': con_one}]
   
                
        # Run the minimizer    
        results = minimize(partial(regressLLPoisson, sraw,  xrawPix), c_old,\
                           method='SLSQP', bounds=bnds, constraints=cons, \
                           jac = partial(jacobians, self.nb_c, xrawPix, sraw))
        results = np.asarray(results.x)
        

        c_new = results.reshape(int(len(results) / self.nb_c), self.nb_c)
        
        
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
                
            elif init == 'nKmeans':
                print('Initializing with {}'.format(init))
                self.Sini = nKmeansInit.initialisation(self.X,self.nb_c)
                self.S = self.Sini.copy()
        
        elif type(init) == type(np.array([1])):
            print('Initializing with given spectra')
            self.S = init
            
        else:
            raise('Initialization method not found')
        
        
    
    

        
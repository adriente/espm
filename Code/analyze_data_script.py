######################################
# Small script for cluster execution #
######################################

import numpy as np
from snmf import SNMF
import utils_V2 as u2
import hyperspy.api as hs
import matplotlib.pyplot as plt

################
# Data loading #
################

filename="Data/aspim036_N150_2ptcls_brstlg"

S=hs.load(filename+".hspy")
X=S.data
# Collection of spectra containing only phase 0
X_part=S.inav[60:,:].data

#####################
# Utility functions #
#####################

# This function will find the best matching endmember for each true spectrum. This is useful since the A and P matrice are initialized at random. 

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
            list_angles.append(u2.MetricsUtils.spectral_angle(list_true_vectors[i],list_algo_vectors[j]))
        ind_min=np.argmin(np.array(list_angles))
        list_algo_vectors[ind_min]=1e28*np.ones(size)
        ordered_angles.append(u2.MetricsUtils.spectral_angle(list_true_vectors[i],copy_algo_vectors[ind_min]))
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
            list_maps.append(u2.MetricsUtils.MSE_map(list_true_maps[i],list_algo_maps[j]))
        ind_min=np.argmin(np.array(list_maps))
        list_algo_maps[ind_min]=1e28*np.ones(size)
        ordered_maps.append(u2.MetricsUtils.MSE_map(list_true_maps[i],copy_algo_maps[ind_min]))
    return ordered_maps

##############
# Parameters #
##############

# True bremsstrahlung parameters
c0  =  3.943136127751902
c1  =  3.9446849862408535 
c2  =  0.027663073842682524
b0  =  0.1414560446115408
b1  =  -0.1057210517202927
b2  =  0.026461615841445782

# SNMF parameters
brstlg_pars = [b1,b2,c0,c1,c2]
tol = 1e-4
max_iter = 50000
b_tol = 0.1
mu_sparse = 0.0
eps_sparse = 1.0
phases = 3

# Loading of ground truth
true_spectra=[]
true_maps=[]
true_spectra.append(np.genfromtxt(filename+"spectrum_p0"))
true_spectra.append(np.genfromtxt(filename+"spectrum_p1"))
true_spectra.append(np.genfromtxt(filename+"spectrum_p2"))
true_maps.append(np.load(filename+"map_p0.npy"))
true_maps.append(np.load(filename+"map_p1.npy"))
true_maps.append(np.load(filename+"map_p2.npy"))

# If required the b_matr optimization can be bypassed using a brstlg input
x_scale = u2.Gaussians().x
brstlg = u2.Distributions.simplified_brstlg_2(x_scale,b0,b1,b2,c0,c1,c2)

# If mu_sparse !=0 a good initialization of the first phase is required, it can be done using the spectrum below
init_matrix=np.average(X_part,axis=(0,1))

########
# SNMF #
########

# Creation of an SNMF object with the parameters above
mdl = SNMF(max_iter = max_iter, tol = tol, b_tol = b_tol, mu_sparse=mu_sparse, eps_sparse = eps_sparse, num_phases=phases, bremsstrahlung=None, brstlg_pars = brstlg_pars, init_spectrum = None)
mdl.fit(X)

################
# Save results #
################

# Returns the angles between the ground truth and the endmembers found using SNMF
angles=find_min_angle(true_spectra,[mdl.get_phase_spectrum(0),mdl.get_phase_spectrum(1),mdl.get_phase_spectrum(2)])

maps=find_min_MSE(true_maps,[mdl.get_phase_map(0),mdl.get_phase_map(1),mdl.get_phase_map(2)])

print("mu = {}, epsilon = {}".format(mu_sparse,eps_sparse))
print("Angle phase 0 :",angles[0])
print("Angle phase 1 :",angles[1])
print("Angle phase 2 :",angles[2])
print("MSE phase 0 :",maps[0])
print("MSE phase 1 :",maps[1])
print("MSE phase 2 :",maps[2])
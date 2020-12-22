######################################
# Small script for cluster execution #
######################################

import numpy as np
from snmf import SNMF
import hyperspy.api as hs
import json
import sys
import pprint
import EDXS_model
import h5py
import utils as u

####################
# system arguments #
####################

json_path = sys.argv[1]
mu_sparse = float(sys.argv[2])
eps_sparse = float(sys.argv[3])

####################
# config json load #
####################

with open(json_path,"r") as f :
    json_dict = json.load(f)

print("mu = {}, epsilon = {}".format(mu_sparse,eps_sparse))
pprint.pprint(json_dict)

################
# Data loading #
################

S=hs.load(json_dict["paths"]["data"]+".hspy")
X=S.data

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
            list_angles.append(u.Functions.spectral_angle(list_true_vectors[i],list_algo_vectors[j]))
        ind_min=np.argmin(np.array(list_angles))
        list_algo_vectors[ind_min]=1e28*np.ones(size)
        ordered_angles.append(u.Functions.spectral_angle(list_true_vectors[i],copy_algo_vectors[ind_min]))
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
            list_maps.append(u.Functions.MSE_map(list_true_maps[i],list_algo_maps[j]))
        ind_min=np.argmin(np.array(list_maps))
        list_algo_maps[ind_min]=1e28*np.ones(size)
        ordered_maps.append(u.Functions.MSE_map(list_true_maps[i],copy_algo_maps[ind_min]))
    return ordered_maps

##############
# Parameters #
##############

# SNMF parameters
brstlg_pars = json_dict["brstlg_pars"]
tol = json_dict["tol"]
max_iter = json_dict["max_iter"]
b_tol = json_dict["b_tol"]
phases = json_dict["phases"]
em = EDXS_model.EDXS_Model(json_dict["paths"]["xray_db"])
em.generate_g_matr(json_dict["elements_list"])

# Loading of ground truth
true_spectra=[]
true_maps=[]
true_spectra.append(np.genfromtxt(json_dict["paths"]["data"]+"spectrum_p0"))
true_spectra.append(np.genfromtxt(json_dict["paths"]["data"]+"spectrum_p1"))
true_spectra.append(np.genfromtxt(json_dict["paths"]["data"]+"spectrum_p2"))
true_maps.append(np.load(json_dict["paths"]["data"]+"map_p0.npy"))
true_maps.append(np.load(json_dict["paths"]["data"]+"map_p1.npy"))
true_maps.append(np.load(json_dict["paths"]["data"]+"map_p2.npy"))

# If mu_sparse !=0 a good initialization of the first phase is required, it can be done using the spectrum below
init_matrix=np.loadtxt(json_dict["paths"]["init_spectrum"])

########
# SNMF #
########

# Creation of an SNMF object with the parameters above
mdl = SNMF(max_iter = max_iter, tol = tol, b_tol = b_tol, mu_sparse=mu_sparse, eps_sparse = eps_sparse, num_phases=phases,edxs_model=em, brstlg_pars = brstlg_pars, init_spectrum = None)
mdl.fit(X)

################
# Save results #
################

# Returns the angles between the ground truth and the endmembers found using SNMF
angles=find_min_angle(true_spectra,[mdl.get_phase_spectrum(0),mdl.get_phase_spectrum(1),mdl.get_phase_spectrum(2)])

maps=find_min_MSE(true_maps,[mdl.get_phase_map(0),mdl.get_phase_map(1),mdl.get_phase_map(2)])

print("Angle phase 0 :",angles[0])
print("Angle phase 1 :",angles[1])
print("Angle phase 2 :",angles[2])
print("MSE phase 0 :",maps[0])
print("MSE phase 1 :",maps[1])
print("MSE phase 2 :",maps[2])

fname = "mu{}_eps{}_out.hdf5".format(mu_sparse,eps_sparse)

with h5py.File(fname,"w") as file :
    file.create_dataset("a_matr",data= mdl.a_matr)
    file.create_dataset("p_matr",data = mdl.p_matr)
    file.create_dataset("b_matr",data = mdl.b_matr)
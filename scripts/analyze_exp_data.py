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

##############
# Parameters #
##############

# SNMF parameters
brstlg_pars = json_dict["brstlg_pars"]
tol = json_dict["tol"]
max_iter = json_dict["max_iter"]
b_tol = json_dict["b_tol"]
phases = json_dict["phases"]
em = EDXS_model.EDXS_Model(json_dict["paths"]["xray_db"],e_offset=S.axes_manager[2].offset, e_scale=S.axes_manager[2].scale, e_size= S.axes_manager[2].size)
em.generate_g_matr(json_dict["elements_list"])

# If mu_sparse !=0 a good initialization of the first phase is required, it can be done using the spectrum below
init_matrix=(hs.load(json_dict["paths"]["init_spectrum"])).data

########
# SNMF #
########

# Creation of an SNMF object with the parameters above
mdl = SNMF(max_iter = max_iter, tol = tol, b_tol = b_tol, mu_sparse=mu_sparse, eps_sparse = eps_sparse, num_phases=phases,edxs_model=em, brstlg_pars = brstlg_pars, init_spectrum = init_matrix)
mdl.fit(X,eval_print = False)

fname = json_dict["paths"]["data"] + "_mu{}_eps{}.hdf5".format(mu_sparse,eps_sparse)

with h5py.File(fname,"w") as file :
    file.create_dataset("a_matr",data= mdl.a_matr)
    file.create_dataset("p_matr",data = mdl.p_matr)
    file.create_dataset("b_matr",data = mdl.b_matr)
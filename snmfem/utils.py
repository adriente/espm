import numpy as np
from scipy.optimize import nnls

def rescaled_DA(D,A) : 
    _, p = A.shape
    o = np.ones((p,))
    s = np.linalg.lstsq(A.T, o, rcond=None)[0]
    if (s<=0).any():
        s = np.maximum(nnls(A.T, o)[0], 1e-10)
    D_rescale = D@np.diag(1/s)
    A_rescale = np.diag(s)@A
    return D_rescale, A_rescale
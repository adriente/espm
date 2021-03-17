import numpy as np

def rescaled_DA(D,A) : 
    k, p = A.shape
    o = np.ones((p,))
    s = np.linalg.lstsq(A.T, o, rcond=None)[0]
    D_rescale = D@np.diag(1/s)
    A_rescale = np.diag(s)@A
    return D_rescale, A_rescale
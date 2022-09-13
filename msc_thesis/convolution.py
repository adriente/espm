
from scipy import signal
import math
import numpy as np
import quadprog
import numpy

# Remove spatial margins when using a filter of the shape indicated (used to crop useful values of Y_vol and H_vol after convolution)
def remove_spatial_margins(vol, filter_shape):
    vol = vol[:, filter_shape[0]//2:-filter_shape[0]//2+1,  filter_shape[1]//2:-filter_shape[1]//2+1]
    return vol

# Convolve dataset with 2D noncentral weights filter 
def get_smoothed_dataset(Y_vol, noncentral_weights):

    # Assert correct dimensionality
    assert len(Y_vol.shape) == 3, f"input is not three dimensional, got: {len(Y_vol.shape)} dimensions"
    assert len(noncentral_weights.shape) == 2, f"filter is not two dimensional, got: {len(noncentral_weights.shape)} dimensions"

    # Assert filter fits in input
    assert Y_vol.shape[1] >= noncentral_weights.shape[0] and Y_vol.shape[2] >= noncentral_weights.shape[1], f"filter does not fit in input, filter shape: {noncentral_weights.shape}, input shape: {Y_vol.shape}"

    Y_n_vol = np.zeros(Y_vol.shape) # Initialize
    # Convolve every channel separately
    for c in range(Y_vol.shape[0]):
        Y_n_vol[c,:,:] = signal.convolve2d(Y_vol[c,:,:], noncentral_weights, boundary='symm', mode='same')
    return Y_n_vol

# Solve quadratic program (using quadprog library)
def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -numpy.vstack([A, G]).T
        qp_b = -numpy.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

# Build noncentral filter weights
def build_noncentral_filter_weights(Y_vol, N_rows, size_filter, isFake):

    # Assert correct dimensionality
    assert len(Y_vol.shape) == 3, f"input is not three dimensional, got: {len(Y_vol.shape)} dimensions"
    assert len(size_filter) == 3, f"filter is not three dimensional, got: {len(size_filter)} dimensions"

    # Assert non-negative filter size
    assert size_filter[0] > 0 and size_filter[1] > 0 and size_filter[2] > 0, f"filter shape contains non-positive dimensions, filter size: {size_filter}"

    # Assert filter fits in input
    assert Y_vol.shape[0] >= size_filter[0] and Y_vol.shape[1] >= size_filter[1] and Y_vol.shape[2] >= size_filter[2], f"filter does not fit in input, filter shape: {size_filter}, input shape: {Y_vol.shape}"

    # Filter size
    S0, S1, S2 = size_filter[0], size_filter[1], size_filter[2]

    # Get fake non-central filter weights if you prefer to avoid running the quadratic problem  
    if isFake:
        noncentral_weights = np.zeros(size_filter)

        # Construct fake non-central filter weights
        for i in range(size_filter[0]):
            for j in range(size_filter[1]):
                for k in range(size_filter[2]):
                    dist = np.sqrt((i-S0//2)**2 + (j-S1//2)**2 + (k-S2//2)**2)
                    noncentral_weights[i,j,k] = np.exp(-dist)
        
        noncentral_weights[S0//2,S1//2,S2//2] = 0              # Set central weight equal to 0
        noncentral_weights = noncentral_weights/np.sum(noncentral_weights)   # Normalize so that sum of weights equal to 1

        return noncentral_weights

    # Input dimensions
    L, P1, P2= Y_vol.shape[0], Y_vol.shape[1], Y_vol.shape[2]

    # ---- Note ----
    # N_rows is the number of colums we use to solve the quadratic problem, which is equivalent to the number of central-noncentral voxel value relationships.
    # Ideally we should have all central voxels be part of the quadratic problem but for large datasets it becomes intractable, so we pick a random subset of voxels instead of all of them
    
    # Pick N_rows random voxels inside volum for which complete set of neighbours exists
    rand_l = np.random.randint(S0//2,L-(S0//2)-1,N_rows)
    rand_p1 = np.random.randint(S1//2,P1-(S1//2)-1,N_rows)
    rand_p2 = np.random.randint(S2//2,P2-(S2//2)-1,N_rows)

    N = np.zeros((N_rows, S0*S1*S2), dtype = np.float32)
    y = np.zeros(N_rows, dtype = np.float32)

    # Build central voxel array (y) and neighbour matrix (N)
    for i, (l, p1, p2) in enumerate(zip(rand_l, rand_p1, rand_p2)):
        N[i,:] = Y_vol[l-S0//2:l+S0//2+1,p1-S1//2:p1+S1//2+1,p2-S2//2:p2+S2//2+1].flatten()
        y[i] = Y_vol[l,p1,p2]

    # Discard rows where central voxel is 0 (redundant)
    N = N[np.where(y>0),:]
    y = y[np.where(y>0)]
    N = N[0,:,:]

    # Solve the quadratic problem: minimize  1/2w^T P w + q^T w  s.t Cw <= d and Aw = b
    P_ = 2*N.T@N
    q_ = -2*N.T@y

    # INEQUALITY CONSTRAINTS
    # 1. All weights must be nonnegative
    C_geq0 = -np.eye((S0*S1*S2))                # Apply to all weights
    d_geq0 = np.zeros((S0*S1*S2,))-0.0000001    # Weights greater or equal than 0

    # 2. Central weight has to be equal to 0
    C_central_leq0 = np.zeros((1,S0*S1*S2))     # Initialize
    C_central_leq0[0,(S0*S1*S2)//2] = 1         # Apply only to central weight
    d_central_leq0 = np.zeros(1)+0.0000001      # Central weight lower or equal than 0

    # 3. Weights have to sum to 1
    C_sum_leq1 = np.ones((1,S0*S1*S2))          # Apply to sum of weights
    d_sum_leq1 = np.ones(1)*1.0+0.0000001       # Sum of weights lower or equal than 1
    C_sum_geq1 = -np.ones((1,S0*S1*S2))         # Apply to sum of weights
    d_sum_geq1 = -np.ones(1)*1.0-0.0000001      # Sum of weights greater or equal than 1

    # Stack constraints to build C and d
    C_ = np.vstack((C_geq0, C_central_leq0, C_sum_leq1, C_sum_geq1))
    d_ = np.hstack((d_geq0, d_central_leq0, d_sum_leq1, d_sum_geq1))

    # Solve the quadratic program
    noncentral_weights = quadprog_solve_qp(np.array(P_, dtype=np.double),np.array(q_, dtype=np.double), G=np.array(C_, dtype=np.double), h=np.array(d_, dtype=np.double))
    noncentral_weights = noncentral_weights.reshape(size_filter)   # Reshape to 3D

    return noncentral_weights
   
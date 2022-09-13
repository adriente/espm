from scipy import interpolate
import numpy as np
from esmpy.estimators import SmoothNMF
import time
from esmpy.measures import find_min_angle, ordered_r2
from msc_thesis.plotting import plot_results_report
from msc_thesis.binning import bin_dataset_3D, bin_dataset_2D

def runNMF(Dtilde, Htilde, lambda_, mu_, binning_sizes, alpha, num_iterations, upsampleMode, Y_vol, Y_n_vol, G):

    L, P1, P2, K, P = Y_vol.shape[0], Y_vol.shape[1], Y_vol.shape[2], Htilde.shape[0],  Y_vol.shape[1]*Y_vol.shape[2]
    updateW = True

    Y = Y_vol.reshape(L, P)
    H =None
    W = None

    if alpha >= 0:
        Y_vol =  Y_vol*(alpha)+ Y_n_vol*(1-alpha)


    for i in range(len(binning_sizes)):

        start_time = time.time()
        B_3D = binning_sizes[i]

        Y_hat_vol = bin_dataset_3D(Y_vol, B_3D)
        Y_hat = Y_hat_vol.reshape(L//B_3D[0], P//(B_3D[1]*B_3D[2]))

        if G is not None:
            G_hat = bin_dataset_2D(G, (B_3D[0],1))
        else:
            G_hat = None

        if (i > 0):
            H_vol = H.reshape((K, P1//binning_sizes[i-1][1], P2//binning_sizes[i-1][2]))
            H_vol = H_vol.repeat(binning_sizes[i-1][1]//binning_sizes[i][1], axis = 1).repeat(binning_sizes[i-1][2]//binning_sizes[i][2], axis = 2)
            H = H_vol.reshape((K, P//B_3D[1]//B_3D[2])) +0.01
            W =  W 
        
        est = SmoothNMF( n_components = K,tol=0.0001, max_iter = num_iterations, G = G_hat, lambda_L= lambda_, mu=mu_, force_simplex=False, normalize =True)
        
        if i > 0 and upsampleMode == 3:
            updateW = False
        D = est.fit_transform(Y_hat, W=W, H=H, update_W=updateW)

        H = est.H_
        W = est.W_


    H_vol = H.reshape(3, P1//B_3D[1], P2//B_3D[2])

    # Upsampling
    if upsampleMode == 0:
        H_vol = H_vol.repeat(B_3D[1], axis = 1).repeat(B_3D[2], axis = 2)
        H = H_vol.reshape(K, P)

    # Least squares regression
    if upsampleMode == 2:
        H = np.linalg.pinv(D)@Y

    # Interpolation
    if upsampleMode == 1:
        x1_lr = np.linspace(-1, 1, P1//B_3D[1])
        x2_lr = np.linspace(-1, 1, P2//B_3D[2])

        H_vol_interpolated = np.zeros((K, P1, P2))

        for i in range(H_vol.shape[0]):
            f = interpolate.interp2d(x1_lr, x2_lr, H_vol[i,:,:], kind='cubic')
            x1_hr = np.linspace(-1, 1,  P1)
            x2_hr = np.linspace(-1, 1,  P2)
            H_vol_interpolated[i,:,:] = f(x1_hr, x2_hr)

        H = H_vol_interpolated.reshape(K, P)
        
  

    # Upsample spectra
    D = D.repeat(B_3D[0], axis = 0)

    Yest = D@H
    Ydot = Dtilde@Htilde

    # Calculate Errors
    angle, true_inds = find_min_angle(Dtilde.T, D.T, unique=True, get_ind=True)
    r2 = ordered_r2(Htilde, H, true_inds)
    plot_results_report(Dtilde, D, Htilde, H, np.linspace(0,1,L), P1, P2)

    Yest[Yest<0] = 0
    kl =  np.sum(-Ydot*np.log(Yest+0.000000000001)+ Yest + Ydot*np.log(Ydot+0.000000000001) - Ydot)
    mse = np.mean((Ydot.flatten()-Yest.flatten())**2)
    return sum(angle)/len(angle), sum(r2)/len(r2), time.time()-start_time, mse, kl


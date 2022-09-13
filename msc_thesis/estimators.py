import numpy as np
from msc_thesis.binning import bin_dataset_3D
def get_mse_and_mprime_vs_alpha(Ytilde, Y_vol, noncentral_weights, alphas, K):

    N_channels_util = 200

    P1, P2, L, F1, F2 = Y_vol.shape[1], Y_vol.shape[2], Y_vol.shape[0], noncentral_weights.shape[0], noncentral_weights.shape[1]

    # Dimensions
    P1_clean = P1 - F1 + 1
    P2_clean = P2 - F2 + 1

    Ytilde_vol = Ytilde.reshape(L, P1, P2)
    noncentral_weights_flattened = noncentral_weights.flatten()

    # Neibour Matrix - Each row contains the neighbour pixels of the central pixel
    N_vol = np.zeros((N_channels_util, P1_clean, P2_clean, F1*F2))
    for px in range(0, P1_clean):
        for py in range(0, P2_clean):
            N_vol[:,px,py,:] =Y_vol[:N_channels_util,px:px+F1,py:py+F2].reshape(N_channels_util, F1*F2)
    N_matrix = N_vol.reshape(-1, F1*F2)
    b_ =  Y_vol[:N_channels_util,F1//2:-F1//2+1,F2//2:-F2//2+1].flatten()
    Y_central_est = b_.copy()                                                       # y_{i}
    Y_neighbor_est = np.sum(N_matrix*noncentral_weights_flattened, axis=1)          # \sum_{k \in \mathcal{K}} (w_k n_{i,k})
    Y_neighbor2_est =  np.sum(N_matrix*noncentral_weights_flattened**2, axis=1)     # \sum_{k \in \mathcal{K}} (w_k^2 n_{i,k})

    Ntilde_vol= np.zeros((N_channels_util, P1_clean, P2_clean, F1*F2))
    for px in range(0, P1_clean):
        for py in range(0, P2_clean):
            Ntilde_vol[:N_channels_util,px,py,:] =Ytilde_vol[:N_channels_util,px:px+F1,py:py+F2].reshape(N_channels_util, F1*F2)
    Ntilde_matrix = Ntilde_vol.reshape(-1,F1*F2)
    b_tilde = Ytilde_vol[:N_channels_util,F1//2:-F1//2+1,F2//2:-F2//2+1].flatten()
    Y_central = b_tilde.copy()                                                      # \tilde{y_{i}}
    Y_neighbor = np.sum(Ntilde_matrix*noncentral_weights_flattened, axis=1)         # \sum_{k \in \mathcal{K}} (w_k \tilde{n_{i,k}})
    Y_neighbor2 =  np.sum(Ntilde_matrix*noncentral_weights_flattened**2, axis=1)    # \sum_{k \in \mathcal{K}} (w_k^2 \tilde{n_{i,k}})

    vars_est, vars_true, biases_est, biases_true = [], [], [], []

    # For each alpha
    for alpha in alphas:

        # Estimator of variance (Lemma 4.3) - \widehat{Var} (\hat{y}_i) = \alpha ^2 y_{i}+ (1-\alpha)^2 \sum_{k \in \mathcal{K}} (w_k^2 n_{i,k})
        var_est = np.mean(Y_neighbor2_est*(1-alpha)**2 + Y_central_est*alpha**2)

        # Estimator of squared bias (Lemma 4.4) - \widehat{Bias^2}(\hat{y}_i) = (1-\alpha)^2\left((y_{n_i} - y_{i})^2 - \sum_{k\in \mathcal{K}} w_k^2y_{i,k} - y_{i} \right)
        bias_est = (1-alpha)**2*np.mean((Y_central_est-Y_neighbor_est)**2 - (Y_neighbor2_est+Y_central_est))

        # To calculate true variance, same equation as for estimator of variance but using ground truth values
        var_true = np.mean(Y_neighbor2*(1-alpha)**2 + Y_central_est*alpha**2)

        # To calculate true bias, simply compare with ground truth
        Y_hat_tilde = Y_central*alpha + Y_neighbor*(1-alpha)
        bias_true = np.mean((Y_hat_tilde-Y_central)**2)

        vars_est = np.append(vars_est, var_est)
        vars_true = np.append(vars_true, var_true)
        biases_est = np.append(biases_est, bias_est)
        biases_true = np.append(biases_true, bias_true)

    # MSE estimators
    mses_est = vars_est + biases_est
    mses_true = vars_true + biases_true

    # Mprime estimators
    mprimes_est = vars_est*K/L + biases_est
    mprimes_true = vars_true*K/L + biases_true

    
    return biases_est, biases_true, vars_est, vars_true, mses_est, mses_true, mprimes_est, mprimes_true


def get_mse_and_mprime_vs_binningfactor(Ytilde_vol, Y_vol, binning_sizes, K):

    L = Y_vol.shape[0]  # Number of energy channels

    vars_est, vars_true, biases_est, biases_true = [], [], [], []

    # For each alpha
    for i in range(len(binning_sizes)):
        BS = binning_sizes[i]
        B = BS[0]*BS[1]*BS[2]

        # Bin the measurement dataset and upsample to bring it back to its original dimensionality
        Y_vol_binned = bin_dataset_3D(Y_vol, BS).repeat(BS[1], axis=1).repeat(BS[2], axis=2)

        # Bin the ground truth and upsample to bring it back to its original dimensionality
        Ytilde_vol_binned = bin_dataset_3D(Ytilde_vol, BS).repeat(BS[1], axis=1).repeat(BS[2], axis=2)

        # Estimator of variance (Lemma 4.3) - \widehat{Var} (\hat{y}_i) = \alpha ^2 y_{i}+ (1-\alpha)^2 \sum_{k \in \mathcal{K}} (w_k^2 n_{i,k})
        vars_est = np.append(vars_est, np.mean(Y_vol_binned*1/B))
        
        # To calculate true variance, same equation as for estimator of variance but using ground truth values
        vars_true = np.append(vars_true, np.mean(Ytilde_vol*1/B))

        # Estimator of squared bias (Lemma 4.4) - \widehat{Bias^2}(\hat{y}_i) = (1-\alpha)^2\left((y_{n_i} - y_{i})^2 - \sum_{k\in \mathcal{K}} w_k^2y_{i,k} - y_{i} \right)
        biases_est = np.append(biases_est, np.mean((Y_vol-Y_vol_binned)**2 - 1/B*Y_vol_binned - (1-2/B)*Y_vol))
        
        # To calculate true bias, simply compare binned g.t with original g.t
        biases_true = np.append(biases_true, np.mean((Ytilde_vol- Ytilde_vol_binned)**2))

    # MSE estimators
    mses_est = vars_est + biases_est
    mses_true = vars_true + biases_true
    
    # Mprime estimators
    mprimes_est = vars_est*K/L + biases_est
    mprimes_true = vars_true*K/L + biases_true  

    return biases_est, biases_true, vars_est, vars_true, mses_est, mses_true, mprimes_est, mprimes_true

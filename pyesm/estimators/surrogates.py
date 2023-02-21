import numpy as np
from espm.conf import sigmaL
from espm.measures import trace_xtLx



def smooth_l2_surrogate(Ht, L, H=None, sigmaL=sigmaL, lambda_L=1):
    r"""Compute the smooth L2 surrogate of the Laplacian regularizer :math:`\lambda_L/2 tr(H \Delta H^\top)` at :math:`H^t`

    This function essentially computes:

    .. math::

       g(H,H^t ) = \frac{\lambda_L}{2} \left( tr\left( H^t \Delta H^{t\top} \right) + 2 tr \left(H^t \Delta  (H - H^t)\top \right) +  \sigma_L \| H - H^t \|_F^2 )

    where 
    
    * :math:`H^t` is the current estimate of :math:`\dot{H}`,
    * :math:`H` is the main variable of :math:`g`, 
    * :math:`\Delta` is the Laplacian matrix,
    * :math:`\lambda_L` is the regularization parameter for the Laplacian regularizer,and 
    * :math:`\sigma_L` is the maximum eigenvalue of the Laplacian :math:`\Delta`.

    Parameters
    ----------
    Ht : np.ndarray
        Current estimate of :math:`\dot{H}`.
    L : np.ndarray or scipy.sparse.csr_matrix
        Laplacian matrix.
    H : np.ndarray, optional
        Main variable of :math:`g`, by default None
    sigmaL : float, optional
        Maximum eigenvalue of the Laplacian :math:`\Delta`, by default sigmaL
    lambda_L : float, optional
        Regularization parameter for the Laplacian regularizer, by default 1
        

    Returns
    -------
    float
        Value of the smooth L2 surrogate at :math:`H`

    """
    HtTL = Ht @ L
    t1 = np.sum(HtTL * Ht)
    if H is None:
        t2 = t1
        t3 = 0
    else:
        t2 = np.sum(HtTL * H )
        t3 = np.sum((Ht-H)**2)
    return lambda_L / 2 * (2*t2 - t1 + sigmaL * t3)

# def smooth_l2_surrogate_alt(Ht, L, H=None, sigmaL=sigmaL, lambda_L=1):
#     HtTL = Ht @ L
#     t1 = np.sum(HtTL * Ht)
#     if H is None:
#         return lambda_L / 2 * t1
    
#     t2 = 2 * np.sum(HtTL * (H - Ht) )
#     t3 = sigmaL * np.sum((Ht-H)**2)
    
#     return lambda_L / 2 * (t1 + t2 + t3)

def smooth_dgkl_surrogate(Ht, L, H=None, sigmaL=sigmaL, lambda_L=1):
    r"""Compute the smooth KL surrogate of the Laplacian regularizer :math:`\lambda_L/2 tr(H \Delta H^\top)` at :math:`H^t`

    This function essentially computes:

    .. math::

         g(H,H^t ) = \frac{\lambda_L}{2} \left( tr\left( H^t \Delta H^{t\top} \right) + 2 tr \left(H^t \Delta  (H - H^t)\top \right) +  \sigma_L \sum_{i=1}^n \max_{j} \left( H_{ij} \log \frac{H_{ij}}{H^t_{ij}} - H_{ij} + H^t_{ij} \right) )

    where 
    
    * :math:`H^t` is the current estimate of :math:`\dot{H}`,
    * :math:`H` is the main variable of :math:`g`,
    * :math:`\Delta` is the Laplacian matrix,
    * :math:`\lambda_L` is the regularization parameter for the Laplacian regularizer, and
    * :math:`\sigma_L` is the maximum eigenvalue of the Laplacian :math:`\Delta`.

    Parameters
    ----------
    Ht : np.ndarray
        Current estimate of :math:`\dot{H}`.
    L : np.ndarray or scipy.sparse.csr_matrix
        Laplacian matrix.
    H : np.ndarray, optional
        Main variable of :math:`g`, by default None
    sigmaL : float, optional
        Maximum eigenvalue of the Laplacian :math:`\Delta`, by default sigmaL
    lambda_L : float, optional
        Regularization parameter for the Laplacian regularizer, by default 1

    Returns
    -------
    float
        Value of the smooth KL surrogate at :math:`H`

    """
    HtTL = Ht @ L
    t1 = np.sum(HtTL * Ht)
    
    def dgkl(p, q):
        return p * np.log(p/q) - p + q
    
    if H is None:
        t2 = t1
        t3 = 0
    else:
        t2 = np.sum(HtTL * H )
        maxH = np.max(H, axis=1)
        t3 = np.sum(maxH * np.sum(dgkl(Ht, H), axis=1))
    return lambda_L / 2 * (2*t2 - t1 + sigmaL * t3)

def diff_surrogate(Ht, H, L, sigmaL=sigmaL, lambda_L=1, algo="log_surrogate"):
    r"""Compute the difference between the surrogate and the true value of the Laplacian regularizer at :math:`H^t`.
    
    Parameters
    ----------
    Ht : np.ndarray
        Current estimate of :math:`\dot{H}`.
    H : np.ndarray
        Main variable of :math:`g`.
    L : np.ndarray or scipy.sparse.csr_matrix
        Laplacian matrix.
    sigmaL : float, optional
        Maximum eigenvalue of the Laplacian :math:`\Delta`, by default sigmaL
    lambda_L : float, optional
        Regularization parameter for the Laplacian regularizer, by default 1
    algo : str, optional
        Algorithm to use for the surrogate, by default "log_surrogate"
        * "log_surrogate" : use the smooth KL surrogate
        * "l2_surrogate" : use the smooth L2 surrogate

    Returns
    -------
    float
        Value of the difference between the surrogate and the true value of the Laplacian regularizer at :math:`H^t

    """
    b_inf = trace_xtLx(L, H.T) * lambda_L / 2
    if algo=="log_surrogate":
        b_supp = smooth_dgkl_surrogate(Ht, L=L, H=H, sigmaL=sigmaL, lambda_L=lambda_L)
    elif algo== "l2_surrogate":
        b_supp = smooth_l2_surrogate(Ht, L=L, H=H, sigmaL=sigmaL, lambda_L=lambda_L)
    else: 
        raise "Unknown algorithm"
    return b_supp - b_inf



def quadratic_surrogate(x, xt, f_xt, gradf_xt, sigma):
    r"""Compute the quadratic surrogate function of :math:`f` at :math:`x^t`

    This function essentially computes:

    .. math::
        
        g(x,x^t) = f(x^t) + \left< x - x^t , \nabla f (x^t) \right> + \sigma \| x - x^t \|_2^2 

    :param np.array x: variable
    :param np.array xt: variable
    :param np.array f_xt: function to be upper bounded
    :param np.array gradf_xt: function that compute the gradient
    :param float sigma: Lipschitz constant of the gradient

    :returns: the answer

    """
    return f_xt + np.sum((x-xt) * gradf_xt) + sigma * np.sum((x-xt)**2)


import numpy as np
from sklearn.decomposition._nmf import _initialize_nmf as initialize_nmf

from espm.conf import dicotomy_tol, log_shift, sigmaL
from espm.estimators.dicotomy import (
    dichotomy_simplex,
    dichotomy_simplex_acc,
    dichotomy_simplex_projected_gradient,
)
from espm.utils import get_array_module, to_numpy


def multiplicative_step_w(
    X,
    G,
    W,
    H,
    simplex_W=False,
    log_shift=log_shift,
    safe=True,
    l2=False,
    fixed_W=None,
    physics_model=None,
    use_bregman=False,
):
    """
    Multiplicative step in W.
    """
    if safe:
        # Allow for very small negative values!
        assert not (H < -log_shift / 2).any()
        assert not (W < -log_shift / 2).any()
        assert not (G < -log_shift / 2).any()

        H = np.maximum(H, log_shift)
        W = np.maximum(W, log_shift)

    if l2:
        GG = G.T @ G
        HH = H @ H.T
        GGWHH = (GG @ W) @ HH

        GXH = G.T @ (X @ H.T)

        new_W = W / GGWHH * GXH
    else:
        GW = G @ W
        GWH = GW @ H
        if use_bregman:
            # check if G is the identity matrix
            if G.shape[0] == G.shape[1] and np.array_equal(G, np.eye(G.shape[0])):
                sigmaR = X.sum(axis=1, keepdims=True)
            else:
                sigmaR = X.sum()
            num = sigmaR * W
            gradg = -G.T @ ((X / GWH) @ H.T) + np.outer(G.sum(axis=0), H.sum(axis=1))
            denum = gradg * W + sigmaR

        else:
            if X.nbytes > 1e8:
                xp = get_array_module(X)
                chunk_size = 256
                op1_H = xp.empty((X.shape[0], H.shape[0]), dtype=X.dtype)
                buf = xp.empty(
                    (min(chunk_size, X.shape[0]), H.shape[1]), dtype=X.dtype
                )
                for i in range(0, X.shape[0], chunk_size):
                    actual = min(chunk_size, X.shape[0] - i)
                    buf_c = buf[:actual]
                    np.matmul(GW[i : i + actual], H, out=buf_c)
                    np.divide(X[i : i + actual], buf_c, out=buf_c)
                    if np.isnan(buf_c).any():
                        np.matmul(GW[i : i + actual], H, out=buf_c)
                        np.maximum(buf_c, log_shift, out=buf_c)
                        np.divide(X[i : i + actual], buf_c, out=buf_c)
                    np.matmul(buf_c, H.T, out=op1_H[i : i + actual])
                num = W * (G.T @ op1_H)
            else:
                # Split to debug timing...
                # term1 = G.T @ (X / (GWH + eps)) @ H.T
                op1 = X / GWH
                if np.isnan(op1).any():
                    GWH = np.maximum(GWH, log_shift)
                    op1 = X / GWH

                num = W * (G.T @ (op1 @ H.T))
            denum = np.outer(G.sum(axis=0), H.sum(axis=1))
            if simplex_W:
                if physics_model is not None:
                    indices = physics_model.NMF_simplex()
                    nu = dichotomy_simplex(
                        num[indices, :],
                        denum[indices, :],
                        log_shift=log_shift,
                        tol=dicotomy_tol,
                        safe=safe,
                    )
                    denum[indices, :] = denum[indices, :] + nu
                else:
                    nu = dichotomy_simplex(
                        num, denum, log_shift=log_shift, tol=dicotomy_tol, safe=safe
                    )
                    denum = denum + nu

        new_W = num / denum

    new_W = np.maximum(new_W, log_shift)

    # TODO: exclude the fixed values in the update process. It is not straightforward
    if fixed_W is not None:
        xp = get_array_module(new_W)
        fixed_W = xp.asarray(fixed_W)
        new_W[fixed_W >= 0] = fixed_W[fixed_W >= 0]

    return new_W


def multiplicative_step_h(
    X,
    G,
    W,
    H,
    simplex_H=False,
    mu=0,
    log_shift=log_shift,
    epsilon_reg=1,
    safe=True,
    dicotomy_tol=dicotomy_tol,
    lambda_L=0,
    L=None,
    l2=False,
    sigmaL=sigmaL,
    fixed_H=None,
    use_bregman=False,
):
    """
    Multiplicative step in A.
    The main terms are calculated first.
    With mu_sparse = 0, the steps in A are calculated once. For mu_sparse != 0, the steps
    in A are calculated first with particle regularization. Then only the entries allowed
    by the mask are calculaed, without particle regularization. Note that mu can be passed
    as a vector to regularize the different phase of A differently.
    To calculate the regularized step, we make a linear approximation of the log.
    """
    if lambda_L != 0:
        if L is None:
            raise ValueError("Please provide the laplacian")
        HL = H @ L

    if safe:
        # Allow for very small negative values!
        # TODO: update this
        assert not (H < -log_shift / 2).any()
        assert not (W < -log_shift / 2).any()
        assert not (G < -log_shift / 2).any()
        H = np.maximum(H, log_shift)
        W = np.maximum(W, log_shift)

    GW = G @ W  # Also called D

    if l2:
        assert lambda_L == 0
        if np.isscalar(mu):
            assert mu == 0
        else:
            assert not (mu != 0).any()
        WGGW = GW.T @ GW
        WGX = GW.T @ X
        num = WGX
        denum = WGGW @ H
    else:
        if use_bregman:
            GWH = GW @ H
            sigmaR = X.sum(axis=0, keepdims=True)
            num = sigmaR / H
            gradg = -GW.T @ (X / GWH) + GW.sum(axis=0, keepdims=True).T
            denum = gradg + sigmaR / H
        else:
            if X.nbytes > 1e8:
                xp = get_array_module(X)
                chunk_size = 256
                num = xp.empty_like(H)
                buf = xp.empty(
                    (min(chunk_size, X.shape[0]), H.shape[1]), dtype=X.dtype
                )
                buf_k = xp.empty_like(H)
                for i in range(0, X.shape[0], chunk_size):
                    actual = min(chunk_size, X.shape[0] - i)
                    buf_c = buf[:actual]
                    GW_c = GW[i : i + actual]
                    np.matmul(GW_c, H, out=buf_c)
                    np.divide(X[i : i + actual], buf_c, out=buf_c)
                    if np.isnan(buf_c).any():
                        np.matmul(GW_c, H, out=buf_c)
                        np.maximum(buf_c, log_shift, out=buf_c)
                        np.divide(X[i : i + actual], buf_c, out=buf_c)
                    if i == 0:
                        np.matmul(GW_c.T, buf_c, out=num)
                    else:
                        np.matmul(GW_c.T, buf_c, out=buf_k)
                        num += buf_k
            else:
                GWH = GW @ H
                num = GW.T @ (X / GWH)
                if np.isnan(num).any():
                    GWH = np.maximum(GWH, log_shift)
                    num = GW.T @ (X / GWH)
            denum = GW.sum(axis=0, keepdims=True).T

        if not (np.isscalar(mu) and mu == 0):
            if len(np.shape(mu)) == 1:
                mu = np.expand_dims(mu, axis=1)
            denum = denum + mu / (H + epsilon_reg)
        if lambda_L != 0:
            maxH = H.max(axis=1, keepdims=True)
            num = num + lambda_L * sigmaL * maxH
            denum = denum + lambda_L * sigmaL * maxH + lambda_L * HL
    num = H * num
    if simplex_H:
        nu = dichotomy_simplex(
            num, denum, log_shift=log_shift, tol=dicotomy_tol, safe=safe
        )
    else:
        nu = 0
    if safe:
        assert not (denum < 0).any()
        assert not (num < 0).any()

    # Add the shift...
    new_H = np.maximum(num / (denum + nu), log_shift)

    if fixed_H is not None:
        xp = get_array_module(new_H)
        fixed_H = xp.asarray(fixed_H)
        new_H[fixed_H >= 0] = fixed_H[fixed_H >= 0]
    return new_H


def initialize_algorithms(
    X,
    G,
    W,
    H,
    n_components,
    init,
    random_state,
    simplex_H,
    simplex_W,
    logshift=log_shift,
    physics_model=None,
):
    # Handle initialization
    xp = get_array_module(X)

    if G is None:
        skip_second = True
        # G = sparse.diags(np.ones(X.shape[0]).astype(X.dtype))
        G = xp.eye(X.shape[0], dtype=X.dtype)

    # elif callable(G) :
    #     assert not(model_params is None), "You need to input model_parameters"
    #     assert not(g_params is None), "You need to input g_parameters"
    #     G = G(model_params,g_params)
    #     skip_second = False

    else:
        skip_second = False
        G = xp.asarray(G)

    if W is None:
        if H is None:
            X_cpu = to_numpy(X)
            D, H = initialize_nmf(
                X_cpu, n_components=n_components, init=init, random_state=random_state
            )
            D = xp.asarray(D)
            H = xp.asarray(H)
            # D, A = u.rescaled_DA(D,A)
            if simplex_H:
                H = np.nan_to_num(H, nan=1.0 / H.shape[0])
                scale = H.sum(axis=0, keepdims=True)
                H = H / scale

                # D = np.abs(np.linalg.lstsq(H.T, X.T,rcond=None)[0].T)
                D = D * scale.mean()
        else:
            H = xp.asarray(H)
            D = np.abs(np.linalg.lstsq(H.T, X.T, rcond=None)[0].T)
        if skip_second:
            W = D
        else:
            if physics_model is not None:
                # [np.where(G[:,:-2].sum(axis=1)<(np.max(G[:,:-2].sum(axis=1))*0.001))[0],:]
                # Divide in two parts the initial fitting, otherwise the bremsstrahlung (which has a low intensity) tends to be poorly learned
                # First fit the caracteristic Xrays, then subtract that contribution to obtain a rough estimate of the bremsstralung parameters
                W = xp.asarray(physics_model.NMF_initialize_W(D))
                # Wbrem = (np.linalg.lstsq(G[:,-2:],D - G[:,:-2]@Wcarac,rcond = None)[0]).clip(min = 0)
                if simplex_W:
                    indices = physics_model.NMF_simplex()
                    W = np.nan_to_num(W, nan=1.0 / W.shape[0])
                    scale = W[indices, :].sum(axis=0, keepdims=True)
                    W[indices, :] = W[indices, :] / scale
            # P = np.abs(np.linalg.lstsq(G, D,rcond=None)[0])
            else:
                W = np.abs(np.linalg.lstsq(G, D, rcond=None)[0])

                if simplex_W:
                    W = np.nan_to_num(W, nan=1.0 / W.shape[0])
                    scale = W.sum(axis=0, keepdims=True)
                    W = W / scale

    elif H is None:
        W = xp.asarray(W)
        D = G @ W
        H = np.abs(np.linalg.lstsq(D, X, rcond=None)[0])
        if simplex_H:
            scale = H.sum(axis=0, keepdims=True)
            H = H / scale
    else:
        W = xp.asarray(W)
        H = xp.asarray(H)

    W = np.maximum(W, log_shift)
    H = np.maximum(H, log_shift)

    return G, W, H


def update_q(D, H, log_shift=log_shift):
    """Perform a Q step."""
    return H.T[np.newaxis, :, :] * (
        D[:, np.newaxis, :] / ((D @ H + log_shift)[:, :, np.newaxis])
    )


def multiplicative_step_wq(
    X, G, W, H, simplex_W=True, log_shift=log_shift, safe=True, physics_model=None
):
    """
    Multiplicative step in W using the WQ technique.

    This function does exactly the same as `multiplicative_step_w` and is probably slower.
    """

    if safe:
        # Allow for very small negative values!
        assert not np.any(H < -log_shift / 2)
        assert not np.any(W < -log_shift / 2)
        assert not np.any(G < -log_shift / 2)

    GW = G @ W
    XQ = GW * ((X / (GW @ H + log_shift)) @ H.T)

    term1 = G.T @ (XQ / (GW + log_shift))

    term2 = np.outer(np.sum(G, axis=0), np.sum(H, axis=1))
    if simplex_W:
        if physics_model is not None:
            indices = physics_model.NMF_simplex()
            nu = dichotomy_simplex(
                term1[indices, :],
                term2[indices, :],
                log_shift=log_shift,
                tol=dicotomy_tol,
                safe=safe,
            )
            term2[indices, :] = term2[indices, :] + nu
        else:
            nu = dichotomy_simplex(
                term1, term2, log_shift=log_shift, tol=dicotomy_tol, safe=safe
            )
            term2 = term2 + nu
    return W / term2 * term1


def multiplicative_step_hq(
    X,
    G,
    W,
    H,
    simplex_H=True,
    log_shift=log_shift,
    safe=True,
    dicotomy_tol=dicotomy_tol,
    lambda_L=0,
    L=None,
    sigmaL=sigmaL,
    fixed_H=None,
):
    """
    Multiplicative step in H.
    """
    if lambda_L != 0 and L is None:
        raise ValueError("Please provide the laplacian")

    if safe:
        # Allow for very small negative values!
        assert not (H < -log_shift / 2).any()
        assert not (W < -log_shift / 2).any()
        assert not (G < -log_shift / 2).any()

    GW = G @ W  # Also called D
    GWH = GW @ H

    minus_c = H * (GW.T @ (X / (GWH + log_shift)))

    b = GW.sum(axis=0, keepdims=True).T
    if lambda_L != 0:
        b = b + lambda_L * H @ L - lambda_L * sigmaL * H
        a = lambda_L * sigmaL
        if simplex_H:
            nu = dichotomy_simplex_acc(
                a, b, minus_c, log_shift=log_shift, tol=dicotomy_tol, safe=safe
            )
            b = b + nu
        new_H = (-b + np.sqrt(b**2 + 4 * a * minus_c)) / (2 * a)
    else:  # We recover the classic case: multiplicative_step_a
        if simplex_H:
            nu = dichotomy_simplex(
                minus_c, b, log_shift=log_shift, tol=dicotomy_tol, safe=safe
            )
            b = b + nu
        new_H = minus_c / b

    # Add the shift...
    new_H = np.maximum(new_H, log_shift)

    if fixed_H is not None:
        xp = get_array_module(new_H)
        fixed_H = xp.asarray(fixed_H)
        new_H[fixed_H >= 0] = fixed_H[fixed_H >= 0]
    return new_H


def gradW(X, G, W, H, log_shift=log_shift, safe=False, l2=False):
    if safe:
        H = np.maximum(H, log_shift)
        W = np.maximum(W, log_shift)
    if l2:
        grad = 2 * G.T @ ((G @ W) @ H - X) @ H.T
    else:
        D = G @ W
        DH = D @ H
        grad = G.T @ (-(X / DH) @ H.T + H.sum(axis=1, keepdims=True).T)
    return grad


def gradH(
    X,
    G,
    W,
    H,
    mu=0,
    lambda_L=0,
    L=None,
    epsilon_reg=1,
    log_shift=log_shift,
    safe=False,
    l2=False,
):
    if lambda_L != 0:
        if L is None:
            raise ValueError("Please provide the laplacian")

    if safe:
        H = np.maximum(H, log_shift)
        W = np.maximum(W, log_shift)

    D = G @ W
    if l2:
        grad = D.T @ (D @ H - X)
    else:
        DH = D @ H
        grad = -D.T @ (X / DH) + D.sum(axis=0, keepdims=True).T

    if not (np.isscalar(mu) and mu == 0):
        if len(np.shape(mu)) == 1:
            mu = np.expand_dims(mu, axis=1)
        grad = grad + mu / (H + epsilon_reg)

    if lambda_L != 0:
        grad = grad + (lambda_L * L @ H.T).T

    return grad


def proj_grad_step_w(
    X,
    G,
    W,
    H,
    gamma,
    simplex_W=True,
    log_shift=log_shift,
    safe=True,
    l2=False,
    fixed_W=None,
):
    """Projected gradient step for the variable W."""

    if safe:
        H = np.maximum(H, log_shift)
        W = np.maximum(W, log_shift)

    grad = gradW(X, G, W, H, log_shift=log_shift, safe=safe, l2=l2)

    # gradient step
    new_W = W - 1 / gamma * grad
    # projection
    new_W = np.maximum(new_W, log_shift)

    if fixed_W is not None:
        new_W[fixed_W >= 0] = fixed_W[fixed_W >= 0]

    if simplex_W:
        raise NotImplementedError(
            "Simplex constraint not implemented for W using the projected gradient method"
        )
    return new_W


def proj_grad_step_h(
    X,
    G,
    W,
    H,
    gamma,
    simplex_H=True,
    mu=0,
    log_shift=log_shift,
    epsilon_reg=1,
    safe=True,
    dicotomy_tol=dicotomy_tol,
    lambda_L=0,
    L=None,
    l2=False,
    fixed_H=None,
):
    """Projected gradient step for the variable H."""

    if safe:
        H = np.maximum(H, log_shift)
        W = np.maximum(W, log_shift)

    # gradient step
    grad = gradH(
        X,
        G,
        W,
        H,
        log_shift=log_shift,
        safe=safe,
        mu=mu,
        epsilon_reg=epsilon_reg,
        lambda_L=lambda_L,
        L=L,
        l2=l2,
    )
    new_H = H - 1 / gamma * grad

    # Dichotomy
    if simplex_H:
        nu = dichotomy_simplex_projected_gradient(
            new_H, log_shift=log_shift, tol=dicotomy_tol, safe=safe
        )
    else:
        nu = 0

    # projection
    new_H = np.maximum(new_H + nu, log_shift)

    if fixed_H is not None:
        new_H[fixed_H >= 0] = fixed_H[fixed_H >= 0]
    return new_H


def estimate_Lipschitz_bound_w(log_shift, X, G, k):
    xp = get_array_module(X)
    if G is None:
        G = xp.eye(X.shape[0])
    else:
        G = xp.asarray(G)
    Wlim = xp.ones([G.shape[1], k]) * log_shift
    Hlim = xp.ones([k, X.shape[1]]) * log_shift
    D = G @ Wlim
    DH = D @ Hlim
    gamma = ((Hlim.sum(axis=0, keepdims=True) * X / (DH**2)) @ Hlim.T).max()
    return float(gamma)


def estimate_Lipschitz_bound_h(log_shift, X, G, k, lambda_L=0, mu=0, epsilon_reg=1):
    xp = get_array_module(X)
    if G is None:
        G = xp.eye(X.shape[0])
    else:
        G = xp.asarray(G)
    Wlim = xp.ones([G.shape[1], k]) * log_shift
    Hlim = xp.ones([k, X.shape[1]]) * log_shift
    D = G @ Wlim
    DH = D @ Hlim

    gamma = (
        (D.T @ (D.sum(axis=1, keepdims=True) * X / (DH**2))).max()
        + 2 * lambda_L
        + mu * epsilon_reg
    )
    return float(gamma)

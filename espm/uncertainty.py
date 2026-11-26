"""Analytical uncertainty quantification for ESpM-NMF decompositions.

This module implements exact constrained Poisson Cramér–Rao bounds (CRB) and
sandwich covariance estimators for regularized non-negative matrix factorization
of the form :math:`Y = G W H`, exploiting the pixel-wise block-diagonal structure
of the Fisher information matrix to achieve :math:`\\mathcal{O}(p K^3 + m^2 K^2 p)`
complexity over :math:`p` pixels.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from espm.cupy import to_numpy


@dataclass
class ROIUncertaintyResult:
    """Uncertainty summary for a spatially averaged ROI or full spectrum image."""

    composition: np.ndarray
    std_composition: np.ndarray
    cov_composition: np.ndarray
    intensity: np.ndarray
    std_intensity: np.ndarray
    cov_intensity: np.ndarray
    indices: np.ndarray
    n_pixels: int


@dataclass
class ESPMUncertaintyResult:
    """Container for fitted ESpM-NMF parameter and quantification uncertainties.

    Attributes
    ----------
    G : np.ndarray of shape (n, m)
        Known physical spectral dictionary matrix.
    W : np.ndarray of shape (m, K)
        Estimated phase composition / spectral loading coefficients.
    H : np.ndarray of shape (K, p)
        Estimated pixel-wise phase abundances.
    cov_W : np.ndarray of shape (K, m, K, m)
        Constrained covariance tensor of ``W``, indexed as ``(k1, l1, k2, l2)``.
    std_W : np.ndarray of shape (m, K)
        Marginal 1-sigma uncertainties of ``W`` entries.
    std_H : np.ndarray of shape (K, p)
        Marginal 1-sigma uncertainties of ``H`` entries.
    std_M : np.ndarray of shape (m, p)
        Marginal 1-sigma uncertainties of unnormalized elemental maps ``M = W @ H``.
    gauge_null_dim : int
        Number of near-zero eigenvalues of the constrained Schur-complement Fisher
        information matrix on the tangent space of ``W`` (positive under continuous
        ``W @ H = (W A)(A^{-1} H)`` gauge ambiguity).
    is_identifiable : bool
        ``True`` if ``gauge_null_dim == 0`` on the constrained tangent space.
    method : str
        ``"crb"`` for constrained Poisson CRB or ``"sandwich"`` for regularized M-estimator
        sandwich covariance.
    """

    G: np.ndarray
    W: np.ndarray
    H: np.ndarray
    cov_W: np.ndarray
    std_W: np.ndarray
    std_H: np.ndarray
    std_M: np.ndarray
    gauge_null_dim: int
    is_identifiable: bool
    method: str
    simplex_W: bool
    simplex_H: bool
    simplex_indices: np.ndarray | None
    rate_floor: float
    _state: dict = field(default_factory=dict, repr=False)

    def component_composition(
        self,
        indices: Iterable[int] | None = None,
        scale: float = 100.0,
        return_cov: bool = False,
    ):
        """Compute normalized elemental compositions of each component and their uncertainties.

        Propagates ``cov_W`` via the exact multivariate delta method onto
        :math:`C_{l, k} = \\text{scale} \\cdot W_{l, k} / \\sum_{l' \\in \\mathcal{S}} W_{l', k}`
        for each column :math:`k`.

        Parameters
        ----------
        indices : iterable of int, optional
            Row indices of ``W`` included in the normalization (for example,
            ``model.NMF_simplex()`` or a subset corresponding to ``selected_elts``).
            Defaults to ``self.simplex_indices`` if set, otherwise all rows of ``W``.
        scale : float, default=100.0
            Multiplier applied to fractional compositions (100.0 for atomic %).
        return_cov : bool, default=False
            If ``True``, also return the ``(K, len(indices), len(indices))`` per-phase
            covariance matrices.

        Returns
        -------
        C : np.ndarray of shape (len(indices), K)
            Normalized compositions.
        std_C : np.ndarray of shape (len(indices), K)
            Marginal 1-sigma uncertainties of ``C``.
        cov_C : np.ndarray of shape (K, len(indices), len(indices)), optional
            Returned only when ``return_cov=True``.
        """
        idx = _resolve_indices(self.W.shape[0], indices, self.simplex_indices)
        s = len(idx)
        K = self.W.shape[1]
        C = np.zeros((s, K), dtype=np.float64)
        std_C = np.zeros((s, K), dtype=np.float64)
        cov_C = np.zeros((K, s, s), dtype=np.float64)

        for k in range(K):
            w_sub = self.W[idx, k]
            cov_sub = self.cov_W[k, :, k, :][np.ix_(idx, idx)]
            c_k, cov_ck, std_ck = delta_method_composition(cov_sub, w_sub, scale=scale)
            C[:, k] = c_k
            std_C[:, k] = std_ck
            cov_C[k] = cov_ck

        if return_cov:
            return C, std_C, cov_C
        return C, std_C

    def pixel_quantification(
        self,
        indices: Iterable[int] | None = None,
        scale: float = 100.0,
        nav_shape: tuple[int, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute pixel-wise normalized elemental maps and their 1-sigma uncertainties.

        Evaluates :math:`Q_{l, r} = \\text{scale} \\cdot (W H)_{l, r} / \\sum_{l' \\in \\mathcal{S}} (W H)_{l', r}`
        and propagates the full joint ``(W, H)`` covariance (including ``W``--``W``,
        pixel ``H``--``H``, and ``W``--``H`` cross-terms) to each pixel.

        Parameters
        ----------
        indices : iterable of int, optional
            Row indices of ``W`` included in the normalization. Defaults to
            ``self.simplex_indices`` if set, otherwise all rows of ``W``.
        scale : float, default=100.0
            Multiplier applied to fractional compositions (100.0 for atomic %).
        nav_shape : tuple of int, optional
            If provided, reshapes the spatial dimension ``p`` to ``nav_shape``.

        Returns
        -------
        Q : np.ndarray of shape (len(indices), *nav_shape) or (len(indices), p)
            Pixel-wise normalized elemental concentrations.
        std_Q : np.ndarray of same shape as ``Q``
            Pixel-wise 1-sigma uncertainties of ``Q``.
        """
        idx = _resolve_indices(self.W.shape[0], indices, self.simplex_indices)
        _, _, std_Q = _compute_pixel_marginals_and_quant(
            self.G,
            self.W,
            self.H,
            self.cov_W,
            self._state,
            quant_indices=idx,
            quant_scale=scale,
        )
        W_sub = self.W[idx, :]
        Q = np.zeros((len(idx), self.H.shape[1]), dtype=np.float64)
        chunk_size = self._state["chunk_size"]
        for start in range(0, self.H.shape[1], chunk_size):
            end = min(start + chunk_size, self.H.shape[1])
            m_chunk = W_sub @ self.H[:, start:end]
            denom = np.sum(m_chunk, axis=0, keepdims=True)
            valid = denom > 0
            q_chunk = np.zeros_like(m_chunk)
            np.divide(m_chunk * scale, denom, out=q_chunk, where=valid)
            Q[:, start:end] = q_chunk

        if nav_shape is not None:
            target_shape = (len(idx),) + tuple(nav_shape)
            return Q.reshape(target_shape), std_Q.reshape(target_shape)
        return Q, std_Q

    def roi_quantification(
        self,
        mask: np.ndarray | None = None,
        indices: Iterable[int] | None = None,
        weights: np.ndarray | None = None,
        scale: float = 100.0,
    ) -> ROIUncertaintyResult:
        """Compute ROI-averaged elemental composition and exact uncertainty.

        Includes all intra-pixel and cross-pixel covariances induced by shared
        factor ``W`` (and by spatial Laplacian coupling when ``method="sandwich"``
        with ``lambda_L > 0``).

        Parameters
        ----------
        mask : np.ndarray of bool, optional
            Boolean pixel mask of shape ``(p,)`` or ``nav_shape``. If ``None``
            and ``weights`` is ``None``, averages uniformly over all ``p`` pixels.
        indices : iterable of int, optional
            Row indices of ``W`` included in the normalization. Defaults to
            ``self.simplex_indices`` if set, otherwise all rows of ``W``.
        weights : np.ndarray of shape (p,) or nav_shape, optional
            Explicit per-pixel weights (takes precedence over ``mask``).
        scale : float, default=100.0
            Multiplier applied to fractional compositions (100.0 for atomic %).

        Returns
        -------
        ROIUncertaintyResult
            Spatially averaged intensity, normalized composition, covariance,
            and marginal 1-sigma uncertainties.
        """
        idx = _resolve_indices(self.W.shape[0], indices, self.simplex_indices)
        p = self.H.shape[1]
        if weights is not None:
            w_pix = np.asarray(to_numpy(weights), dtype=np.float64).reshape(-1)
            if w_pix.shape[0] != p:
                raise ValueError(f"Expected {p} pixel weights, got {w_pix.shape[0]}.")
            n_pixels = int(np.count_nonzero(w_pix))
        elif mask is not None:
            m_flat = np.asarray(to_numpy(mask), dtype=bool).reshape(-1)
            if m_flat.shape[0] != p:
                raise ValueError(
                    f"Expected mask with {p} pixels, got {m_flat.shape[0]}."
                )
            n_pixels = int(np.count_nonzero(m_flat))
            if n_pixels == 0:
                raise ValueError("ROI mask selects zero pixels.")
            w_pix = m_flat.astype(np.float64) / float(n_pixels)
        else:
            n_pixels = p
            w_pix = np.full(p, 1.0 / float(p), dtype=np.float64)

        m_bar = self.W @ (self.H @ w_pix)
        cov_m_bar = _compute_roi_M_covariance(
            self.G, self.W, self.H, self.cov_W, w_pix, self._state
        )

        m_sub = m_bar[idx]
        cov_sub = cov_m_bar[np.ix_(idx, idx)]
        std_sub = np.sqrt(np.maximum(np.diag(cov_sub), 0.0))
        q_bar, cov_q_bar, std_q_bar = delta_method_composition(
            cov_sub, m_sub, scale=scale
        )
        return ROIUncertaintyResult(
            composition=q_bar,
            std_composition=std_q_bar,
            cov_composition=cov_q_bar,
            intensity=m_sub,
            std_intensity=std_sub,
            cov_intensity=cov_sub,
            indices=idx,
            n_pixels=n_pixels,
        )


def _resolve_indices(
    m: int, indices: Iterable[int] | None, default_indices: np.ndarray | None
) -> np.ndarray:
    """Resolve subset row indices for normalization."""
    if indices is None:
        if default_indices is not None:
            return np.asarray(default_indices, dtype=int)
        return np.arange(m, dtype=int)
    idx = np.asarray(list(indices), dtype=int)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("indices must be a non-empty 1D sequence of row indices.")
    if np.any(idx < 0) or np.any(idx >= m):
        raise ValueError(f"indices must lie in [0, {m}), got {idx}.")
    return idx


def delta_method_composition(
    cov_vec: np.ndarray, v: np.ndarray, scale: float = 100.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate covariance onto normalized ratios ``scale * v / sum(v)``.

    Parameters
    ----------
    cov_vec : np.ndarray of shape (..., s, s)
        Covariance matrix (or batch of matrices) of unnormalized nonnegative quantities ``v``.
    v : np.ndarray of shape (..., s)
        Positive vector (or batch of vectors).
    scale : float, default=100.0
        Scale factor applied to normalized fractions.

    Returns
    -------
    comp : np.ndarray of shape (..., s)
        Normalized composition values.
    cov_comp : np.ndarray of shape (..., s, s)
        Propagated covariance matrices.
    std_comp : np.ndarray of shape (..., s)
        Marginal standard deviations ``sqrt(diag(cov_comp))``.
    """
    v_arr = np.asarray(v, dtype=np.float64)
    cov_arr = np.asarray(cov_vec, dtype=np.float64)
    s = v_arr.shape[-1]
    denom = np.sum(v_arr, axis=-1, keepdims=True)
    valid = denom > 0.0

    q = np.zeros_like(v_arr)
    np.divide(v_arr, denom, out=q, where=valid)
    comp = scale * q

    eye = np.eye(s, dtype=np.float64)
    # J_ij = scale * (delta_ij - q_i) / denom
    jac = eye - q[..., :, None]
    inv_denom = np.zeros_like(denom)
    np.divide(scale, denom, out=inv_denom, where=valid)
    jac = jac * inv_denom[..., :, None]

    cov_comp = np.matmul(np.matmul(jac, cov_arr), np.swapaxes(jac, -1, -2))
    diag_idx = np.arange(s)
    std_comp = np.sqrt(np.maximum(cov_comp[..., diag_idx, diag_idx], 0.0))
    return comp, cov_comp, std_comp


def _nullspace_from_constraints(
    n_vars: int,
    free_mask: np.ndarray,
    equality_rows: np.ndarray | None = None,
    rtol: float = 1e-12,
) -> np.ndarray:
    """Return an orthonormal basis ``U`` of shape ``(n_vars, d)`` for the constraint nullspace."""
    free_idx = np.flatnonzero(free_mask)
    if free_idx.size == 0:
        return np.zeros((n_vars, 0), dtype=np.float64)

    if equality_rows is None or equality_rows.shape[0] == 0:
        U = np.zeros((n_vars, free_idx.size), dtype=np.float64)
        U[free_idx, np.arange(free_idx.size)] = 1.0
        return U

    C_free = np.asarray(equality_rows[:, free_idx], dtype=np.float64)
    row_norms = np.linalg.norm(C_free, axis=1)
    C_free = C_free[row_norms > 0.0]
    if C_free.shape[0] == 0:
        U = np.zeros((n_vars, free_idx.size), dtype=np.float64)
        U[free_idx, np.arange(free_idx.size)] = 1.0
        return U

    _, singular_values, vh = np.linalg.svd(C_free, full_matrices=True)
    tol = (
        rtol * max(C_free.shape) * (singular_values[0] if singular_values.size else 1.0)
    )
    rank = int(np.sum(singular_values > tol))
    null_free = vh[rank:, :].T
    U = np.zeros((n_vars, null_free.shape[1]), dtype=np.float64)
    U[free_idx, :] = null_free
    return U


def _sym_pinv_eigh(
    mat: np.ndarray, rcond: float = 1e-10
) -> tuple[np.ndarray, np.ndarray | int]:
    """Compute symmetric PSD pseudoinverse via eigendecomposition (2D or batched 3D)."""
    if mat.shape[-1] == 0:
        if mat.ndim == 2:
            return np.zeros((0, 0), dtype=np.float64), 0
        return np.zeros_like(mat, dtype=np.float64), np.zeros(mat.shape[:-2], dtype=int)

    sym = 0.5 * (mat + np.swapaxes(mat, -1, -2))
    evals, evecs = np.linalg.eigh(sym)
    max_eig = np.maximum(np.max(evals, axis=-1, keepdims=True), 0.0)
    thresh = np.maximum(rcond * max_eig, 1e-30)
    keep = evals > thresh
    inv_evals = np.zeros_like(evals)
    np.divide(1.0, evals, out=inv_evals, where=keep)

    pinv = np.matmul(evecs * inv_evals[..., None, :], np.swapaxes(evecs, -1, -2))
    pinv = 0.5 * (pinv + np.swapaxes(pinv, -1, -2))
    null_dim = np.sum(~keep, axis=-1)
    if mat.ndim == 2:
        return pinv, int(null_dim)
    return pinv, null_dim


def _compute_free_masks(
    W: np.ndarray,
    H: np.ndarray,
    fixed_W: np.ndarray | None,
    fixed_H: np.ndarray | None,
    boundary_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify non-fixed and non-boundary-active parameters in ``W`` and ``H``."""
    free_W = np.ones(W.shape, dtype=bool)
    if fixed_W is not None:
        fw = np.asarray(to_numpy(fixed_W), dtype=np.float64)
        if fw.shape != W.shape:
            raise ValueError(
                f"fixed_W shape {fw.shape} does not match W shape {W.shape}."
            )
        free_W &= fw < 0.0
    if boundary_tol > 0.0:
        free_W &= W > boundary_tol

    free_H = np.ones(H.shape, dtype=bool)
    if fixed_H is not None:
        fh = np.asarray(to_numpy(fixed_H), dtype=np.float64)
        if fh.shape != H.shape:
            raise ValueError(
                f"fixed_H shape {fh.shape} does not match H shape {H.shape}."
            )
        free_H &= fh < 0.0
    if boundary_tol > 0.0:
        free_H &= H > boundary_tol

    return free_W, free_H


def _build_W_tangent_basis(
    m: int,
    K: int,
    free_W: np.ndarray,
    simplex_W: bool,
    simplex_indices: np.ndarray | None,
) -> np.ndarray:
    """Build orthonormal tangent basis ``U_W`` of shape ``(K*m, d_W)`` in ``(k, l)`` order."""
    # Parameter ordering is (k, l) flattened to K * m
    free_W_kl = free_W.T.reshape(-1)
    if not simplex_W:
        return _nullspace_from_constraints(K * m, free_W_kl, None)

    s_idx = np.arange(m, dtype=int) if simplex_indices is None else simplex_indices
    eq_rows = np.zeros((K, K * m), dtype=np.float64)
    for k in range(K):
        eq_rows[k, k * m + s_idx] = 1.0
    return _nullspace_from_constraints(K * m, free_W_kl, eq_rows)


def _group_H_tangent_bases(
    free_H: np.ndarray, simplex_H: bool
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Group pixels by free-variable pattern and build per-pattern bases ``U_H^{(pat)}``."""
    K, p = free_H.shape
    eq_row = np.ones((1, K), dtype=np.float64) if simplex_H else None
    patterns_T = np.ascontiguousarray(free_H.T)  # (p, K)
    # Encode boolean rows into integer codes when K <= 62 for fast grouping
    if K <= 62:
        weights = 1 << np.arange(K, dtype=np.int64)
        codes = patterns_T.astype(np.int64) @ weights
        unique_codes, first_idx, inverse = np.unique(
            codes, return_index=True, return_inverse=True
        )
        unique_masks = patterns_T[first_idx]
    else:
        unique_masks, inverse = np.unique(patterns_T, axis=0, return_inverse=True)

    bases = [_nullspace_from_constraints(K, mask, eq_row) for mask in unique_masks]
    return inverse, bases


def _compute_log_reg_hess_H(
    H: np.ndarray, mu: float | np.ndarray, epsilon_reg: float, reg_scale: float
) -> np.ndarray | None:
    """Diagonal Hessian contribution ``d^2/dH^2 [reg_scale * sum mu_k log(H + eps)]``."""
    mu_arr = np.asarray(to_numpy(mu), dtype=np.float64)
    if np.all(mu_arr == 0.0):
        return None
    K = H.shape[0]
    if mu_arr.ndim == 0:
        mu_vec = np.full((K, 1), float(mu_arr), dtype=np.float64)
    else:
        mu_vec = mu_arr.reshape(K, 1)
    return -reg_scale * mu_vec / np.square(H + float(epsilon_reg))


def _assemble_separable_schur_blocks(
    G: np.ndarray,
    D: np.ndarray,
    H: np.ndarray,
    inv_pat: np.ndarray,
    bases_H: list[np.ndarray],
    reg_hess_H: np.ndarray | None,
    rate_floor: float,
    rcond: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Compute I_WW, Schur complement, and meat terms via chunked O(p) BLAS passes."""
    n, m = G.shape
    K, p = H.shape
    G_D = (G[:, :, None] * D[:, None, :]).reshape(n, m * K)
    DD = (D[:, :, None] * D[:, None, :]).reshape(n, K * K)

    S_WW = np.zeros((n, K * K), dtype=np.float64)
    schur_hess = np.zeros((K * K, m * m), dtype=np.float64)
    schur_meat = (
        np.zeros((K * K, m * m), dtype=np.float64) if reg_hess_H is not None else None
    )
    n_singular_H = 0

    for start in range(0, p, chunk_size):
        end = min(start + chunk_size, p)
        H_c = H[:, start:end]  # (K, b)
        Y_c = np.maximum(D @ H_c, rate_floor)  # (n, b)
        Omega_c = 1.0 / Y_c  # (n, b)

        HH_c = (
            (H_c[:, None, :] * H_c[None, :, :]).reshape(K * K, end - start).T
        )  # (b, K*K)
        S_WW += Omega_c @ HH_c

        I_HH_c = (Omega_c.T @ DD).reshape(end - start, K, K)
        B_c = (G_D.T @ Omega_c).T.reshape(end - start, m, K)

        if reg_hess_H is not None:
            H_hess_c = I_HH_c.copy()
            diag_idx = np.arange(K)
            H_hess_c[:, diag_idx, diag_idx] += reg_hess_H[:, start:end].T
        else:
            H_hess_c = I_HH_c

        P_c = np.zeros_like(I_HH_c)
        pat_c = inv_pat[start:end]
        for pat_id in np.unique(pat_c):
            sel = pat_c == pat_id
            U_pat = bases_H[pat_id]
            if U_pat.shape[1] == 0:
                continue
            F_sub = np.matmul(U_pat.T @ H_hess_c[sel], U_pat)
            F_inv, nulls = _sym_pinv_eigh(F_sub, rcond=rcond)
            n_singular_H += int(np.sum(nulls > 0))
            P_c[sel] = np.matmul(U_pat @ F_inv, U_pat.T)

        Q_c = np.matmul(np.matmul(B_c, P_c), np.swapaxes(B_c, -1, -2))
        schur_hess += HH_c.T @ Q_c.reshape(end - start, m * m)

        if schur_meat is not None:
            # 2 P - P I_fim P
            M_H_c = 2.0 * P_c - np.matmul(np.matmul(P_c, I_HH_c), P_c)
            Q_meat_c = np.matmul(np.matmul(B_c, M_H_c), np.swapaxes(B_c, -1, -2))
            schur_meat += HH_c.T @ Q_meat_c.reshape(end - start, m * m)

    # Contract G^T diag(S_WW) G into (K, m, K, m) -> (K*m, K*m)
    I_WW = np.zeros((K, m, K, m), dtype=np.float64)
    S_4 = S_WW.reshape(n, K, K)
    for k1 in range(K):
        for k2 in range(k1, K):
            block = G.T @ (S_4[:, k1, k2][:, None] * G)
            I_WW[k1, :, k2, :] = block
            if k2 != k1:
                I_WW[k2, :, k1, :] = block.T
    I_WW_mat = I_WW.reshape(K * m, K * m)

    def _unpack_schur(acc: np.ndarray) -> np.ndarray:
        return acc.reshape(K, K, m, m).transpose(0, 2, 1, 3).reshape(K * m, K * m)

    schur_hess_mat = _unpack_schur(schur_hess)
    schur_meat_mat = (
        _unpack_schur(schur_meat) if schur_meat is not None else schur_hess_mat
    )
    return I_WW_mat, schur_hess_mat, schur_meat_mat, S_WW, n_singular_H


def _build_sparse_H_projection(
    inv_pat: np.ndarray, bases_H: list[np.ndarray], K: int, p: int
) -> tuple[sparse.csc_matrix, np.ndarray]:
    """Assemble global block-diagonal tangent projection ``U_H`` of shape ``(K*p, d_H)``."""
    dims = np.array([U.shape[1] for U in bases_H], dtype=int)[inv_pat]
    offsets = np.zeros(p + 1, dtype=int)
    np.cumsum(dims, out=offsets[1:])
    d_H = int(offsets[-1])

    rows_parts = []
    cols_parts = []
    data_parts = []
    for pat_id, U_pat in enumerate(bases_H):
        d_pat = U_pat.shape[1]
        if d_pat == 0:
            continue
        pix = np.flatnonzero(inv_pat == pat_id)
        if pix.size == 0:
            continue
        r_local, c_local = np.nonzero(U_pat)
        vals_local = U_pat[r_local, c_local]
        n_pix = pix.size
        rows = (pix[:, None] * K + r_local[None, :]).ravel()
        cols = (offsets[pix][:, None] + c_local[None, :]).ravel()
        data = np.tile(vals_local, n_pix)
        rows_parts.append(rows)
        cols_parts.append(cols)
        data_parts.append(data)

    if not data_parts:
        return sparse.csc_matrix((K * p, 0), dtype=np.float64), offsets
    U_H = sparse.csc_matrix(
        (
            np.concatenate(data_parts),
            (np.concatenate(rows_parts), np.concatenate(cols_parts)),
        ),
        shape=(K * p, d_H),
    )
    return U_H, offsets


def _assemble_coupled_laplacian_cov_W(
    G: np.ndarray,
    D: np.ndarray,
    H: np.ndarray,
    U_W: np.ndarray,
    inv_pat: np.ndarray,
    bases_H: list[np.ndarray],
    L: sparse.spmatrix | np.ndarray,
    lambda_eff: float,
    reg_hess_H: np.ndarray | None,
    rate_floor: float,
    rcond: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """Exact sparse Schur + sandwich covariance for ``W`` when ``lambda_L > 0``."""
    n, m = G.shape
    K, p = H.shape
    G_D = (G[:, :, None] * D[:, None, :]).reshape(n, m * K)
    DD = (D[:, :, None] * D[:, None, :]).reshape(n, K * K)

    S_WW = np.zeros((n, K * K), dtype=np.float64)
    I_HH_blocks = np.zeros((p, K, K), dtype=np.float64)
    B_W_full = np.zeros((K * p, U_W.shape[1]), dtype=np.float64)

    # Pre-contract U_W for faster B_u assembly:
    # (I_WH @ U_W)[r, k2, u] = sum_{k1, l} B^{(r)}_{l, k2} * H_{k1, r} * U_W[(k1, l), u]
    U_W_kl = U_W.reshape(K, m, -1)  # (K, m, d_W)

    for start in range(0, p, chunk_size):
        end = min(start + chunk_size, p)
        b = end - start
        H_c = H[:, start:end]
        Omega_c = 1.0 / np.maximum(D @ H_c, rate_floor)
        HH_c = (H_c[:, None, :] * H_c[None, :, :]).reshape(K * K, b).T
        S_WW += Omega_c @ HH_c

        I_HH_blocks[start:end] = (Omega_c.T @ DD).reshape(b, K, K)
        B_c = (G_D.T @ Omega_c).T.reshape(b, m, K)  # (b, m, K)

        # sum_{k1, l} H_{k1, r} * B_c[r, l, k2] * U_W_kl[k1, l, u]
        HU_c = np.einsum("kr,kld->rld", H_c, U_W_kl)  # (b, m, d_W)
        BW_c = np.matmul(np.swapaxes(B_c, -1, -2), HU_c)  # (b, K, d_W)
        B_W_full[start * K : end * K, :] = BW_c.reshape(b * K, -1)

    I_WW = np.zeros((K, m, K, m), dtype=np.float64)
    S_4 = S_WW.reshape(n, K, K)
    for k1 in range(K):
        for k2 in range(k1, K):
            block = G.T @ (S_4[:, k1, k2][:, None] * G)
            I_WW[k1, :, k2, :] = block
            if k2 != k1:
                I_WW[k2, :, k1, :] = block.T
    I_WW_mat = I_WW.reshape(K * m, K * m)
    A_u = U_W.T @ I_WW_mat @ U_W

    # Sparse I_HH in pixel-major (r, k) order, size (K*p, K*p)
    I_HH_sparse = _block_diag_pixel_major(I_HH_blocks)

    L_sp = sparse.csc_matrix(L, dtype=np.float64)
    if L_sp.shape != (p, p):
        raise ValueError(
            f"Laplacian shape {L_sp.shape} does not match (p, p)=({p}, {p})."
        )
    # In pixel-major (r, k) order, kron(I_K, L) becomes kron(L, I_K)
    reg_sp = lambda_eff * sparse.kron(
        L_sp, sparse.eye(K, dtype=np.float64), format="csc"
    )
    if reg_hess_H is not None:
        # reg_hess_H is (K, p) -> pixel-major ravel is reg_hess_H.T.ravel()
        reg_sp = reg_sp + sparse.diags(reg_hess_H.T.ravel(), format="csc")

    U_H, offsets = _build_sparse_H_projection(inv_pat, bases_H, K, p)
    d_H = U_H.shape[1]

    I_HH_u = (U_H.T @ I_HH_sparse @ U_H).tocsc()
    C_u = (U_H.T @ (I_HH_sparse + reg_sp) @ U_H).tocsc()
    B_u = (U_H.T @ B_W_full).T  # (d_W, d_H)

    if d_H == 0:
        S_u = A_u
        Sigma_S, null_dim = _sym_pinv_eigh(S_u, rcond=rcond)
        cov_W_flat = U_W @ Sigma_S @ U_W.T
        state = {
            "coupled_laplacian": True,
            "lu": None,
            "U_W": U_W,
            "U_H": U_H,
            "offsets": offsets,
            "X": np.zeros((0, U_W.shape[1]), dtype=np.float64),
            "Sigma_S": Sigma_S,
            "A_u": A_u,
            "B_u": B_u,
            "I_HH_u": I_HH_u,
            "I_HH_blocks": I_HH_blocks,
            "I_WW_mat": I_WW_mat,
            "B_W_full": B_W_full,
        }
        return cov_W_flat, Sigma_S, null_dim, state

    lu = splu(C_u)
    X = lu.solve(B_u.T)  # (d_H, d_W)
    S_u = A_u - B_u @ X
    Sigma_S, null_dim = _sym_pinv_eigh(S_u, rcond=rcond)

    # Sandwich meat: M_W = A_u - B_u X - X^T B_u^T + X^T I_HH_u X
    I_X = I_HH_u @ X
    M_W = A_u - B_u @ X - X.T @ B_u.T + X.T @ I_X
    cov_W_u = Sigma_S @ M_W @ Sigma_S
    cov_W_flat = U_W @ cov_W_u @ U_W.T
    cov_W_flat = 0.5 * (cov_W_flat + cov_W_flat.T)

    state = {
        "coupled_laplacian": True,
        "lu": lu,
        "U_W": U_W,
        "U_H": U_H,
        "offsets": offsets,
        "X": X,
        "Sigma_S": Sigma_S,
        "cov_W_u": cov_W_u,
        "A_u": A_u,
        "B_u": B_u,
        "I_HH_u": I_HH_u,
        "I_HH_blocks": I_HH_blocks,
        "I_WW_mat": I_WW_mat,
        "B_W_full": B_W_full,
    }
    return cov_W_flat, Sigma_S, null_dim, state


def _block_diag_pixel_major(blocks: np.ndarray) -> sparse.csc_matrix:
    """Assemble sparse block-diagonal matrix from ``blocks`` of shape ``(p, K, K)``."""
    p, K, _ = blocks.shape
    r_local, c_local = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
    pix_offsets = (np.arange(p, dtype=int) * K)[:, None, None]
    rows = (pix_offsets + r_local[None, :, :]).ravel()
    cols = (pix_offsets + c_local[None, :, :]).ravel()
    return sparse.csc_matrix((blocks.ravel(), (rows, cols)), shape=(K * p, K * p))


def _compute_pixel_marginals_and_quant(
    G: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    cov_W: np.ndarray,
    state: dict,
    quant_indices: np.ndarray | None = None,
    quant_scale: float = 100.0,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Compute pixel-wise ``std_H``, ``std_M``, and optionally ``std_Q`` on ``quant_indices``."""
    n, m = G.shape
    K, p = H.shape
    D = G @ W
    chunk_size = state["chunk_size"]
    rate_floor = state["rate_floor"]
    rcond = state["rcond"]
    inv_pat = state["inv_pat"]
    bases_H = state["bases_H"]
    reg_hess_H = state["reg_hess_H"]

    use_coupled_dense = False
    if state.get("coupled_laplacian", False):
        U_H = state["U_H"]
        d_H = U_H.shape[1]
        max_dense_H = state.get("max_dense_H", 2500)
        if d_H == 0:
            use_coupled_dense = True
            C_u_inv = np.zeros((0, 0), dtype=np.float64)
        elif d_H <= max_dense_H:
            use_coupled_dense = True
            C_u_inv = state["lu"].solve(np.eye(d_H, dtype=np.float64))
        else:
            warnings.warn(
                f"Coupled Laplacian sandwich pixel-wise variance requires inverting {d_H} "
                f"coupled H DOFs (> max_dense_H={max_dense_H}); falling back to local "
                "separable approximation for per-pixel std_H/std_M/std_Q while cov_W "
                "and ROI uncertainties remain exact.",
                UserWarning,
                stacklevel=3,
            )

    if use_coupled_dense:
        return _pixel_marginals_from_dense_coupled(
            W, H, cov_W, state, C_u_inv, quant_indices, quant_scale
        )

    G_D = (G[:, :, None] * D[:, None, :]).reshape(n, m * K)
    DD = (D[:, :, None] * D[:, None, :]).reshape(n, K * K)
    cov_W_4d = cov_W.reshape(K, m, K, m)

    std_H = np.zeros((K, p), dtype=np.float64)
    std_M = np.zeros((m, p), dtype=np.float64)
    std_Q = (
        np.zeros((len(quant_indices), p), dtype=np.float64)
        if quant_indices is not None
        else None
    )

    if state.get("mu_sandwich", False):
        # Need cov_H_sol_W and M_W in W coordinates for exact mu != 0 sandwich per pixel
        Sigma_H_W = state["Sigma_H_W"]
        Sigma_H_W_4d = Sigma_H_W.reshape(K, m, K, m)
        M_W_4d = state["M_W"].reshape(K, m, K, m)
    else:
        Sigma_H_W_4d = None
        M_W_4d = None

    I_m = np.eye(m, dtype=np.float64)
    if quant_indices is not None:
        W_sub = W[quant_indices, :]

    for start in range(0, p, chunk_size):
        end = min(start + chunk_size, p)
        b = end - start
        H_c = H[:, start:end]
        Omega_c = 1.0 / np.maximum(D @ H_c, rate_floor)
        I_HH_c = (Omega_c.T @ DD).reshape(b, K, K)
        B_c = (G_D.T @ Omega_c).T.reshape(b, m, K)

        if reg_hess_H is not None:
            H_hess_c = I_HH_c.copy()
            diag_idx = np.arange(K)
            H_hess_c[:, diag_idx, diag_idx] += reg_hess_H[:, start:end].T
        else:
            H_hess_c = I_HH_c

        P_c = np.zeros_like(I_HH_c)
        pat_c = inv_pat[start:end]
        for pat_id in np.unique(pat_c):
            sel = pat_c == pat_id
            U_pat = bases_H[pat_id]
            if U_pat.shape[1] == 0:
                continue
            F_sub = np.matmul(U_pat.T @ H_hess_c[sel], U_pat)
            F_inv, _ = _sym_pinv_eigh(F_sub, rcond=rcond)
            P_c[sel] = np.matmul(U_pat @ F_inv, U_pat.T)

        # Build effective m x m operator A_r = I_m - W P_r (B^{(r)})^T
        WP_c = np.matmul(W[None, :, :], P_c)  # (b, m, K)
        A_c = I_m[None, :, :] - np.matmul(WP_c, np.swapaxes(B_c, -1, -2))  # (b, m, m)
        PB_c = np.matmul(P_c, np.swapaxes(B_c, -1, -2))  # (b, K, m)

        HH_mat = (H_c[:, None, :] * H_c[None, :, :]).reshape(K * K, b).T  # (b, K*K)

        if Sigma_H_W_4d is None:
            # Standard CRB: Cov(H_r) = P_r + PB_r^T (sum HH cov_W) PB_r
            cov_W_r = (
                HH_mat @ cov_W_4d.transpose(0, 2, 1, 3).reshape(K * K, m * m)
            ).reshape(b, m, m)
            cov_H_c = P_c + np.matmul(
                np.matmul(PB_c, cov_W_r), np.swapaxes(PB_c, -1, -2)
            )
            cov_M_c = np.matmul(WP_c, W.T[None, :, :]) + np.matmul(
                np.matmul(A_c, cov_W_r), np.swapaxes(A_c, -1, -2)
            )
        else:
            # Exact separable sandwich with mu != 0 (C_reg != C_fim)
            cov_W_r = (
                HH_mat @ cov_W_4d.transpose(0, 2, 1, 3).reshape(K * K, m * m)
            ).reshape(b, m, m)
            sig_W_r = (
                HH_mat @ Sigma_H_W_4d.transpose(0, 2, 1, 3).reshape(K * K, m * m)
            ).reshape(b, m, m)
            # For unit functional on H_r: u_w = -Sigma_H_W B_r P_r, u_{h,r'} = delta P_r - P_{r'} B_{r'}^T u_w
            P_fim_P = np.matmul(np.matmul(P_c, I_HH_c), P_c)
            # Cross matrix on K x K: (B_r - I_fim P_r B_r)
            res_B_c = B_c - np.matmul(B_c, np.matmul(P_c, I_HH_c))  # (b, m, K)
            # Cross term = - P_r (res_B_c)^T sig_W_r (P_c B_c^T)^T + sym
            cross_H = -np.matmul(
                np.matmul(P_c, np.swapaxes(res_B_c, -1, -2)),
                np.matmul(sig_W_r, np.swapaxes(PB_c, -1, -2)),
            )
            cov_H_c = (
                P_fim_P
                + np.matmul(np.matmul(PB_c, cov_W_r), np.swapaxes(PB_c, -1, -2))
                + cross_H
                + np.swapaxes(cross_H, -1, -2)
            )
            # For M_r: W P_r (C_fim P_r W^T) + A_r cov_W_r A_r^T + cross
            cross_M = np.matmul(
                np.matmul(WP_c, np.swapaxes(res_B_c, -1, -2)),
                np.matmul(sig_W_r, np.swapaxes(A_c, -1, -2)),
            )
            cov_M_c = (
                np.matmul(np.matmul(W[None, :, :], P_fim_P), W.T[None, :, :])
                + np.matmul(np.matmul(A_c, cov_W_r), np.swapaxes(A_c, -1, -2))
                + cross_M
                + np.swapaxes(cross_M, -1, -2)
            )

        diag_K = np.arange(K)
        std_H[:, start:end] = np.sqrt(np.maximum(cov_H_c[:, diag_K, diag_K], 0.0)).T
        diag_m = np.arange(m)
        std_M[:, start:end] = np.sqrt(np.maximum(cov_M_c[:, diag_m, diag_m], 0.0)).T

        if quant_indices is not None:
            m_sub_c = (W_sub @ H_c).T  # (b, s)
            cov_sub_c = cov_M_c[:, quant_indices, :][:, :, quant_indices]
            _, _, std_q_c = delta_method_composition(
                cov_sub_c, m_sub_c, scale=quant_scale
            )
            std_Q[:, start:end] = std_q_c.T

    return std_H, std_M, std_Q


def _pixel_marginals_from_dense_coupled(
    W: np.ndarray,
    H: np.ndarray,
    cov_W: np.ndarray,
    state: dict,
    C_u_inv: np.ndarray,
    quant_indices: np.ndarray | None,
    quant_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Exact per-pixel sandwich covariance when ``lambda_L > 0`` and ``d_H <= max_dense_H``."""
    m, K = W.shape
    p = H.shape[1]
    U_W = state["U_W"]
    U_H = state["U_H"]
    X = state["X"]
    Sigma_S = state["Sigma_S"]
    I_HH_u = state["I_HH_u"]
    B_u = state["B_u"]
    A_u = state["A_u"]
    d_H = U_H.shape[1]

    # Precompute full meat sandwich covariance of u = (u_w, u_h) on tangent space?
    # Note M_sandwich_HH = C_u_inv @ I_HH_u @ C_u_inv for exact marginal blocks!
    if d_H > 0:
        # Block inverse H_u^+ = [[Sigma_S, -Sigma_S @ X.T], [-X @ Sigma_S, C_u_inv + X @ Sigma_S @ X.T]]
        H_inv_WH = -Sigma_S @ X.T
        H_inv_HH = C_u_inv + X @ Sigma_S @ X.T
        # Assemble small tangent-space covariance for H and W? Better: for each pixel r,
        # g_H^{(r)} has at most K rows! No: we have C_u_inv of size d_H x d_H (<=2500 x 2500),
        # so exact full tangent sandwich Cov_u = H_inv @ I_u @ H_inv fits in <=50 MB!
        I_u_dense = np.block([[A_u, B_u], [B_u.T, I_HH_u.toarray()]])
        H_u_inv = np.block([[Sigma_S, H_inv_WH], [H_inv_WH.T, H_inv_HH]])
        cov_u = H_u_inv @ I_u_dense @ H_u_inv
        cov_u = 0.5 * (cov_u + cov_u.T)
        d_W = U_W.shape[1]
        cov_uw = cov_u[:d_W, :d_W]
        cov_uwh = cov_u[:d_W, d_W:]
        cov_uh = cov_u[d_W:, d_W:]
    else:
        cov_uw = Sigma_S @ A_u @ Sigma_S
        cov_uwh = np.zeros((U_W.shape[1], 0), dtype=np.float64)
        cov_uh = np.zeros((0, 0), dtype=np.float64)

    offsets = state["offsets"]
    inv_pat = state["inv_pat"]
    bases_H = state["bases_H"]
    U_W_kl = U_W.reshape(K, m, -1)

    std_H = np.zeros((K, p), dtype=np.float64)
    std_M = np.zeros((m, p), dtype=np.float64)
    std_Q = (
        np.zeros((len(quant_indices), p), dtype=np.float64)
        if quant_indices is not None
        else None
    )
    W_sub = W[quant_indices, :] if quant_indices is not None else None

    for r in range(p):
        U_r = bases_H[inv_pat[r]]
        s_r, e_r = offsets[r], offsets[r + 1]
        if e_r > s_r:
            cov_hr = U_r @ cov_uh[s_r:e_r, s_r:e_r] @ U_r.T
            cov_uwh_r = cov_uwh[:, s_r:e_r] @ U_r.T  # (d_W, K)
        else:
            cov_hr = np.zeros((K, K), dtype=np.float64)
            cov_uwh_r = np.zeros((U_W.shape[1], K), dtype=np.float64)

        std_H[:, r] = np.sqrt(np.maximum(np.diag(cov_hr), 0.0))

        # J_W^{(r)} in u_w coordinates: J_u[l, dw] = sum_k H[k, r] U_W_kl[k, l, dw]
        J_u = np.einsum("k,kld->ld", H[:, r], U_W_kl)  # (m, d_W)
        cross = J_u @ cov_uwh_r @ W.T
        cov_Mr = J_u @ cov_uw @ J_u.T + W @ cov_hr @ W.T + cross + cross.T
        std_M[:, r] = np.sqrt(np.maximum(np.diag(cov_Mr), 0.0))

        if quant_indices is not None:
            m_sub_r = W_sub @ H[:, r]
            cov_sub_r = cov_Mr[np.ix_(quant_indices, quant_indices)]
            _, _, std_qr = delta_method_composition(
                cov_sub_r, m_sub_r, scale=quant_scale
            )
            std_Q[:, r] = std_qr

    return std_H, std_M, std_Q


def _compute_roi_M_covariance(
    G: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    cov_W: np.ndarray,
    w_pix: np.ndarray,
    state: dict,
) -> np.ndarray:
    """Compute exact ``(m, m)`` covariance of ``M_bar = sum_r w_r W H_{:, r}``."""
    n, m = G.shape
    K, p = H.shape

    if state.get("coupled_laplacian", False):
        U_W = state["U_W"]
        U_H = state["U_H"]
        X = state["X"]
        Sigma_S = state["Sigma_S"]
        lu = state["lu"]
        B_u = state["B_u"]
        A_u = state["A_u"]
        I_HH_u = state["I_HH_u"]

        h_bar = H @ w_pix  # (K,)
        # G_W in (K, m, m) -> (K*m, m): G_W[(k, l), j] = delta_{l, j} h_bar[k]
        G_W = np.zeros((K, m, m), dtype=np.float64)
        for j in range(m):
            G_W[:, j, j] = h_bar
        G_W_u = U_W.T @ G_W.reshape(K * m, m)  # (d_W, m)

        if U_H.shape[1] == 0:
            u_w_sol = Sigma_S @ G_W_u
            cov_m = u_w_sol.T @ A_u @ u_w_sol
            return 0.5 * (cov_m + cov_m.T)

        # G_H in pixel-major (p, K, m): w_pix[r] * W.T[k, l]
        G_H = (w_pix[:, None, None] * W.T[None, :, :]).reshape(K * p, m)
        G_H_u = U_H.T @ G_H  # (d_H, m)

        Z_H = lu.solve(G_H_u)  # (d_H, m)
        u_w_sol = Sigma_S @ (G_W_u - X.T @ G_H_u)
        u_h_sol = Z_H - X @ u_w_sol

        cross = u_w_sol.T @ (B_u @ u_h_sol)
        cov_m = (
            u_w_sol.T @ A_u @ u_w_sol + cross + cross.T + u_h_sol.T @ (I_HH_u @ u_h_sol)
        )
        return 0.5 * (cov_m + cov_m.T)

    # Separable O(p) pass (covers CRB and mu != 0 sandwich cleanly)
    D = G @ W
    chunk_size = state["chunk_size"]
    rate_floor = state["rate_floor"]
    rcond = state["rcond"]
    inv_pat = state["inv_pat"]
    bases_H = state["bases_H"]
    reg_hess_H = state["reg_hess_H"]

    G_D = (G[:, :, None] * D[:, None, :]).reshape(n, m * K)
    DD = (D[:, :, None] * D[:, None, :]).reshape(n, K * K)
    I_m = np.eye(m, dtype=np.float64)

    pure_H = np.zeros((m, m), dtype=np.float64)
    E_bar = np.zeros((m, K, m), dtype=np.float64)

    use_mu_sandwich = state.get("mu_sandwich", False)
    if use_mu_sandwich:
        E_res_bar = np.zeros((m, K, m), dtype=np.float64)

    for start in range(0, p, chunk_size):
        end = min(start + chunk_size, p)
        b = end - start
        H_c = H[:, start:end]
        w_c = w_pix[start:end]
        if not np.any(w_c) and not use_mu_sandwich:
            # Pixels with zero ROI weight still don't contribute directly,
            # though wait: their W-H coupling in Sigma_W was already inverted!
            continue

        Omega_c = 1.0 / np.maximum(D @ H_c, rate_floor)
        I_HH_c = (Omega_c.T @ DD).reshape(b, K, K)
        B_c = (G_D.T @ Omega_c).T.reshape(b, m, K)

        if reg_hess_H is not None:
            H_hess_c = I_HH_c.copy()
            diag_idx = np.arange(K)
            H_hess_c[:, diag_idx, diag_idx] += reg_hess_H[:, start:end].T
        else:
            H_hess_c = I_HH_c

        P_c = np.zeros_like(I_HH_c)
        pat_c = inv_pat[start:end]
        for pat_id in np.unique(pat_c):
            sel = pat_c == pat_id
            U_pat = bases_H[pat_id]
            if U_pat.shape[1] == 0:
                continue
            F_sub = np.matmul(U_pat.T @ H_hess_c[sel], U_pat)
            F_inv, _ = _sym_pinv_eigh(F_sub, rcond=rcond)
            P_c[sel] = np.matmul(U_pat @ F_inv, U_pat.T)

        WP_c = np.matmul(W[None, :, :], P_c)  # (b, m, K)
        A_c = I_m[None, :, :] - np.matmul(WP_c, np.swapaxes(B_c, -1, -2))  # (b, m, m)

        w2_c = w_c * w_c
        if not use_mu_sandwich:
            # sum_r w_r^2 W P_r W^T
            P_sum = np.einsum("r,rij->ij", w2_c, P_c)
            pure_H += W @ P_sum @ W.T
        else:
            P_fim_P_sum = np.einsum(
                "r,rij->ij", w2_c, np.matmul(np.matmul(P_c, I_HH_c), P_c)
            )
            pure_H += W @ P_fim_P_sum @ W.T
            res_B_c = B_c - np.matmul(B_c, np.matmul(P_c, I_HH_c))
            WP_resB = np.matmul(WP_c, np.swapaxes(res_B_c, -1, -2))  # (b, m, m)
            E_res_bar += np.einsum("r,kr,rlj->lkj", w_c, H_c, WP_resB)

        # Accumulate E_bar[l1, k, l2] += sum_r w_r * H[k, r] * A_r[l1, l2]
        E_bar += np.einsum("r,kr,rlj->lkj", w_c, H_c, A_c)

    E_flat = E_bar.reshape(m, K * m)
    cov_W_flat = cov_W.reshape(K * m, K * m)
    if not use_mu_sandwich:
        cov_m = pure_H + E_flat @ cov_W_flat @ E_flat.T
    else:
        E_res_flat = E_res_bar.reshape(m, K * m)
        Sigma_H_W = state["Sigma_H_W"]
        cross = E_res_flat @ Sigma_H_W @ E_flat.T
        cov_m = pure_H + E_flat @ cov_W_flat @ E_flat.T + cross + cross.T
    return 0.5 * (cov_m + cov_m.T)


def compute_espm_uncertainty(
    G: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    *,
    simplex_W: bool = True,
    simplex_H: bool = False,
    simplex_indices: Iterable[int] | None = None,
    fixed_W: np.ndarray | None = None,
    fixed_H: np.ndarray | None = None,
    treat_boundary_zeros: bool = True,
    boundary_tol: float | None = None,
    log_shift: float = 1e-14,
    rate_floor: float = 1e-12,
    rcond: float = 1e-10,
    chunk_size: int = 2048,
    method: str = "crb",
    lambda_L: float = 0.0,
    L: sparse.spmatrix | np.ndarray | None = None,
    mu: float | np.ndarray = 0.0,
    epsilon_reg: float = 1.0,
    reg_scale: float = 1.0,
    max_dense_H: int = 2500,
    compute_pixel_marginals: bool = True,
) -> ESPMUncertaintyResult:
    """Compute constrained Poisson CRB or sandwich uncertainties for ``Y = G W H``.

    Parameters
    ----------
    G : np.ndarray of shape (n, m)
        Spectral basis dictionary.
    W : np.ndarray of shape (m, K)
        Estimated component coefficients.
    H : np.ndarray of shape (K, p)
        Estimated pixel abundances.
    simplex_W : bool, default=True
        Whether columns of ``W`` (on ``simplex_indices``) satisfy a sum-to-constant
        simplex constraint.
    simplex_H : bool, default=False
        Whether columns of ``H`` satisfy a sum-to-constant simplex constraint.
    simplex_indices : iterable of int, optional
        Row indices of ``W`` participating in the ``simplex_W`` constraint. Defaults
        to all ``m`` rows when ``None``.
    fixed_W : np.ndarray of shape (m, K), optional
        Non-negative entries mark fixed entries in ``W``.
    fixed_H : np.ndarray of shape (K, p), optional
        Non-negative entries mark fixed entries in ``H``.
    treat_boundary_zeros : bool, default=True
        Treat entries clamped at the lower positivity floor as active zero constraints.
    boundary_tol : float, optional
        Threshold below which positive entries are treated as active zeros. Defaults
        to ``max(100.0 * log_shift, 1e-10)`` when ``treat_boundary_zeros=True``.
    log_shift : float, default=1e-14
        Lower positivity floor used by the estimator.
    rate_floor : float, default=1e-12
        Minimum expected Poisson count floor when inverting ``Y_hat = G W H``.
    rcond : float, default=1e-10
        Relative eigenvalue threshold for pseudoinversion and gauge-nullity detection.
    chunk_size : int, default=2048
        Pixel batch size for bounded-memory vectorized contractions.
    method : {"crb", "sandwich"}, default="crb"
        ``"crb"`` computes the pure constrained Poisson Cramér–Rao bound.
        ``"sandwich"`` additionally accounts for ``SmoothNMF`` regularization
        (``lambda_L``, ``mu``).
    lambda_L : float, default=0.0
        Laplacian regularization strength (used when ``method="sandwich"``).
    L : sparse matrix or np.ndarray of shape (p, p), optional
        Graph Laplacian matrix (used when ``method="sandwich"`` and ``lambda_L > 0``).
    mu : float or np.ndarray, default=0.0
        Log-sparsity regularization parameter (used when ``method="sandwich"``).
    epsilon_reg : float, default=1.0
        Offset parameter in the log-sparsity penalty.
    reg_scale : float, default=1.0
        Scaling factor multiplying the regularization Hessian relative to raw Poisson
        counts (``1.0 / norm_factor_`` when ``NMFEstimator.normalize=True``).
    max_dense_H : int, default=2500
        Maximum coupled ``H`` DOFs for exact dense inversion of per-pixel sandwich maps
        when ``lambda_L > 0``.
    compute_pixel_marginals : bool, default=True
        If ``True``, immediately populate ``std_H`` and ``std_M``.

    Returns
    -------
    ESPMUncertaintyResult
        Container holding parameter covariances, marginal standard deviations, and
        composition / quantification propagation methods.
    """
    if method not in ("crb", "sandwich"):
        raise ValueError(
            f"Unknown uncertainty method {method!r}; expected 'crb' or 'sandwich'."
        )

    G_np = np.asarray(to_numpy(G), dtype=np.float64)
    W_np = np.asarray(to_numpy(W), dtype=np.float64)
    H_np = np.asarray(to_numpy(H), dtype=np.float64)

    if G_np.ndim != 2 or W_np.ndim != 2 or H_np.ndim != 2:
        raise ValueError("G, W, and H must all be 2D arrays.")
    n, m = G_np.shape
    if W_np.shape[0] != m or H_np.shape[0] != W_np.shape[1]:
        raise ValueError(
            f"Incompatible shapes: G={G_np.shape}, W={W_np.shape}, H={H_np.shape}."
        )
    K, p = H_np.shape

    s_idx = (
        np.asarray(list(simplex_indices), dtype=int)
        if simplex_indices is not None
        else None
    )
    if treat_boundary_zeros:
        b_tol = (
            float(boundary_tol)
            if boundary_tol is not None
            else max(100.0 * float(log_shift), 1e-10)
        )
    else:
        b_tol = -1.0

    # Adapt chunk_size so working array (n, chunk_size) stays <= ~32 MB
    eff_chunk = max(64, min(int(chunk_size), max(1, 4_000_000 // max(n, 1))))

    free_W, free_H = _compute_free_masks(W_np, H_np, fixed_W, fixed_H, b_tol)
    U_W = _build_W_tangent_basis(m, K, free_W, simplex_W, s_idx)
    inv_pat, bases_H = _group_H_tangent_bases(free_H, simplex_H)

    D = G_np @ W_np
    reg_hess_H = None
    use_coupled = False
    if method == "sandwich":
        reg_hess_H = _compute_log_reg_hess_H(H_np, mu, epsilon_reg, reg_scale)
        if lambda_L != 0.0 and L is not None:
            use_coupled = True

    if use_coupled:
        L_cpu = to_numpy(L)
        cov_W_flat, _, gauge_null_dim, coupled_state = (
            _assemble_coupled_laplacian_cov_W(
                G_np,
                D,
                H_np,
                U_W,
                inv_pat,
                bases_H,
                L_cpu,
                float(lambda_L) * float(reg_scale),
                reg_hess_H,
                rate_floor,
                rcond,
                eff_chunk,
            )
        )
        state = {
            **coupled_state,
            "inv_pat": inv_pat,
            "bases_H": bases_H,
            "reg_hess_H": reg_hess_H,
            "rate_floor": rate_floor,
            "rcond": rcond,
            "chunk_size": eff_chunk,
            "max_dense_H": max_dense_H,
            "mu_sandwich": False,
        }
    else:
        I_WW_mat, schur_hess_mat, schur_meat_mat, _, _ = (
            _assemble_separable_schur_blocks(
                G_np,
                D,
                H_np,
                inv_pat,
                bases_H,
                reg_hess_H,
                rate_floor,
                rcond,
                eff_chunk,
            )
        )
        S_hess_u = U_W.T @ (I_WW_mat - schur_hess_mat) @ U_W
        Sigma_S_hess, gauge_null_dim = _sym_pinv_eigh(S_hess_u, rcond=rcond)

        state = {
            "coupled_laplacian": False,
            "inv_pat": inv_pat,
            "bases_H": bases_H,
            "reg_hess_H": reg_hess_H,
            "rate_floor": rate_floor,
            "rcond": rcond,
            "chunk_size": eff_chunk,
            "mu_sandwich": reg_hess_H is not None,
        }
        if reg_hess_H is None:
            cov_W_flat = U_W @ Sigma_S_hess @ U_W.T
        else:
            M_W_mat = I_WW_mat - schur_meat_mat
            Sigma_H_W = U_W @ Sigma_S_hess @ U_W.T
            cov_W_flat = Sigma_H_W @ M_W_mat @ Sigma_H_W
            state["Sigma_H_W"] = Sigma_H_W
            state["M_W"] = M_W_mat

        cov_W_flat = 0.5 * (cov_W_flat + cov_W_flat.T)

    cov_W = cov_W_flat.reshape(K, m, K, m)
    std_W = np.zeros((m, K), dtype=np.float64)
    for k in range(K):
        std_W[:, k] = np.sqrt(np.maximum(np.diag(cov_W[k, :, k, :]), 0.0))

    if compute_pixel_marginals:
        std_H, std_M, _ = _compute_pixel_marginals_and_quant(
            G_np, W_np, H_np, cov_W, state, quant_indices=None
        )
    else:
        std_H = np.zeros((K, p), dtype=np.float64)
        std_M = np.zeros((m, p), dtype=np.float64)

    return ESPMUncertaintyResult(
        G=G_np,
        W=W_np,
        H=H_np,
        cov_W=cov_W,
        std_W=std_W,
        std_H=std_H,
        std_M=std_M,
        gauge_null_dim=gauge_null_dim,
        is_identifiable=(gauge_null_dim == 0),
        method=method,
        simplex_W=simplex_W,
        simplex_H=simplex_H,
        simplex_indices=s_idx,
        rate_floor=rate_floor,
        _state=state,
    )


def uncertainty_from_estimator(
    estimator,
    *,
    method: str = "crb",
    simplex_indices: Iterable[int] | None = None,
    treat_boundary_zeros: bool = True,
    boundary_tol: float | None = None,
    rate_floor: float = 1e-12,
    rcond: float = 1e-10,
    chunk_size: int = 2048,
    max_dense_H: int = 2500,
    compute_pixel_marginals: bool = True,
) -> ESPMUncertaintyResult:
    """Compute uncertainty directly from a fitted ``NMFEstimator`` / ``SmoothNMF``.

    Parameters
    ----------
    estimator : NMFEstimator
        Fitted ESpM NMF estimator instance containing ``G_``, ``W_``, and ``H_``.
    method : {"crb", "sandwich"}, default="crb"
        Uncertainty estimator type.
    simplex_indices : iterable of int, optional
        Override for rows participating in ``simplex_W``. If ``None``, uses
        ``estimator.physics_model_.NMF_simplex()`` when available.
    treat_boundary_zeros : bool, default=True
        Treat entries pinned at ``log_shift`` as active boundary constraints.
    boundary_tol : float, optional
        Explicit active zero tolerance.
    rate_floor : float, default=1e-12
        Minimum expected Poisson count floor.
    rcond : float, default=1e-10
        Relative eigenvalue threshold for gauge-nullity detection and inversion.
    chunk_size : int, default=2048
        Pixel batch size.
    max_dense_H : int, default=2500
        Maximum coupled ``H`` DOFs for dense per-pixel sandwich computation.
    compute_pixel_marginals : bool, default=True
        Whether to compute per-pixel ``std_H`` and ``std_M``.

    Returns
    -------
    ESPMUncertaintyResult
        Fitted uncertainty object.
    """
    for attr in ("G_", "W_", "H_"):
        if not hasattr(estimator, attr):
            raise ValueError(
                f"Estimator must be fitted before computing uncertainties (missing {attr})."
            )

    physics_model = getattr(estimator, "physics_model_", None)
    if simplex_indices is None and getattr(estimator, "simplex_W", False):
        if physics_model is not None:
            simplex_indices = physics_model.NMF_simplex()

    normalize = getattr(estimator, "normalize", False)
    norm_factor = float(getattr(estimator, "norm_factor_", 1.0))
    reg_scale = 1.0 / norm_factor if normalize and norm_factor > 0.0 else 1.0

    return compute_espm_uncertainty(
        estimator.G_,
        estimator.W_,
        estimator.H_,
        simplex_W=getattr(estimator, "simplex_W", True),
        simplex_H=getattr(estimator, "simplex_H", False),
        simplex_indices=simplex_indices,
        fixed_W=getattr(estimator, "fixed_W", None),
        fixed_H=getattr(estimator, "fixed_H", None),
        treat_boundary_zeros=treat_boundary_zeros,
        boundary_tol=boundary_tol,
        log_shift=getattr(estimator, "log_shift", 1e-14),
        rate_floor=rate_floor,
        rcond=rcond,
        chunk_size=chunk_size,
        method=method,
        lambda_L=getattr(estimator, "lambda_L", 0.0),
        L=getattr(estimator, "L_", None),
        mu=getattr(estimator, "mu", 0.0),
        epsilon_reg=getattr(estimator, "epsilon_reg", 1.0),
        reg_scale=reg_scale,
        max_dense_H=max_dense_H,
        compute_pixel_marginals=compute_pixel_marginals,
    )

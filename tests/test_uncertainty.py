"""Unit and statistical validation tests for espm.uncertainty and EDSespm uncertainty reporting."""

import io
import contextlib
import numpy as np
import pytest
import hyperspy.api as hs

from espm.datasets.base import generate_spim_sample
from espm.datasets.eds_spim import EDSespm
from espm.estimators import SmoothNMF
from espm.models import EDXS
from espm.uncertainty import (
    _nullspace_from_constraints,
    _sym_pinv_eigh,
    compute_espm_uncertainty,
    delta_method_composition,
)
from espm.utils import create_laplacian_matrix, quant_spectrum


def _brute_force_kkt_covariance(
    G,
    W,
    H,
    *,
    simplex_W=True,
    simplex_H=False,
    simplex_indices=None,
    fixed_W=None,
    fixed_H=None,
    boundary_tol=1e-10,
    method="crb",
    lambda_L=0.0,
    L=None,
    mu=0.0,
    epsilon_reg=1.0,
    reg_scale=1.0,
):
    """Explicit full (m*K + K*p)-dimensional FIM / KKT sandwich reference."""
    n, m = G.shape
    K, p = H.shape
    D = G @ W
    Y = np.maximum(D @ H, 1e-12)
    Omega = 1.0 / Y

    dim_W = K * m
    dim_H = p * K
    total_dim = dim_W + dim_H
    FIM = np.zeros((total_dim, total_dim), dtype=np.float64)
    for i in range(n):
        for r in range(p):
            grad = np.zeros(total_dim, dtype=np.float64)
            for k in range(K):
                grad[k * m : (k + 1) * m] = G[i, :] * H[k, r]
                grad[dim_W + r * K + k] = D[i, k]
            FIM += Omega[i, r] * np.outer(grad, grad)

    Hess = FIM.copy()
    if method == "sandwich":
        if lambda_L != 0.0 and L is not None:
            L_dense = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
            Hess[dim_W:, dim_W:] += (lambda_L * reg_scale) * np.kron(L_dense, np.eye(K))
        mu_arr = np.asarray(mu, dtype=np.float64)
        if np.any(mu_arr != 0.0):
            mu_vec = np.broadcast_to(mu_arr, (K,))
            for r in range(p):
                for k in range(K):
                    Hess[dim_W + r * K + k, dim_W + r * K + k] -= (
                        reg_scale * mu_vec[k] / (H[k, r] + epsilon_reg) ** 2
                    )

    free = np.ones(total_dim, dtype=bool)
    if fixed_W is not None:
        free[:dim_W] &= (fixed_W < 0).T.ravel()
    if fixed_H is not None:
        free[dim_W:] &= (fixed_H < 0).T.ravel()
    if boundary_tol > 0:
        free[:dim_W] &= (W > boundary_tol).T.ravel()
        free[dim_W:] &= (H > boundary_tol).T.ravel()

    eq_rows = []
    if simplex_W:
        s_idx = (
            np.arange(m)
            if simplex_indices is None
            else np.asarray(simplex_indices, dtype=int)
        )
        for k in range(K):
            row = np.zeros(total_dim, dtype=np.float64)
            row[k * m + s_idx] = 1.0
            eq_rows.append(row)
    if simplex_H:
        for r in range(p):
            row = np.zeros(total_dim, dtype=np.float64)
            row[dim_W + r * K : dim_W + (r + 1) * K] = 1.0
            eq_rows.append(row)

    eq_mat = np.array(eq_rows) if eq_rows else None
    U = _nullspace_from_constraints(total_dim, free, eq_mat)
    H_pinv_u, null_dim = _sym_pinv_eigh(U.T @ Hess @ U, rcond=1e-12)
    H_pinv = U @ H_pinv_u @ U.T
    cov = H_pinv if method == "crb" else H_pinv @ FIM @ H_pinv
    return cov, null_dim


def test_schur_matches_full_fim_all_modes():
    """Verify O(p) Schur CRB and sandwich covariances against full brute-force KKT."""
    rng = np.random.default_rng(7)
    n, m, K, ny, nx = 36, 5, 3, 3, 4
    p = ny * nx
    G = rng.uniform(0.15, 2.0, size=(n, m))
    # Construct uniquely identifiable W via zero pattern across elements [0..3], keeping row 4 unconstrained
    W = rng.uniform(0.25, 1.0, size=(m, K))
    W[0, 0] = 1e-14
    W[1, 0] = 1e-14
    W[1, 1] = 1e-14
    W[2, 1] = 1e-14
    W[0, 2] = 1e-14
    W[2, 2] = 1e-14
    s_idx = [0, 1, 2, 3]
    W[s_idx, :] /= W[s_idx, :].sum(axis=0, keepdims=True)
    H = rng.uniform(0.2, 1.8, size=(K, p)) * 60.0
    fixed_W = -np.ones_like(W)
    fixed_W[4, 0] = W[4, 0]
    fixed_H = -np.ones_like(H)
    fixed_H[0, 5] = H[0, 5]
    L = create_laplacian_matrix(ny, nx).tocsc()

    mask = np.zeros((ny, nx), dtype=bool)
    mask[0:2, 1:3] = True
    w_pix = mask.ravel().astype(float) / mask.sum()

    configs = [
        ("crb", True, False, 0.0, 0.0),
        ("crb", False, True, 0.0, 0.0),
        ("sandwich", True, False, 0.0, np.array([0.4, 0.2, 0.1])),
        ("sandwich", True, False, 1.5, 0.25),
    ]

    for method, sw, sh, lambda_L, mu in configs:
        res = compute_espm_uncertainty(
            G,
            W,
            H,
            simplex_W=sw,
            simplex_H=sh,
            simplex_indices=s_idx,
            fixed_W=fixed_W,
            fixed_H=fixed_H,
            method=method,
            lambda_L=lambda_L,
            L=L,
            mu=mu,
        )
        cov_ref, n_ref = _brute_force_kkt_covariance(
            G,
            W,
            H,
            simplex_W=sw,
            simplex_H=sh,
            simplex_indices=s_idx,
            fixed_W=fixed_W,
            fixed_H=fixed_H,
            method=method,
            lambda_L=lambda_L,
            L=L,
            mu=mu,
        )
        dim_W = K * m
        assert n_ref == 0
        assert res.is_identifiable
        np.testing.assert_allclose(
            res.cov_W.reshape(dim_W, dim_W),
            cov_ref[:dim_W, :dim_W],
            rtol=1e-7,
            atol=1e-8,
        )

        # Fixed & active zero entries must have zero marginal std
        assert res.std_W[4, 0] == 0.0
        assert res.std_W[0, 0] == 0.0
        assert res.std_H[0, 5] == 0.0

        std_H_ref = (
            np.sqrt(np.maximum(np.diag(cov_ref[dim_W:, dim_W:]), 0.0)).reshape(p, K).T
        )
        np.testing.assert_allclose(res.std_H, std_H_ref, rtol=1e-7, atol=1e-8)

        Q, std_Q = res.pixel_quantification(indices=s_idx, nav_shape=(ny, nx))
        np.testing.assert_allclose(Q.sum(axis=0), 100.0, rtol=1e-12)
        for r in [0, 5, 11]:
            J_r = np.zeros((len(s_idx), dim_W + p * K))
            for i, l in enumerate(s_idx):
                for k in range(K):
                    J_r[i, k * m + l] = H[k, r]
                    J_r[i, dim_W + r * K + k] = W[l, k]
            cov_Mr = J_r @ cov_ref @ J_r.T
            _, _, std_qr_ref = delta_method_composition(
                cov_Mr, W[s_idx, :] @ H[:, r], scale=100.0
            )
            np.testing.assert_allclose(
                std_Q.reshape(len(s_idx), p)[:, r], std_qr_ref, rtol=1e-7, atol=1e-8
            )

        roi = res.roi_quantification(mask=mask, indices=s_idx)
        J_roi = np.zeros((len(s_idx), dim_W + p * K))
        for r in range(p):
            if w_pix[r] == 0.0:
                continue
            for i, l in enumerate(s_idx):
                for k in range(K):
                    J_roi[i, k * m + l] += w_pix[r] * H[k, r]
                    J_roi[i, dim_W + r * K + k] += w_pix[r] * W[l, k]
        cov_Mroi = J_roi @ cov_ref @ J_roi.T
        q_ref, cov_qref, std_qref = delta_method_composition(
            cov_Mroi, W[s_idx, :] @ (H @ w_pix), scale=100.0
        )
        np.testing.assert_allclose(roi.composition, q_ref, rtol=1e-12)
        np.testing.assert_allclose(roi.std_composition, std_qref, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(roi.cov_composition, cov_qref, rtol=1e-7, atol=1e-8)


def test_gauge_degeneracy_and_invariance():
    """Verify gauge nullity detection and strict gauge invariance of M, Q, and ROI."""
    rng = np.random.default_rng(19)
    n, m, K, p = 35, 5, 3, 12
    G = rng.uniform(0.2, 2.0, size=(n, m))
    W = rng.uniform(0.3, 1.2, size=(m, K))
    H = rng.uniform(0.3, 1.5, size=(K, p)) * 80.0

    # Without simplex constraints or zeros, continuous WH = (WA)(A^{-1}H) has K^2 = 9 null modes
    res1 = compute_espm_uncertainty(G, W, H, simplex_W=False, simplex_H=False)
    assert not res1.is_identifiable
    assert res1.gauge_null_dim == K * K

    A = np.array([[1.4, 0.2, 0.1], [0.1, 0.9, 0.15], [0.05, 0.25, 1.3]])
    W_alt = W @ A
    H_alt = np.linalg.solve(A, H)
    assert np.all(W_alt > 1e-6) and np.all(H_alt > 1e-6)

    res2 = compute_espm_uncertainty(G, W_alt, H_alt, simplex_W=False, simplex_H=False)
    assert res2.gauge_null_dim == K * K

    np.testing.assert_allclose(res1.std_M, res2.std_M, rtol=1e-9, atol=1e-9)
    Q1, std_Q1 = res1.pixel_quantification(indices=[0, 1, 2])
    Q2, std_Q2 = res2.pixel_quantification(indices=[0, 1, 2])
    np.testing.assert_allclose(Q1, Q2, rtol=1e-11)
    np.testing.assert_allclose(std_Q1, std_Q2, rtol=1e-9, atol=1e-9)

    roi1 = res1.roi_quantification(indices=[0, 1, 2])
    roi2 = res2.roi_quantification(indices=[0, 1, 2])
    np.testing.assert_allclose(
        roi1.std_composition, roi2.std_composition, rtol=1e-9, atol=1e-9
    )

    # Diagonal column scaling invariance for simplex_W=True normalization
    res_s1 = compute_espm_uncertainty(
        G, W, H, simplex_W=True, simplex_indices=[0, 1, 2, 3]
    )
    scale = np.array([2.5, 0.4, 1.7])
    res_s2 = compute_espm_uncertainty(
        G,
        W * scale[None, :],
        H / scale[:, None],
        simplex_W=True,
        simplex_indices=[0, 1, 2, 3],
    )
    C1, std_C1 = res_s1.component_composition()
    C2, std_C2 = res_s2.component_composition()
    np.testing.assert_allclose(C1, C2, rtol=1e-12)
    np.testing.assert_allclose(std_C1, std_C2, rtol=1e-9, atol=1e-9)
    _, std_Qs1 = res_s1.pixel_quantification()
    _, std_Qs2 = res_s2.pixel_quantification()
    np.testing.assert_allclose(std_Qs1, std_Qs2, rtol=1e-9, atol=1e-9)


def test_normalize_reg_scale_consistency():
    """Verify estimator normalize=True properly rescales regularization vs unnormalized fit."""
    rng = np.random.default_rng(11)
    n, m, K, ny, nx = 30, 4, 2, 4, 4
    p = ny * nx
    G = rng.uniform(0.2, 2.0, size=(n, m))
    W_true = np.array([[0.7, 0.0], [0.3, 0.4], [0.0, 0.6], [0.0, 0.0]])
    W_true[-1, :] = [0.2, 0.1]
    H_true = rng.uniform(0.5, 2.0, size=(K, p)) * 50.0
    Y = rng.poisson(G @ W_true @ H_true).astype(float)

    est = SmoothNMF(
        n_components=K,
        G=G,
        shape_2d=(ny, nx),
        normalize=True,
        lambda_L=0.5,
        mu=0.1,
        max_iter=80,
        verbose=0,
        random_state=0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        est.fit_transform(Y)

    unc_est = est.uncertainties(method="sandwich")
    reg_scale = 1.0 / est.norm_factor_
    unc_manual = compute_espm_uncertainty(
        est.G_,
        est.W_,
        est.H_,
        simplex_W=est.simplex_W,
        simplex_H=est.simplex_H,
        log_shift=est.log_shift,
        method="sandwich",
        lambda_L=est.lambda_L,
        L=est.L_,
        mu=est.mu,
        epsilon_reg=est.epsilon_reg,
        reg_scale=reg_scale,
    )
    np.testing.assert_allclose(unc_est.cov_W, unc_manual.cov_W, rtol=1e-12)
    np.testing.assert_allclose(unc_est.std_H, unc_manual.std_H, rtol=1e-12)


def test_max_dense_H_fallback_warns_cleanly():
    """Large coupled Laplacian maps emit a warning and fall back cleanly for pixel marginals."""
    rng = np.random.default_rng(33)
    n, m, K, ny, nx = 20, 3, 2, 5, 5
    p = ny * nx
    G = rng.uniform(0.2, 1.5, size=(n, m))
    W = np.array([[0.6, 0.0], [0.4, 0.5], [0.0, 0.5]])
    H = rng.uniform(0.5, 2.0, size=(K, p)) * 40.0
    L = create_laplacian_matrix(ny, nx).tocsc()

    with pytest.warns(UserWarning, match="max_dense_H"):
        res = compute_espm_uncertainty(
            G,
            W,
            H,
            method="sandwich",
            lambda_L=1.0,
            L=L,
            max_dense_H=10,
        )
    assert np.all(np.isfinite(res.std_W))
    assert np.all(np.isfinite(res.std_M))
    roi = res.roi_quantification()
    assert np.all(np.isfinite(roi.std_composition))


def _make_synthetic_eds_spim(seed=42, shape_2d=(4, 5), counts_per_pixel=1500):
    model_params = {
        "e_offset": 0.2,
        "e_size": 450,
        "e_scale": 0.02,
        "width_slope": 0.01,
        "width_intercept": 0.065,
        "db_name": "200keV_xrays.json",
        "E0": 200,
        "params_dict": {
            "Abs": {"thickness": 100.0e-7, "toa": 35, "density": 4.0},
            "Det": "SDD_efficiency.txt",
        },
    }
    elements = ["Fe", "Si", "O"]
    model = EDXS(**model_params)
    model.generate_g_matr(
        g_type="bremsstrahlung",
        ignored_elements=[],
        elements=elements,
        elements_dict={},
    )
    # Two pure distinguishable phases + modest bremsstrahlung; model element order matches valid_elts
    elt_order = model.get_elements(False)
    w_map = {
        "26": [0.70, 0.00],  # Fe
        "14": [0.00, 0.34],  # Si
        "8": [0.30, 0.66],  # O
    }
    W_true = np.vstack([w_map[e] for e in elt_order] + [[0.08, 0.06], [0.05, 0.04]])
    W_true = np.maximum(W_true, 1e-14)
    ny, nx = shape_2d
    p = ny * nx
    rng = np.random.default_rng(seed)
    H_true = np.zeros((2, p))
    H_true[0, : p // 2] = 0.85
    H_true[1, : p // 2] = 0.15
    H_true[0, p // 2 :] = 0.20
    H_true[1, p // 2 :] = 0.80
    H_true *= counts_per_pixel / np.sum(model.G @ W_true @ H_true, axis=0).mean()

    Y = rng.poisson(model.G @ W_true @ H_true).astype(np.float64)
    data = Y.T.reshape(ny, nx, model_params["e_size"])
    s = EDSespm(data)
    s.axes_manager[-1].offset = model_params["e_offset"]
    s.axes_manager[-1].scale = model_params["e_scale"]
    s.set_microscope_parameters(
        beam_energy=200, tilt_stage=0.0, azimuth_angle=0.0, elevation_angle=35.0
    )
    s.metadata.Acquisition_instrument.TEM.Stage.tilt_beta = 0.0
    s.set_analysis_parameters(
        thickness=100.0e-7, density=4.0, detector_type="SDD_efficiency.txt"
    )
    s.set_elements(elements=elements)
    s.build_G(problem_type="bremsstrahlung", ignored_elements=[])
    return s, W_true, H_true


def test_eds_spim_uncertainty_integration():
    """End-to-end test on EDSespm: reports, elemental mapping, ROI, and quant_spectrum."""
    s, W_true, _ = _make_synthetic_eds_spim(
        seed=5, shape_2d=(4, 5), counts_per_pixel=2000
    )

    est = SmoothNMF(
        n_components=2,
        G=s.model,
        hspy_comp=True,
        lambda_L=0.0,
        max_iter=150,
        verbose=0,
        random_state=0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        s.decomposition(algorithm=est)

    # Default CRB concentration report (at.% uncertainties)
    elts, W_pct, err_pct = s.concentration_report()
    assert len(elts) == 3
    assert W_pct.shape == (3, 2)
    assert err_pct.shape == (3, 2)
    np.testing.assert_allclose(W_pct.sum(axis=0), 100.0, rtol=1e-6)
    assert np.all(np.isfinite(err_pct))
    # Elements pinned near 0 have zero CRB error instead of exploding to 1e6
    assert np.all(err_pct < 10.0)

    # Subset renormalization and relative/legacy options
    sub_elts, W_sub, err_sub = s.concentration_report(selected_elts=["Fe", "O"])
    assert sub_elts == ["Fe", "O"]
    np.testing.assert_allclose(W_sub.sum(axis=0), 100.0, rtol=1e-6)
    assert err_sub.shape == (2, 2)

    _, _, err_rel = s.concentration_report(relative_error=True)
    assert np.any(np.isinf(err_rel)) or np.all(err_rel >= err_pct)
    _, _, err_leg = s.concentration_report(error_type="legacy")
    assert err_leg.shape == (3, 2)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        s.print_concentration_report()
    out = buf.getvalue()
    assert "std (at.%)" in out
    assert "Cramér–Rao bound" in out

    # Elemental mapping populates error signals without requiring GUI plotting
    s.elemental_mapping(plot=False, skipped_elements=["O"])
    assert len(s.quantification_list) == 2
    assert len(s.quantification_errors_list) == 2
    assert s.quantification_errors_signal.data.shape == (2, 4, 5)
    assert s.quantification_errors_signal_1d.data.shape == (4, 5, 2)
    assert (
        s.quantification_errors_signal.metadata.Signal.quantity == "Atomic % (1-sigma)"
    )
    np.testing.assert_allclose(
        s.quantification_signal.data.sum(axis=0), 100.0, rtol=1e-6
    )
    assert np.all(s.quantification_errors_signal.data > 0.0)
    assert np.all(s.quantification_errors_signal.data < 15.0)

    # ROI uncertainty over subset vs full image
    mask_sub = np.zeros((4, 5), dtype=bool)
    mask_sub[:2, :3] = True
    r_elts, r_sub, r_sub_err = s.roi_uncertainty(mask=mask_sub, skipped_elements=["O"])
    _, r_all, r_all_err = s.roi_uncertainty(skipped_elements=["O"])
    assert r_elts == ["Fe", "Si"]
    np.testing.assert_allclose(r_sub.sum(), 100.0)
    np.testing.assert_allclose(r_all.sum(), 100.0)
    # Larger ROI has lower or equal statistical uncertainty than smaller ROI
    assert np.all(r_all_err <= r_sub_err + 1e-9)

    # Single-spectrum quant_spectrum with return_errors=True
    s_single = hs.signals.Signal1D(s.data[0, 0].copy())
    s_single.axes_manager[-1].offset = s.axes_manager[-1].offset
    s_single.axes_manager[-1].scale = s.axes_manager[-1].scale
    s_single.set_signal_type("EDS_espm")
    s_single.set_microscope_parameters(
        beam_energy=200, tilt_stage=0.0, azimuth_angle=0.0, elevation_angle=35.0
    )
    s_single.metadata.Acquisition_instrument.TEM.Stage.tilt_beta = 0.0
    s_single.set_analysis_parameters(
        thickness=100.0e-7, density=4.0, detector_type="SDD_efficiency.txt"
    )
    s_single.set_elements(elements=["Fe", "Si", "O"])
    q_dict, err_dict, _ = quant_spectrum(s_single, return_errors=True)
    assert set(q_dict.keys()) == {"Fe", "Si", "O"}
    assert set(err_dict.keys()) == {"Fe", "Si", "O"}
    np.testing.assert_allclose(sum(q_dict.values()), 100.0, rtol=1e-6)
    assert all(0.0 < v < 15.0 for v in err_dict.values())

    # Verify split-line elements_dict leaves _lo as nuisance and keeps CRB consistent
    s_split = s.deepcopy()
    s_split.build_G(
        problem_type="bremsstrahlung", elements_dict={"Fe": 3.0}, ignored_elements=[]
    )
    est_split = SmoothNMF(
        n_components=2,
        G=s_split.model,
        hspy_comp=True,
        lambda_L=0.0,
        max_iter=80,
        verbose=0,
        random_state=0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        s_split.decomposition(algorithm=est_split)
    elts_sp, W_sp, err_sp = s_split.concentration_report()
    assert len(elts_sp) == 3
    np.testing.assert_allclose(W_sp.sum(axis=0), 100.0, rtol=1e-6)
    assert np.all(np.isfinite(err_sp))
    s_split.elemental_mapping(plot=False)
    assert s_split.quantification_errors_signal.data.shape == (3, 4, 5)


def test_monte_carlo_crb_calibration():
    """Validate analytical CRB std against empirical Poisson Monte Carlo spread (K=1 & K=2)."""
    # Part A: K=1 single spectrum with realistic peak overlap
    rng = np.random.default_rng(101)
    x_grid = np.linspace(0.5, 8.5, 80)
    g0 = np.exp(-0.5 * ((x_grid - 2.0) / 0.35) ** 2)
    g1 = np.exp(-0.5 * ((x_grid - 2.45) / 0.35) ** 2)  # Overlapping doublet
    g2 = np.exp(-0.5 * ((x_grid - 5.5) / 0.40) ** 2)
    G1 = np.column_stack([g0 / g0.sum(), g1 / g1.sum(), g2 / g2.sum()])
    W1_true = np.array([[0.30], [0.45], [0.25]])
    H1_true = np.array([[4000.0]])
    Y1_true = G1 @ W1_true @ H1_true

    unc1 = compute_espm_uncertainty(G1, W1_true, H1_true, simplex_W=True)
    _, crb_std1 = unc1.component_composition(scale=100.0)

    n_trials = 150
    W1_est = np.zeros((3, n_trials))
    for t in range(n_trials):
        X_t = rng.poisson(Y1_true).astype(float)
        est = SmoothNMF(
            n_components=1, G=G1, lambda_L=0.0, max_iter=150, tol=1e-7, verbose=0
        )
        with contextlib.redirect_stdout(io.StringIO()):
            est.fit_transform(X_t, W=W1_true.copy(), H=H1_true.copy())
        W1_est[:, t] = est.W_[:, 0] / est.W_[:, 0].sum() * 100.0

    mc_std1 = W1_est.std(axis=1, ddof=1)
    ratio1 = mc_std1 / crb_std1[:, 0]
    np.testing.assert_allclose(ratio1, 1.0, atol=0.22)

    # Part B: K=2 identifiable multi-pixel factor model with active zeros and spatial overlap
    G2 = np.column_stack([g0 / g0.sum(), g1 / g1.sum(), g2 / g2.sum()])
    W2_true = np.array([[0.55, 1e-14], [0.45, 0.35], [1e-14, 0.65]])
    W2_true /= W2_true.sum(axis=0, keepdims=True)
    p2 = 8
    H2_true = np.vstack([np.linspace(1.0, 0.2, p2), np.linspace(0.2, 1.0, p2)]) * 1500.0
    Y2_true = G2 @ W2_true @ H2_true

    unc2 = compute_espm_uncertainty(G2, W2_true, H2_true, simplex_W=True)
    _, crb_std2 = unc2.component_composition(scale=100.0)
    _, crb_Q2 = unc2.pixel_quantification(scale=100.0)

    n_trials2 = 120
    C2_est = np.zeros((3, 2, n_trials2))
    Q2_est = np.zeros((3, p2, n_trials2))
    for t in range(n_trials2):
        X_t = rng.poisson(Y2_true).astype(float)
        est = SmoothNMF(
            n_components=2, G=G2, lambda_L=0.0, max_iter=200, tol=1e-7, verbose=0
        )
        with contextlib.redirect_stdout(io.StringIO()):
            est.fit_transform(X_t, W=W2_true.copy(), H=H2_true.copy())
        C2_est[:, :, t] = est.W_ / est.W_.sum(axis=0, keepdims=True) * 100.0
        M_t = est.W_ @ est.H_
        Q2_est[:, :, t] = M_t / M_t.sum(axis=0, keepdims=True) * 100.0

    # Check active interior components and pixel-wise quantification stds
    mc_std_Q2 = Q2_est.std(axis=2, ddof=1)
    ratio_Q2 = mc_std_Q2 / np.maximum(crb_Q2, 1e-12)
    np.testing.assert_allclose(ratio_Q2, 1.0, atol=0.25)

    # Check nonzero elements of W2
    for l, k in [(0, 0), (1, 0), (1, 1), (2, 1)]:
        ratio_ck = C2_est[l, k].std(ddof=1) / crb_std2[l, k]
        assert 0.75 <= ratio_ck <= 1.25

import numpy as np

try:
    import cupy as cp

    CUPY = True
except ImportError:
    CUPY = False


def get_array_module(array):
    return np if not CUPY else cp.get_array_module(array)


def to_numpy(array):
    return array.get() if hasattr(array, "__cuda_array_interface__") else array


# ---------------------------------------------------------------------------
# Fused Elementwise Kernels with automatic NumPy/CuPy dispatch
# ---------------------------------------------------------------------------

_cp_fused_update = None
_cp_fused_div_clip = None
_cp_fused_log_clip = None


def _get_cp_fused_update():
    global _cp_fused_update
    if _cp_fused_update is None and CUPY:
        _cp_fused_update = cp.ElementwiseKernel(
            "T num, T denum, T log_shift",
            "T out",
            "out = max(num / denum, log_shift)",
            "fused_update_kernel",
        )
    return _cp_fused_update


def _get_cp_fused_div_clip():
    global _cp_fused_div_clip
    if _cp_fused_div_clip is None and CUPY:
        _cp_fused_div_clip = cp.ElementwiseKernel(
            "T x, T gwh, T log_shift",
            "T out",
            "out = x / max(gwh, log_shift)",
            "fused_div_clip_kernel",
        )
    return _cp_fused_div_clip


def _get_cp_fused_log_clip():
    global _cp_fused_log_clip
    if _cp_fused_log_clip is None and CUPY:
        _cp_fused_log_clip = cp.ElementwiseKernel(
            "T x, T log_shift",
            "T out",
            "out = log(max(x, log_shift))",
            "fused_log_clip_kernel",
        )
    return _cp_fused_log_clip


def fused_update(num, denum, log_shift, out=None):
    if CUPY and isinstance(num, cp.ndarray):
        kernel = _get_cp_fused_update()
        if out is None:
            return kernel(num, denum, log_shift)
        kernel(num, denum, log_shift, out)
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        if out is None:
            return np.maximum(num / denum, log_shift)
        np.divide(num, denum, out=out)
        return np.maximum(out, log_shift, out=out)


def fused_div_clip(x, gwh, log_shift, out=None):
    if CUPY and isinstance(x, cp.ndarray):
        kernel = _get_cp_fused_div_clip()
        if out is None:
            return kernel(x, gwh, log_shift)
        kernel(x, gwh, log_shift, out)
        return out
    if out is None:
        return x / np.maximum(gwh, log_shift)
    np.maximum(gwh, log_shift, out=out)
    return np.divide(x, out, out=out)


def fused_log_clip(x, log_shift, out=None):
    if CUPY and isinstance(x, cp.ndarray):
        kernel = _get_cp_fused_log_clip()
        if out is None:
            return kernel(x, log_shift)
        kernel(x, log_shift, out)
        return out
    if out is None:
        return np.log(np.maximum(x, log_shift))
    np.maximum(x, log_shift, out=out)
    return np.log(out, out=out)

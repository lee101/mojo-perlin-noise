"""ctypes loader for the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import numbers
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.environ.get(
    "MOJO_PERLIN_NOISE_LIB",
    os.path.join(ROOT, "dist", "libmojo-perlin-noise.so"),
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpn_perlin_scalar": ([I, I, I, F, F, F, F, F, F, F, F], F),
    "mpn_perlin_batch": ([I, I, I, I, I, I, I], I),
    "mpn_perlin_gpu_batch": ([I, I, I, I, I, I, I], I),
    "mpn_simplex2_batch": ([I, I, I, I, F, F, I], I),
    "mpn_simplex3_batch": ([I, I, I, I, F, F, I], I),
}

_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB_PATH):
            raise RuntimeError(
                f"compiled library not found at {LIB_PATH}; run `pixi run build`"
            )
        _library = ctypes.CDLL(LIB_PATH)
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def f64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def i64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def coordinate_f64(values) -> np.ndarray:
    """Convert coordinates without silently rounding large integer inputs."""
    original = np.asarray(values)
    if np.issubdtype(original.dtype, np.integer):
        if original.size and np.any(np.abs(original.astype(object)) > 2**53):
            raise OverflowError("integer coordinates must be exactly representable as float64")
    elif original.dtype == np.dtype(object):
        for value in original.flat:
            if isinstance(value, numbers.Integral) and abs(value) > 2**53:
                raise OverflowError(
                    "integer coordinates must be exactly representable as float64"
                )
    return np.ascontiguousarray(values, dtype=np.float64)


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    if array.dtype not in (np.dtype(np.float64), np.dtype(np.int64)):
        raise TypeError("FFI buffers must use float64 or int64")
    if array.size == 0:
        raise ValueError("empty arrays do not have an FFI buffer")
    return int(array.ctypes.data)

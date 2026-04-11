"""CUDA fused kernels for ClarusLM.

JIT-compiled on first import via torch.utils.cpp_extension.
Falls back gracefully if CUDA is unavailable.
"""

from __future__ import annotations

import os

_module = None
_ce_module = None
_loaded = False
_ce_loaded = False


def _try_load():
    global _module, _loaded
    if _loaded:
        return _module
    _loaded = True

    try:
        import torch
        if not torch.cuda.is_available():
            return None

        from torch.utils.cpp_extension import load

        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        _module = load(
            name="clarus_fused_ops",
            sources=[os.path.join(kernel_dir, "fused_ops_kernel.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception:
        _module = None
    return _module


def _try_load_ce():
    global _ce_module, _ce_loaded
    if _ce_loaded:
        return _ce_module
    _ce_loaded = True

    try:
        import torch
        if not torch.cuda.is_available():
            return None

        from torch.utils.cpp_extension import load

        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        _ce_module = load(
            name="clarus_ce_ops",
            sources=[os.path.join(kernel_dir, "ce_relax_kernel.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception:
        _ce_module = None
    return _ce_module


def get_cuda_ops():
    """Return the CUDA ops module, or None if unavailable."""
    return _try_load()


def get_ce_cuda_ops():
    """Return the CUDA CE ops module, or None if unavailable."""
    return _try_load_ce()

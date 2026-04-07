"""CUDA fused kernels for ClarusLM.

JIT-compiled on first import via torch.utils.cpp_extension.
Falls back gracefully if CUDA is unavailable.
"""

from __future__ import annotations

import os

_module = None
_loaded = False


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


def get_cuda_ops():
    """Return the CUDA ops module, or None if unavailable."""
    return _try_load()

"""Clarus Equation -- CE Field Theory Engine.

Rust backend (clarus/core via PyO3) + PyTorch frontend + CUDA fused kernels.
"""

__version__ = "1.2.0"

from clarus.device import auto_device

try:
    from clarus._rust import (
        QCEngine,
        BrainEngine,
        BrainState,
        CeConstants,
        topk_sparse,
        topk_sparse_batch,
        nn_topk_silu_fwd,
        nn_topk_silu_bwd,
        nn_lbo_fused_fwd,
        nn_power_iter,
        nn_gauge_lattice_fwd,
    )
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

try:
    from clarus.kernels import get_cuda_ops
    HAS_CUDA_KERNELS = get_cuda_ops() is not None
except ImportError:
    HAS_CUDA_KERNELS = False

try:
    from clarus.ce_ops import (
        has_rust as ce_has_rust,
        has_cuda as ce_has_cuda,
        ce_backend,
        pack_sparse as ce_pack_sparse,
        build_metric_basis as ce_build_metric_basis,
        codebook_pull as ce_codebook_pull,
        relax as ce_relax,
        relax_packed as ce_relax_packed,
    )
except ImportError:
    pass

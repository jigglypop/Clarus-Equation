"""Clarus Equation -- CE Field Theory Engine.

CORE (모든 도메인 브랜치 공통):
    engine.py, ce_ops.py, quantum.py
브랜치에 따라 device/constants/utils/runtime 등 brain-agi 전용 모듈은
존재하지 않을 수 있으므로 모든 비-CORE import 는 try/except 로 감싼다.
이 한 파일이 main / domain/* 모든 브랜치에서 동일하게 import-safe 하게 동작한다.
"""

__version__ = "1.2.0"

topk_sparse = None
topk_sparse_batch = None
nn_topk_silu_fwd = None
nn_topk_silu_bwd = None
nn_lbo_fused_fwd = None
nn_power_iter = None
nn_gauge_lattice_fwd = None

auto_device = None
safe_print = None
normalize_vector = None
resolve_device = None
AD = PORTAL = BYPASS = T_WAKE = None
ACTIVE_RATIO = STRUCT_RATIO = BACKGROUND_RATIO = None

BrainRuntime = None
BrainRuntimeConfig = None
BrainRuntimeSnapshot = None
HippocampusMemory = None
ModuleLifecycle = None
RuntimeMode = None
RuntimeStep = None

try:
    from .device import auto_device  # type: ignore[no-redef]
except ImportError:
    pass

try:
    from .constants import (  # type: ignore[no-redef]
        AD, PORTAL, BYPASS, T_WAKE,
        ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO,
    )
except ImportError:
    pass

try:
    from .utils import safe_print, normalize_vector, resolve_device  # type: ignore[no-redef]
except ImportError:
    pass

try:
    from . import _rust as _rust_mod

    topk_sparse = _rust_mod.topk_sparse
    topk_sparse_batch = _rust_mod.topk_sparse_batch
    nn_topk_silu_fwd = _rust_mod.nn_topk_silu_fwd
    nn_topk_silu_bwd = _rust_mod.nn_topk_silu_bwd
    nn_lbo_fused_fwd = _rust_mod.nn_lbo_fused_fwd
    nn_power_iter = _rust_mod.nn_power_iter
    nn_gauge_lattice_fwd = _rust_mod.nn_gauge_lattice_fwd
except ImportError:
    pass

try:
    from .ce_ops import (
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

try:
    from .runtime import (  # type: ignore[no-redef]
        BrainRuntime,
        BrainRuntimeConfig,
        BrainRuntimeSnapshot,
        HippocampusMemory,
        ModuleLifecycle,
        RuntimeMode,
        RuntimeStep,
    )
except ImportError:
    pass

__all__ = [
    "topk_sparse",
    "topk_sparse_batch",
    "nn_topk_silu_fwd",
    "nn_topk_silu_bwd",
    "nn_lbo_fused_fwd",
    "nn_power_iter",
    "nn_gauge_lattice_fwd",
    "BrainRuntime",
    "BrainRuntimeConfig",
    "BrainRuntimeSnapshot",
    "HippocampusMemory",
    "ModuleLifecycle",
    "RuntimeMode",
    "RuntimeStep",
    "auto_device",
    "safe_print",
    "normalize_vector",
    "resolve_device",
    "AD", "PORTAL", "BYPASS", "T_WAKE",
    "ACTIVE_RATIO", "STRUCT_RATIO", "BACKGROUND_RATIO",
]

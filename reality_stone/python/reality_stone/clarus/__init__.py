"""Clarus runtime integrated under :mod:`reality_stone`.

Core modules stay import-safe so optional Rust/CUDA kernels and higher-level
runtime pieces can be used independently from the unified Reality Stone package.
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
RealityStoneStatus = None
has_reality_stone = None
reality_stone_status = None
RuntimeAgent = None
RuntimeAgentConfig = None
RuntimeAgentStep = None
RuntimeTextAgent = None
RuntimeTextAgentTurn = None
TextEnvironment = None
TextEnvironmentStep = None
gibbs_reweight = None
manifest_indices = None
nonselected_residual = None
compose_weighted_kernels = None
tropical_compose = None
born_prior = None

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

try:
    from .reality import (  # type: ignore[no-redef]
        RealityStoneStatus,
        has_reality_stone,
        status as reality_stone_status,
    )
except ImportError:
    pass

try:
    from .agent import (  # type: ignore[no-redef]
        RuntimeAgent,
        RuntimeAgentConfig,
        RuntimeAgentStep,
        RuntimeTextAgent,
        RuntimeTextAgentTurn,
        TextEnvironment,
        TextEnvironmentStep,
    )
except ImportError:
    pass

try:
    from .pre_eq import (  # type: ignore[no-redef]
        born_prior,
        compose_weighted_kernels,
        gibbs_reweight,
        manifest_indices,
        nonselected_residual,
        tropical_compose,
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
    "RealityStoneStatus",
    "has_reality_stone",
    "reality_stone_status",
    "RuntimeAgent",
    "RuntimeAgentConfig",
    "RuntimeAgentStep",
    "RuntimeTextAgent",
    "RuntimeTextAgentTurn",
    "TextEnvironment",
    "TextEnvironmentStep",
    "gibbs_reweight",
    "manifest_indices",
    "nonselected_residual",
    "compose_weighted_kernels",
    "tropical_compose",
    "born_prior",
    "auto_device",
    "safe_print",
    "normalize_vector",
    "resolve_device",
    "AD", "PORTAL", "BYPASS", "T_WAKE",
    "ACTIVE_RATIO", "STRUCT_RATIO", "BACKGROUND_RATIO",
]

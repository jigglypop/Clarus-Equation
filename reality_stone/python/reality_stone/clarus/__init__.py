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
NativeLoopsConfig = None
run_native_loops = None
loop8_route_b = None
loop9_route_b = None
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
cosine_action_evidence = None
AdaptiveTowerController = None
CausalEvent = None
CrossScaleCut = None
PolicyDecision = None
TowerStateToken = None
UpperReset = None
NestedTowerGenerator = None
InfiniteTailCertificate = None
RolloutTailCertificate = None
TowerManifest = None
TowerSpec = None
LocalCloudKernelConfig = None
LocalCloudObservation = None
LocalCloudState = None
LocalCloudTransitionKernel = None
SmallGainCertificate = None
ClarusField = None
ClarusFieldCertificate = None
ClarusFieldConfig = None
ClarusFieldDrive = None
ClarusFieldState = None
ClarusFieldStep = None
PhaseOccupancy = None
bounded_hrr_bind = None
normalized_graph_laplacian = None
project_rows_to_unit_ball = None
prediction_error_gate_scores = None
UnifiedMetricConfig = None
UnifiedMetricState = None
MetricPath = None
MetricGoalReadout = None
MetricSurprise = None
UnifiedMetricCertificate = None
UnifiedMetricCore = None
affine_chart_change = None
CovariantMetricConfig = None
CovariantMetricState = None
RouteChoice = None
MetricFlowCertificate = None
CovariantMetricFlow = None
HomogeneousSignedCueState = None
SignedCueReadout = None
HomogeneousSignedCueCertificate = None
HomogeneousSignedCue = None
EligibilityState = None
HardLatchState = None
HomogeneousCreditState = None
StrictMetricState = None
NoTraceState = None
DelayedCreditCertificate = None
decode_binary_reward = None
EligibilityLearner = None
HardLatchLearner = None
HomogeneousLearner = None
StrictMetricControl = None
NoTraceControl = None
BeliefControlConfig = None
BeliefController = None
BeliefPlan = None
BeliefUpdate = None
gibbs_reweight = None
manifest_indices = None
nonselected_residual = None
compose_weighted_kernels = None
tropical_compose = None
born_prior = None
survival_fraction = None
conditioned_prior = None
tilt_survival = None
layer_cake_survival = None
mean_field_bounds = None
CandidateAnswer = None
EvidenceClaim = None
ClaimAudit = None
ClaimActionWeights = None
ClaimAxisEvidence = None
ClaimGraphEdge = None
DefectWeights = None
PreEqVerifierConfig = None
PreEqVerifier = None
ClaimResidualVerifierConfig = None
ClaimResidualVerifier = None
ManifestDecision = None
ResidualAnswerCandidate = None
ResidualAnswerState = None
ResidualClaim = None
ResidualManifestDecision = None
LabeledCandidateSet = None
VerificationMetrics = None
evaluate_labeled_sets = None
HybridState = None
UniverseState = None
UniverseKernel = None
SourceHybridSubsystem = None
NominalParameters = None
NOMINAL_PARAMETERS = None
SOURCE_FIXED_POINTS = None
CUBE_CORNERS = None
KAPPA_OPEN_RIGHT = None
SOURCE_EXTINCTION_AREA = None
source_hybrid_step = None
iterate_source = None
admissible_kappa = None
RoutedTwoCopy = None
routed_drives = None
occupancy_bit = None
IDENTITY_WEIGHT = None
COMPLETE_BINARY_ROUTER = None
REGISTERED_FLUX_E1 = None
REGISTERED_FLUX_E2 = None
R0_MASS = None
R0_BOUNDARY = None
ROUTING_HORIZON = None
WashedRoleSplit = None
role_split_drives = None
registered_start = None
WASH_TASK_TAU1 = None
WASH_TASK_TAU2 = None
ACTIVITY_PAIR_CIRC_MASS = None
ACTIVITY_PAIR_SIGMA = None
ACTIVITY_NEXT_STAR = None
ACTIVITY_NEXT_CIRC = None
ACTIVITY_DELTA_MASS = None
ACTIVITY_DELTA_BOUNDARY = None
registered_activity_pair = None
activity_readout = None
LOOP_TASK_PHI1 = None
LOOP_TASK_PHI2 = None
loop_gate_drives = None
overwrite_sigma_from_action = None
HostTuple = None
registered_host_pair = None
internal_kernel = None

try:
    from .universe_life_kernel import (  # type: ignore[no-redef]
        CUBE_CORNERS,
        KAPPA_OPEN_RIGHT,
        NOMINAL_PARAMETERS,
        SOURCE_EXTINCTION_AREA,
        SOURCE_FIXED_POINTS,
        HybridState,
        NominalParameters,
        SourceHybridSubsystem,
        UniverseKernel,
        UniverseState,
        admissible_kappa,
        iterate_source,
        occupancy_bit,
        routed_drives,
        source_hybrid_step,
        RoutedTwoCopy,
        IDENTITY_WEIGHT,
        COMPLETE_BINARY_ROUTER,
        REGISTERED_FLUX_E1,
        REGISTERED_FLUX_E2,
        R0_MASS,
        R0_BOUNDARY,
        ROUTING_HORIZON,
        WashedRoleSplit,
        role_split_drives,
        registered_start,
        WASH_TASK_TAU1,
        WASH_TASK_TAU2,
        ACTIVITY_PAIR_CIRC_MASS,
        ACTIVITY_PAIR_SIGMA,
        ACTIVITY_NEXT_STAR,
        ACTIVITY_NEXT_CIRC,
        ACTIVITY_DELTA_MASS,
        ACTIVITY_DELTA_BOUNDARY,
        registered_activity_pair,
        activity_readout,
        LOOP_TASK_PHI1,
        LOOP_TASK_PHI2,
        loop_gate_drives,
        overwrite_sigma_from_action,
        HostTuple,
        registered_host_pair,
        internal_kernel,
    )
except ImportError:
    pass

try:
    from .device import auto_device  # type: ignore[no-redef]
except ImportError:
    pass

try:
    from .covariant_metric_flow import (  # type: ignore[no-redef]
        CovariantMetricConfig,
        CovariantMetricFlow,
        CovariantMetricState,
        MetricFlowCertificate,
        RouteChoice,
    )
except ImportError:
    pass

try:
    from .homogeneous_signed_cue import (  # type: ignore[no-redef]
        HomogeneousSignedCue,
        HomogeneousSignedCueCertificate,
        HomogeneousSignedCueState,
        SignedCueReadout,
    )
except ImportError:
    pass

try:
    from .delayed_linear_credit import (  # type: ignore[no-redef]
        DelayedCreditCertificate,
        EligibilityLearner,
        EligibilityState,
        HardLatchLearner,
        HardLatchState,
        HomogeneousCreditState,
        HomogeneousLearner,
        NoTraceControl,
        NoTraceState,
        StrictMetricControl,
        StrictMetricState,
        decode_binary_reward,
    )
except ImportError:
    pass

try:
    from .local_cloud_kernel import (  # type: ignore[no-redef]
        LocalCloudKernelConfig,
        LocalCloudObservation,
        LocalCloudState,
        LocalCloudTransitionKernel,
        SmallGainCertificate,
    )
except ImportError:
    pass

try:
    from .clarus_field import (  # type: ignore[no-redef]
        ClarusField,
        ClarusFieldCertificate,
        ClarusFieldConfig,
        ClarusFieldDrive,
        ClarusFieldState,
        ClarusFieldStep,
        PhaseOccupancy,
        bounded_hrr_bind,
        normalized_graph_laplacian,
        prediction_error_gate_scores,
        project_rows_to_unit_ball,
    )
except ImportError:
    pass

try:
    from .unified_metric import (  # type: ignore[no-redef]
        MetricGoalReadout,
        MetricPath,
        MetricSurprise,
        UnifiedMetricCertificate,
        UnifiedMetricConfig,
        UnifiedMetricCore,
        UnifiedMetricState,
        affine_chart_change,
    )
except ImportError:
    pass

try:
    from .constants import (  # type: ignore[no-redef]
        AD,
        PORTAL,
        BYPASS,
        T_WAKE,
        ACTIVE_RATIO,
        STRUCT_RATIO,
        BACKGROUND_RATIO,
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
    from .runtime_native_loops import NativeLoopsConfig, loop8_route_b, loop9_route_b, run_native_loops  # type: ignore[no-redef]
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
    from .belief_control import (  # type: ignore[no-redef]
        BeliefControlConfig,
        BeliefController,
        BeliefPlan,
        BeliefUpdate,
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
        cosine_action_evidence,
    )
except ImportError:
    pass

try:
    from .adaptive_scc_tower_controller import (  # type: ignore[no-redef]
        AdaptiveTowerController,
        CausalEvent,
        CrossScaleCut,
        PolicyDecision,
        TowerStateToken,
        UpperReset,
    )
    from .nested_scc_tower import (  # type: ignore[no-redef]
        InfiniteTailCertificate,
        NestedTowerGenerator,
        RolloutTailCertificate,
        TowerManifest,
        TowerSpec,
    )
except ImportError:
    pass

try:
    from .pre_eq import (  # type: ignore[no-redef]
        born_prior,
        compose_weighted_kernels,
        conditioned_prior,
        gibbs_reweight,
        layer_cake_survival,
        manifest_indices,
        mean_field_bounds,
        nonselected_residual,
        survival_fraction,
        tilt_survival,
        tropical_compose,
    )
except ImportError:
    pass

try:
    from .llm_pre_eq import (  # type: ignore[no-redef]
        CandidateAnswer,
        ClaimActionWeights,
        ClaimAudit,
        ClaimAxisEvidence,
        ClaimGraphEdge,
        ClaimResidualVerifier,
        ClaimResidualVerifierConfig,
        DefectWeights,
        EvidenceClaim,
        LabeledCandidateSet,
        ManifestDecision,
        PreEqVerifier,
        PreEqVerifierConfig,
        ResidualAnswerCandidate,
        ResidualAnswerState,
        ResidualClaim,
        ResidualManifestDecision,
        VerificationMetrics,
        evaluate_labeled_sets,
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
    "NativeLoopsConfig",
    "run_native_loops",
    "loop8_route_b",
    "loop9_route_b",
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
    "cosine_action_evidence",
    "AdaptiveTowerController",
    "CausalEvent",
    "CrossScaleCut",
    "PolicyDecision",
    "TowerStateToken",
    "UpperReset",
    "NestedTowerGenerator",
    "InfiniteTailCertificate",
    "RolloutTailCertificate",
    "TowerManifest",
    "TowerSpec",
    "LocalCloudKernelConfig",
    "LocalCloudObservation",
    "LocalCloudState",
    "LocalCloudTransitionKernel",
    "SmallGainCertificate",
    "ClarusField",
    "ClarusFieldCertificate",
    "ClarusFieldConfig",
    "ClarusFieldDrive",
    "ClarusFieldState",
    "ClarusFieldStep",
    "PhaseOccupancy",
    "bounded_hrr_bind",
    "normalized_graph_laplacian",
    "prediction_error_gate_scores",
    "project_rows_to_unit_ball",
    "UnifiedMetricConfig",
    "UnifiedMetricState",
    "MetricPath",
    "MetricGoalReadout",
    "MetricSurprise",
    "UnifiedMetricCertificate",
    "UnifiedMetricCore",
    "affine_chart_change",
    "CovariantMetricConfig",
    "CovariantMetricState",
    "RouteChoice",
    "MetricFlowCertificate",
    "CovariantMetricFlow",
    "HomogeneousSignedCueState",
    "SignedCueReadout",
    "HomogeneousSignedCueCertificate",
    "HomogeneousSignedCue",
    "EligibilityState",
    "HardLatchState",
    "HomogeneousCreditState",
    "StrictMetricState",
    "NoTraceState",
    "DelayedCreditCertificate",
    "decode_binary_reward",
    "EligibilityLearner",
    "HardLatchLearner",
    "HomogeneousLearner",
    "StrictMetricControl",
    "NoTraceControl",
    "BeliefControlConfig",
    "BeliefController",
    "BeliefPlan",
    "BeliefUpdate",
    "gibbs_reweight",
    "manifest_indices",
    "nonselected_residual",
    "compose_weighted_kernels",
    "tropical_compose",
    "born_prior",
    "survival_fraction",
    "conditioned_prior",
    "tilt_survival",
    "layer_cake_survival",
    "mean_field_bounds",
    "CandidateAnswer",
    "EvidenceClaim",
    "ClaimAudit",
    "ClaimActionWeights",
    "ClaimAxisEvidence",
    "ClaimGraphEdge",
    "DefectWeights",
    "PreEqVerifierConfig",
    "PreEqVerifier",
    "ClaimResidualVerifierConfig",
    "ClaimResidualVerifier",
    "ManifestDecision",
    "ResidualAnswerCandidate",
    "ResidualAnswerState",
    "ResidualClaim",
    "ResidualManifestDecision",
    "LabeledCandidateSet",
    "VerificationMetrics",
    "evaluate_labeled_sets",
    "HybridState",
    "UniverseState",
    "UniverseKernel",
    "SourceHybridSubsystem",
    "NominalParameters",
    "NOMINAL_PARAMETERS",
    "SOURCE_FIXED_POINTS",
    "CUBE_CORNERS",
    "KAPPA_OPEN_RIGHT",
    "SOURCE_EXTINCTION_AREA",
    "source_hybrid_step",
    "iterate_source",
    "admissible_kappa",
    "RoutedTwoCopy",
    "routed_drives",
    "occupancy_bit",
    "IDENTITY_WEIGHT",
    "COMPLETE_BINARY_ROUTER",
    "REGISTERED_FLUX_E1",
    "REGISTERED_FLUX_E2",
    "R0_MASS",
    "R0_BOUNDARY",
    "ROUTING_HORIZON",
    "WashedRoleSplit",
    "role_split_drives",
    "registered_start",
    "WASH_TASK_TAU1",
    "WASH_TASK_TAU2",
    "ACTIVITY_PAIR_CIRC_MASS",
    "ACTIVITY_PAIR_SIGMA",
    "ACTIVITY_NEXT_STAR",
    "ACTIVITY_NEXT_CIRC",
    "ACTIVITY_DELTA_MASS",
    "ACTIVITY_DELTA_BOUNDARY",
    "registered_activity_pair",
    "activity_readout",
    "LOOP_TASK_PHI1",
    "LOOP_TASK_PHI2",
    "loop_gate_drives",
    "overwrite_sigma_from_action",
    "HostTuple",
    "registered_host_pair",
    "internal_kernel",
    "auto_device",
    "safe_print",
    "normalize_vector",
    "resolve_device",
    "AD",
    "PORTAL",
    "BYPASS",
    "T_WAKE",
    "ACTIVE_RATIO",
    "STRUCT_RATIO",
    "BACKGROUND_RATIO",
]

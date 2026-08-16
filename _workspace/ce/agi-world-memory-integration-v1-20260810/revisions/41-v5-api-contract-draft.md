# G9-CBM V5 executable/API closure draft

Status: COMPLETE

Gate recommendation: PASS for incorporation into frozen V5 bytes, conditional
on an independent recursive-merge and executable-contract audit. This document
is not permission to implement or to open a registered seed.

## Scope and status

This draft closes only the executable/API P0s recorded in
`37-v4-executable-contract-redteam.md`: the contradictory 10/5 source-manifest
recipe, the stale recall name, the missing typed V5 candidate/control/arena and
capability surfaces, leakage-safe dispatch, and the ambiguous caught-exception
and per-origin-feasibility failure payload. The durable opening witness, hard
interruption handling, row-by-row resource recipe, closed owner inventory, LF
transport, fresh V5 roles, and all other V5 state-machine matters are assigned
to their respective V5 closure drafts.

The audited predecessors remain immutable:

- V4 registration raw SHA-256:
  `f414e7f357a4640039520255889362f3389af5b684f88f410217b201a0be2897`.
- V4 amendment raw SHA-256:
  `47ba50da367cf4fea73b2525c988ba813f101a9d2f197b67d3fe9b4e4b7b1ec3`.
- Recursive V2 -> V3 -> V4 canonical SHA-256:
  `8cd745c17fba41924f9774002b6a7b850611f0eb757909e7c70d1bd9032a6e0e`.

No V4/V5 registered seed, RNG, scientific world, calibration, validation, test,
or scientific runner was opened while preparing this draft.

## 1. Exact V5 recursive-merge edits

[Definition] V5 extends the exact V4 registration above. Arrays replace rather
than concatenate. JSON paths below are arrays of exact object-key strings, not
dot-separated strings.

### 1.1 Required deletions

V5 `delete_paths` must include these exact paths:

```json
[
  ["candidate_api"],
  ["artifact_state_machine"],
  ["test_lock"],
  ["implementation_closure_v4", "stage_protocol"],
  ["implementation_closure_v4", "handcrafted_test_evidence"],
  ["implementation_closure_v4", "test_unlock"],
  ["implementation_closure_v4", "per_origin_feasibility", "failure"]
]
```

The whole inherited `candidate_api` is deleted so no implementation can select
between `CandidateRequestV2`/`CandidateResultV2` and the V5 surface. The complete
V4 state protocol, artifact path table, handcrafted-lock record, and unlock
record are deleted as indivisible stale subtrees; the V5 state-machine closure
re-adds their V5 replacements and must not also request a descendant deletion.
The per-origin failure sentence is replaced only by section 7. All other V4
scientific control definitions remain inherited.

### 1.2 Required replacements of existing paths

The V5 `overrides` object must replace these existing leaves:

```text
episodic_ltm / recall_function
calibration / recall_threshold / wrapper
implementation_dependency_manifest / ordered_primary_paths
implementation_dependency_manifest / ordered_dependency_paths
implementation_dependency_manifest / ordered_source_paths
implementation_dependency_manifest / ordered_path_raw_sha256_recipe
implementation_dependency_manifest / ordered_dependency_subset
implementation_dependency_manifest / callable_boundaries
implementation_dependency_manifest / codec_strip_boundary
```

The first two values must both name the same fully qualified V5 boundary:

```text
reality_stone.clarus.agi_world_memory_integration_v5.scoped_hard_recall_v5
```

The newly added V5 state machine must state that an early caught exception or a
failed per-origin feasibility assertion writes exactly
`RegisteredStageFailureV5`; it never fabricates the inherited full split report.
A complete split may still use the inherited normal PASS/FAIL report.

### 1.3 Required newly allowed paths

V5 `merge_semantics.allowed_new_override_paths` must include exactly these API
lane additions (other V5 lanes may add their separately audited paths):

```json
[
  ["candidate_api"],
  ["artifact_state_machine"],
  ["test_lock"],
  ["implementation_dependency_manifest", "registered_path_call_graph"],
  ["implementation_dependency_manifest", "unlisted_registered_path_callable_policy"],
  ["failure_rules", "registered_stage_failure_v5"],
  ["implementation_closure_v5"]
]
```

The new `artifact_state_machine` and `test_lock` objects are supplied in full by
the V5 state-machine closure. For all three scientific stages their
`terminal_failure_schema` is the literal `RegisteredStageFailureV5`. The new
`implementation_closure_v5` contains the V5 definitions in sections 3--7 and
expressly supersedes the deleted V4 subtrees above.

## 2. Exact 10-source / 5-dependency manifest

[Definition] V5 has exactly five primary paths, in this order:

```json
[
  "experiments/preregistration/agi_world_memory_integration_v5.json",
  "reality_stone/python/reality_stone/clarus/agi_world_memory_integration_v5.py",
  "examples/agi/agi_world_memory_integration_v5_gate.py",
  "tests/test_agi_world_memory_integration_v5.py",
  "tests/test_agi_world_memory_integration_integrity_v5.py"
]
```

[Definition] V5 has exactly five dependency paths, in this order:

```json
[
  "experiments/preregistration/agi_world_memory_integration_v4.json",
  "reality_stone/python/reality_stone/clarus/agi_world_memory_integration_v3.py",
  "experiments/preregistration/episodic_ltm_dream_factorial_v2.json",
  "reality_stone/python/reality_stone/clarus/episodic_ltm_dream_bridge_v2.py",
  "reality_stone/python/reality_stone/clarus/episodic_ltm_dream_bridge.py"
]
```

`ordered_source_paths` is the exact concatenation of those two arrays and
therefore has ten records. The two contradictory inherited strings are replaced
by the following exact recipes:

```text
ordered_path_raw_sha256_recipe = serialize one {path,raw_sha256} record for
every ordered_source_paths entry in that exact order, where
raw_sha256=SHA256((repository_root/path).read_bytes()); this ordered 10-record
array is copied byte-for-value through implementation lock, calibration,
validation, test, integrity, and every unlock record without aggregation

ordered_dependency_subset = records at zero-based indices 5,6,7,8,9 of the
ordered 10-record ordered_path_raw_sha256 array correspond byte-for-value and
in order to the complete five-entry ordered_dependency_paths array
```

The implementation must assert lengths `(5,5,10)`, assert
`ordered_source_paths == ordered_primary_paths + ordered_dependency_paths`, and
assert each path occurs once before an implementation lock can be written.

[Definition] `callable_boundaries` is the following exact ordered list. It is
an actual registered-path boundary list, not an allowed superset:

```text
reality_stone.clarus.agi_world_memory_integration_v5.load_merged_registration_v5
reality_stone.clarus.agi_world_memory_integration_v5.scoped_hard_recall_v5
reality_stone.clarus.agi_world_memory_integration_v5.codec_residual_view_v5
reality_stone.clarus.agi_world_memory_integration_v5.constrained_residual_completion_v5
reality_stone.clarus.agi_world_memory_integration_v5.prepare_recall_batch_v5
reality_stone.clarus.agi_world_memory_integration_v5.execute_candidate_v5
reality_stone.clarus.agi_world_memory_integration_v5.dispatch_condition_v5
reality_stone.clarus.agi_world_memory_integration_v5.evaluate_factorial_seed_v5
reality_stone.clarus.agi_world_memory_integration_v5.fit_shared_core_v5
reality_stone.clarus.agi_world_memory_integration_v5._generate_seed_world_v5
reality_stone.clarus.agi_world_memory_integration_v5._make_store_schema_action_v5
reality_stone.clarus.agi_world_memory_integration_v5.build_evaluation_cues_v5
reality_stone.clarus.agi_world_memory_integration_v5.predict_state_into_v5
reality_stone.clarus.agi_world_memory_integration_v5.planning_cost_v5
reality_stone.clarus.agi_world_memory_integration_v5.paired_interval_v5
reality_stone.clarus.agi_world_memory_integration_v5.factorial_effects_v5
reality_stone.clarus.agi_world_memory_integration_v5.run_agi_world_memory_integration_v5_gate
reality_stone.clarus.agi_world_memory_integration_v5.MaskedCueViewV5
reality_stone.clarus.agi_world_memory_integration_v5.CueIdentityV5
reality_stone.clarus.agi_world_memory_integration_v5.OriginRecallAuditV2
reality_stone.clarus.agi_world_memory_integration_v5.CoreArenaViewV5
reality_stone.clarus.agi_world_memory_integration_v5.CodecArenaViewV5
reality_stone.clarus.agi_world_memory_integration_v5.CostArenaViewV5
reality_stone.clarus.agi_world_memory_integration_v5.ArenaEpisodicViewV5
reality_stone.clarus.agi_world_memory_integration_v5.ResidualSchemaViewV5
reality_stone.clarus.agi_world_memory_integration_v5.WakeActionIndexV5
reality_stone.clarus.agi_world_memory_integration_v5.CandidateControlV5
reality_stone.clarus.agi_world_memory_integration_v5.CandidateRequestV5
reality_stone.clarus.agi_world_memory_integration_v5.CandidateResultTargetV5
reality_stone.clarus.agi_world_memory_integration_v5.CandidateResultV5
reality_stone.clarus.agi_world_memory_integration_v5.ConditionDispatchV5
reality_stone.clarus.agi_world_memory_integration_v5.PublicOriginV5
reality_stone.clarus.agi_world_memory_integration_v5.TruthOriginV5
reality_stone.clarus.agi_world_memory_integration_v5.TrainCalibrationV5
reality_stone.clarus.agi_world_memory_integration_v5.SeedEvaluationV5
reality_stone.clarus.agi_world_memory_integration_v5.ConditionArenaV5
reality_stone.clarus.agi_world_memory_integration_v5.ActualCounterV5
reality_stone.clarus.agi_world_memory_integration_v5.TempArenaV5
reality_stone.clarus.agi_world_memory_integration_v5.TempLeaseV5
reality_stone.clarus.agi_world_memory_integration_v5.ResourceEvidenceV5
reality_stone.clarus.agi_world_memory_integration_v5._RegisteredStageCapabilityV5
reality_stone.clarus.agi_world_memory_integration_v5._open_registered_stage_v5
```

`unlisted_registered_path_callable_policy` is `FORBIDDEN`. In particular, the
registered V5 path may not call any V3 or G7-M function, method, constructor,
or record constructor, including V3 `scoped_hard_recall_v3`,
`execute_candidate_v3`, `fit_shared_core_v3`, `_generate_seed_world_v3`,
`_make_store_schema_action`, `build_evaluation_cues_v3`, or either predecessor
runner. Their locked equations are reproduced in arena-aware V5 boundaries;
their raw dependency hashes remain in the 10-path manifest solely for
provenance and off-range differential tests. The registered-path call graph has
no edge into a V3/G7-M runtime symbol. Every record constructor reachable on the
registered path is named in the callable list above. The V5 codec strip boundary is
`codec_residual_view_v5` only; it accepts `(12,8)`, writes residual columns
`0:4` into its caller-supplied `(12,4)` destination, returns `None`, and columns
`4:8` may not cross.

`registered_path_call_graph` contains exactly the following high-level edges
and phases; separately listed scalar/`*_into` kernels are children of their
enclosing node and no dynamic edge is permitted:

```text
evaluate_factorial_seed_v5
  -> _generate_seed_world_v5                 [common phase only]
  -> build_evaluation_cues_v5                 [common phase only]
  -> dispatch_condition_v5                    [once per fixed cell/control]
       -> ConditionArenaV5
       -> ActualCounterV5
       -> TempArenaV5
       -> _make_store_schema_action_v5
       -> constrained_residual_completion_v5
       -> prepare_recall_batch_v5
            -> scoped_hard_recall_v5
       -> execute_candidate_v5                [24 times]
            -> predict_state_into_v5
            -> codec_residual_view_v5
            -> planning_cost_v5               [8 times/origin]
       -> ConditionArenaV5.seal_origin
       -> TempArenaV5.close_seed
       -> ActualCounterV5.snapshot
       -> ConditionArenaV5.close_seed
  -> paired_interval_v5                       [post-condition aggregation]
  -> factorial_effects_v5                     [post-condition aggregation]

run_agi_world_memory_integration_v5_gate
  -> load_merged_registration_v5
  -> _open_registered_stage_v5
  -> fit_shared_core_v5                       [calibration only]
  -> evaluate_factorial_seed_v5
```

Record construction edges are exactly the V5 record classes in
`callable_boundaries`; they do not create an ndarray owner. No V3/G7-M runtime
edge occurs in this graph.

## 3. Total V5 recall boundary

[Definition] The exact typed signature is:

```python
def scoped_hard_recall_v5(
    store: ArenaEpisodicViewV5,
    cue: MaskedCueViewV5,
    completed_view_out: np.ndarray,
    temp_arena: TempArenaV5,
    counter: ActualCounterV5,
    *,
    enabled: bool,
) -> OriginRecallAuditV2
```

`completed_view_out` must be a writable C-contiguous `float64[12,8]` slice
owned by the current `ConditionArenaV5`. `ArenaEpisodicViewV5` is a non-owning
view over the exact trace, occupancy, provenance, scope-index, and
`CodecArenaViewV5` ConditionArena owners. The function creates no ndarray owner
directly; every workspace is an identity-tracked `TempArenaV5` lease.

`MaskedCueViewV5` has no public data field and has the exact methods
`identity()->CueIdentityV5`,
`is_visible(row:int,col:int)->bool`, and
`visible_value(row:int,col:int)->float`. It exposes no `cue_values` ndarray, raw
buffer protocol, Boolean-indexed getter, or method returning all values.
`visible_value` checks `is_visible` before accessing storage; a hidden request
first increments the registered `masked_cue_coordinate_reads` hard counter and
then raises `PermissionError`. The private common-owner value/mask views are
stage-capability guarded, read-only, and not reachable from candidate code.
`CueIdentityV5` contains only the three opaque tokens and is the only
cue-derived object allowed in `CandidateRequestV5`.

The V5-local `OriginRecallAuditV2` is a frozen, slotted scalar-only record with
exact fields `accepted:bool`, `identity:int16`, `confidence:float64`, and
`scope:uint8` and the inherited codebook invariants. It is named V2 solely to
preserve the inherited public schema; it is not the V3 class, contains no
ndarray, and creates no V3/G7-M constructor edge.

The function first writes the arena-backed codec mean, reshaped by a proved
sharing view, into the destination and then visits all `(row,col)` in ascending
C order, calling `visible_value` once exactly when `is_visible` is true. Exactly
24 coordinates are visible. A whole cue array, Boolean-indexed cue array, or
masked-out value is never materialized or read.

Dispatch is exact:

1. Resolve component scope from `store` and `cue.identity()` before consulting
   `enabled`. A cross-context/port/component identity returns the scope-2
   sentinel `(false,-1,-2.0,2)` and performs zero distance rows.
2. For every valid positive or lure slot, lease only the resource-registered
   standardized-visible,
   twelve-score, and scalar-reduction workspaces. In storage order for its exact
   12 physical indices, compute the inherited masked standardized cosine:
   `(trace_visible-mu_visible)/sigma_visible`, numerator dot product, denominator
   product of L2 norms with floor `1e-12`, and score division. Standardization
   uses `max(store.codec.sigma_codec,1e-8)` exactly. No fingerprint or hidden cue
   coordinate is read.
3. If `enabled is False`, discard all 12 scores after their counters advance,
   keep the clamped fallback, and return the scope-0 sentinel
   `(false,-1,-2.0,0)`. Thus disabled and active valid slots perform equal
   measured distance work, but shadow values cannot reach q, audit, or rollout.
4. If enabled, select `argmax` with first-storage-index tie, set confidence to
   that finite score, and accept iff `confidence > store.threshold`. On
   rejection, keep the clamped fallback and return scope 1 with identity -1.
   On acceptance, copy the selected immutable real trace directly from the
   arena view into `completed_view_out`, clamp the 24 visible values again
   through the accessor, and return its physical ledger identity in `0..95` and
   scope 1.
5. Release every lease before return. Positive/lure valid slots therefore
   execute `48*12=576` distance rows and the 24 cross-port slots execute zero.
   The observed formula, strict threshold,
   reconstruction bytes, tie, sentinel, identity, and call/distance counts are
   mathematically identical to the frozen V3/G7-M boundary, but no V3/G7-M
   runtime callable or owning `RecallResult` constructor executes.

`prepare_recall_batch_v5` invokes this boundary in canonical origin order and
finishes all 24 positive completed views and all 24 own-origin audits before a
single q projection. Calibration uses the same boundary and call-count rules;
there is no second calibration-only recall adapter.

[Axiom: implementation equivalence gate] Before implementation lock, the exact
two V5 test files must run differential tests over handcrafted and off-range,
nonregistered fixtures covering accept, reject, equal-score tie, disabled,
invalid scope, poison-hidden coordinates, and thresholds immediately below,
equal to, and above a score. V5 audit fields, completed views, confidences, and
physical identities must equal the frozen V3/G7-M result byte-for-value. Those
tests may call predecessor functions; registered calibration/validation/test
paths may not.

## 4. Exact candidate, result, and control records

[Definition] Every class below is a frozen, slotted record. Constructors validate
shape, dtype, C-contiguity, ownership, and finiteness without copying an array.
An invalid input hard-fails before candidate computation.

### 4.0 Arena-backed model, codec, cost, and cue identity

The exact non-owning records are:

```text
CoreArenaViewV5(intercept:float64[4], diagonal:float64[4],
                bridge:float64[4], action:float64[4,2])
CodecArenaViewV5(mu_codec:float64[96], sigma_codec:float64[96])
CostArenaViewV5(mu_x:float64[4], sigma_x:float64[4],
                action_cost_weight:float, success_threshold:float,
                invalid_penalty:float)
CueIdentityV5(context_token:OpaqueTokenV5, prefix_token:OpaqueTokenV5,
              suffix_token:OpaqueTokenV5)
```

The four core arrays are basic-slice views of `learned_core`; codec arrays are
the two row views of `codec_mean_scale`; and cost arrays are the two row views
of `state_mean_scale`. Every view is read-only, `OWNDATA=False`, and must share
memory with its exact ConditionArena owner. The scalar literals remain exactly
`0.02`, `25.0`, and `10000.0`. Constructing `CoreModelV2`, `CodecSpecV2`, or
`CostSpecV2` within a condition interval is forbidden.

The exact nonallocating kernels are:

```python
def predict_state_into_v5(
    core: CoreArenaViewV5,
    state: np.ndarray,
    action: np.ndarray,
    out: np.ndarray,
) -> None

def codec_residual_view_v5(
    codec: np.ndarray,
    codec_spec: CodecArenaViewV5,
    out: np.ndarray,
    *,
    standardized: bool,
) -> None

def planning_cost_v5(
    future: np.ndarray,
    actions: np.ndarray,
    goal: np.ndarray,
    cost_spec: CostArenaViewV5,
    *,
    valid: bool,
) -> float
```

The first two require writable registered-destination or live TempArena views
of shapes `float64[4]` and `float64[12,4]` respectively and return no array.
The cost boundary returns one Python finite float and creates no ndarray owner.
Every ufunc, reduction, and matrix operation either uses scalar loops or an
explicit registered/lease `out=` destination.

### 4.1 `CandidateControlV5`

Exact fields and unsigned-byte codebooks:

```text
q_policy       : uint8; 0=PROJECT_SUPPLIED_COMPLETION, 1=ZERO
schema_policy  : uint8; 0=NORMAL, 1=SYNTHETIC_TO_COMPONENT_FALLBACK, 2=ZERO
rollout_policy : uint8; 0=RECURSIVE, 1=PERSISTENCE
```

No other code is valid. This record contains no condition/cell label.

### 4.2 `CandidateResultTargetV5`

Exact writable fields, all views into the current origin's persistent arena
slices:

```text
predictions          : float64[8,20,4]
inferred_valid       : bool[8,20]
resolved_schema_keys : int16[8,20]
schema_sources       : uint8[8,20]
q_hat                : float64[4]
inferred_costs       : float64[8]
selected_index       : int64[1]
```

Every field must satisfy `np.shares_memory(field, arena.array(owner_name))` for
its registered allocation-ledger owner. No field may own its base. Every byte
must be initialized exactly once by `execute_candidate_v5` before sealing.

### 4.3 `CandidateRequestV5`

Exact fields, in order:

```text
cue_identity              : CueIdentityV5
anchor_state              : float64[4], read-only common input
numeric_actions           : float64[8,20,2], read-only common input
action_tokens             : OpaqueToken[8,20], read-only common input
public_goal               : float64[20,4], read-only common input
cost_spec                 : CostArenaViewV5
codec_spec                : CodecArenaViewV5
core                      : CoreArenaViewV5
action_index              : WakeActionIndexV5, read-only inferred index
schema                    : ResidualSchemaViewV5, read-only inferred table
q_source_completed_view   : float64[12,8], read-only ConditionArenaV5 view
origin_recall_audit       : OriginRecallAuditV2 for this origin, never rotated
control                   : CandidateControlV5
target                    : CandidateResultTargetV5
```

The record has no `MaskedCueViewV5`, raw cue values, cue mask, or episodic store
because all recall is complete before candidate execution. It has no condition
label, condition ordinal, `World`, `Episode`, master seed, stream id, split,
stage capability, evaluator handle, truth object, future state, evaluator
innovation, true q/schema, generator validity, outcome, realized cost, reward,
oracle rank, or lesion object. No object reachable from the request may contain
one of those forbidden fields.

### 4.4 `CandidateResultV5`

Exact fields, in order:

```text
predictions
inferred_valid
resolved_schema_keys
schema_sources
q_hat
inferred_costs
selected_index
origin_recall_audit
candidate_payload_sha256_before_unseal
```

The seven numeric/Boolean fields are read-only views of the exact target slices;
`selected_index` remains `int64[1]`. The audit is the request's own unrotated
audit. The digest is a lowercase 64-hex string. The constructor never calls
`np.array`, `copy`, `stack`, `concatenate`, or any other owning allocation.

`execute_candidate_v5(request: CandidateRequestV5) -> CandidateResultV5` is the
only candidate executor. It projects q from the supplied completion unless the
q code is ZERO; resolves schema normally, replaces source 2 by same-key source
3 before rollout for `SYNTHETIC_TO_COMPONENT_FALLBACK`, or uses a zero residual
with the otherwise registered key/source audit for `ZERO`; and executes either
the inherited recursive equations or exact anchor persistence. It writes only
`request.target`. It cannot call recall, dream, the generator, or evaluator.

[Definition] Candidate sealing occurs before any evaluator-owned truth is read.
Set all seven target views read-only, then hash them in the fixed order above.
For each array append to SHA-256: ASCII field name, one NUL, the literal dtype
token (`<f8`, `|b1`, `<i2`, `|u1`, or `<i8`), one NUL, comma-separated decimal
shape, one NUL, and `memoryview(view).cast("B")` directly. Each arena dtype must
already equal the named little-endian dtype; conversion, staging, concatenation,
and `np.ascontiguousarray` are forbidden. Then append `origin_recall_audit` as
`json.dumps({"accepted":bool(...),"identity":int(...),"confidence":float(...),
"scope":int(...)},sort_keys=True,ensure_ascii=True,separators=(',',':'),
allow_nan=False).encode('ascii')`. The result is
`candidate_payload_sha256_before_unseal`. No target byte may change after this
digest is computed.

## 5. Condition, arena, counter, and capability APIs

### 5.1 `ConditionDispatchV5`

Exact fields:

```text
recall_enabled            : bool
dream_write_enabled       : bool
completion_binding        : uint8; 0=OWN, 1=NEXT_IN_FIXED_STRATUM
candidate_control         : CandidateControlV5
expose_lesion_diagnostic  : bool
```

The fixed dispatcher table is:

| condition | recall | dream write | binding | q | schema | rollout | lesion |
|---|---:|---:|---|---|---|---|---:|
| M00 | false | false | OWN | PROJECT | NORMAL | RECURSIVE | false |
| M10 | true | false | OWN | PROJECT | NORMAL | RECURSIVE | false |
| M01 | false | true | OWN | PROJECT | NORMAL | RECURSIVE | false |
| M11 | true | true | OWN | PROJECT | NORMAL | RECURSIVE | false |
| persistence | false | false | OWN | ZERO | ZERO | PERSISTENCE | false |
| frozen_core | false | false | OWN | ZERO | ZERO | RECURSIVE | false |
| schema_only | false | false | OWN | PROJECT | NORMAL | RECURSIVE | false |
| shuffled_binding | true | false | NEXT_IN_FIXED_STRATUM | PROJECT | NORMAL | RECURSIVE | false |
| zero_q | true | false | OWN | ZERO | NORMAL | RECURSIVE | false |
| unconstrained_lesion | false | false | OWN | PROJECT | NORMAL | RECURSIVE | true |
| zero_synthetic | false | true | OWN | PROJECT | SYNTHETIC_TO_COMPONENT_FALLBACK | RECURSIVE | false |

`schema_only` and `unconstrained_lesion` therefore produce candidate bytes
identical to M00. The latter differs only in the separately exposed common
21/24 lesion diagnostic. `zero_synthetic` still executes the M01 dream calls
and audits. The oracle is absent from this table and exists only as a post-seal
evaluator diagnostic.

### 5.2 Arena and counter signatures

```python
class ConditionArenaV5:
    def __init__(self, registration, *, counter, capability) -> None: ...
    def array(self, owner_name: str) -> np.ndarray: ...
    def origin_target(self, origin_ordinal: int) -> CandidateResultTargetV5: ...
    def seal_origin(self, origin_ordinal: int, audit) -> CandidateResultV5: ...
    def close_seed(self) -> None: ...

class ActualCounterV5:
    def __init__(self, registration) -> None: ...
    def event(self, row_code: np.uint8, delta: int = 1) -> None: ...
    def snapshot(self, condition_arena, temp_arena) -> ResourceEvidenceV5: ...

class TempArenaV5:
    def __init__(self, registration, *, counter, capability) -> None: ...
    def lease(self, owner_code: np.uint16, shape, dtype) -> TempLeaseV5: ...
    def release(self, lease: TempLeaseV5) -> None: ...
    def close_seed(self) -> None: ...
```

`ConditionArenaV5` is the sole persistent owner and allocates the ledger in
listed order. `origin_target` returns only slices; `seal_origin` verifies view
identity, changes the views to read-only, applies the section-4 digest, and does
not copy. `ActualCounterV5.event` accepts only the 20 event row codes frozen in
the resource closure and requires `type(delta) is int and delta == 1`; it cannot
set owner measurements or registered limits. `snapshot` constructs
`ResourceEvidenceV5` from event rows, actual owners, and registered limits
according to that closure. `TempArenaV5` creates exactly one
`uint8[1048576]` C-order owner named `condition_workspace`; every lease is a
non-owning aligned buffer view into that root, identity-tracked, and cannot be
read or released after release. Both `close_seed` methods hard-fail on a live
temporary lease, a result that is not sealed, an untracked owner, or a target
that does not share memory.

Neither arena nor counter object is reachable from a `CandidateRequestV5`
except for the bounded `CandidateResultTargetV5` slices. Candidate code cannot
allocate by name, inspect counters, or infer a condition identity.

### 5.3 Registered-stage capability

`_RegisteredStageCapabilityV5` is module-private, nonserializable, and has the
exact fields `stage_code:uint8`, `registered_seeds:tuple[int,...]`,
`next_seed_ordinal:int`, `artifact_descriptor:int`, `open_witness_sha256:str`,
`closed:bool`, and `_seal:object`. Its constructor requires identity with a
module-private `_CAPABILITY_SEAL_V5`; otherwise it raises `PermissionError`.
Copying, pickling, equality-by-value, and public construction are forbidden.

`open_witness_sha256` is exactly the raw SHA-256 of the role's committed
`ConsumedMarkerV5`; it is never an intent hash or a hash of pending/final bytes.

Only `_open_registered_stage_v5` may construct it, and only after the separate
state-machine closure has durably consumed the exact stage and produced the
open witness. The exact registered-seed evaluator signature is:

```python
def evaluate_factorial_seed_v5(
    capability: _RegisteredStageCapabilityV5,
    seed_ordinal: int,
    master_seed: int,
    calibration: TrainCalibrationV5,
    registration: Mapping[str, Any],
) -> SeedEvaluationV5
```

Before the first RNG/world operation it checks capability identity, open state,
stage role, `seed_ordinal == next_seed_ordinal`, and
`master_seed == registered_seeds[seed_ordinal]`, then increments the ordinal
exactly once. There is no evaluator overload without a capability, no dry-run
argument, no alternate seed helper, and no V3 runner call. The capability is
consumed by the evaluator only and never enters `CandidateRequestV5`.

## 6. Sole leakage-safe dispatcher sequence

`dispatch_condition_v5` accepts a `ConditionDispatchV5`, public-origin records,
inferred store/schema/action objects, and owned arenas. It does not accept a
condition string, truth record, `EvaluationOriginV3`, master seed, or split.
For each condition it must execute this sequence exactly:

1. Run the common dream traversal/audit into arena slices, with writes enabled
   only by `dream_write_enabled`; run all registered disabled shadow work.
2. In canonical origin order, finish all 24 positive completion destinations
   and own-origin recall audits with `prepare_recall_batch_v5`. Lures and
   cross-port diagnostics follow their inherited audit order and never enter
   candidate execution.
3. Freeze those 24 views. For origin `i` in each fixed `(context,port)` stratum,
   choose its own view for `OWN`, or the already completed view
   `V[(i+1) mod 3]` for `NEXT_IN_FIXED_STRATUM`. Keep origin `i`'s cue, anchor,
   actions, goal, schema anchor, and unrotated audit in both cases.
4. Construct the exact `CandidateRequestV5`, call `execute_candidate_v5` once,
   and seal the result. No truth object is reachable during these calls.
5. Return only sealed results and public audits to
   `evaluate_factorial_seed_v5`. Only then may that evaluator read its separately
   held `TruthOriginV5`, compute true J/generator validity, or run section 7's
   feasibility test. Candidate code is not re-entered after truth is unsealed.

The outer evaluator may use the condition name solely as the key of the report
and to select the fixed table row. It passes only the row value to the dispatcher
and only `candidate_control` to candidate code. Structural tests must walk the
complete request object graph and poison every forbidden truth/masked-cue field.

## 7. Exact caught-exception and per-origin terminal failure

[Definition] `RegisteredStageFailureV5` has exactly these 23 fields:

```text
artifact_kind
experiment
stage
status
failure_phase
failure_code
failure_type
failure_message_sha256
failed_seed_ordinal
failed_origin_ordinal
completed_registered_seed_count
expected_registered_seed_count
registered_seed_execution_count
valid_candidate_leq_25_count
valid_candidate_gt_25_count
candidate_payload_sha256_before_unseal
registration_raw_sha256
implementation_lock_raw_sha256
calibration_raw_sha256
validation_raw_sha256
consumed_marker_raw_sha256
predecessor_artifact_raw_sha256
canonical_payload_sha256_excluding_this_field
```

Common values are `artifact_kind="registered_stage_failure_v5"`,
`status="FAIL"`, `stage` in
`implementation_lock|calibration|validation|test`, and lowercase 64-hex
for every nonnull digest. A not-applicable scalar or digest is literal JSON
`null`; fields are never omitted. `expected_registered_seed_count` is zero for
implementation lock, 40 for calibration/validation, and 60 for test. The
`consumed_marker_raw_sha256` is always the exact committed marker raw hash and
is never null. This schema is distinct from the normal
calibration schema and the complete split report and is the only early terminal
JSON schema.

[Definition] The non-self-referential payload hash is exact:

1. Validate that the object has the exact 23-key set and that
   `canonical_payload_sha256_excluding_this_field` is currently null.
2. Copy the object, remove that one key entirely, and serialize the remaining
   22-key object with
   `json.dumps(value,sort_keys=True,indent=2,ensure_ascii=True,allow_nan=False)`
   followed by exactly one LF, encoded UTF-8 without BOM.
3. Store the lowercase SHA-256 hex digest of those bytes in the removed field.
4. Serialize the final 23-key object by the same canonical recipe. Verification
   removes the field and repeats step 2. Hashing a null-valued field, omitting
   any other field, or hashing the final self-containing object is forbidden.

For a caught Python exception, `failure_code="CAUGHT_EXCEPTION"`;
`failure_type=type(exc).__module__+'.'+type(exc).__qualname__`; and
`failure_message_sha256=SHA256(UTF8(failure_type+'\u0000'+str(exc)))`. No
traceback or plain exception message is serialized. `failure_phase` is one of
`PRE_SEED`, `WORLD`, `RECALL`, `DREAM`, `CANDIDATE`, `POST_SEAL_SCORING`,
`REPORT`, or `FINALIZE_CAUGHT`. Ordinals, feasibility counts, and candidate hash
are null until they actually exist. `completed_registered_seed_count` counts
only wholly completed seeds. `registered_seed_execution_count` counts every seed
whose capability ordinal was claimed, including a partially evaluated seed.

For the validation/test per-origin assertion, after all six generator-valid
candidate true costs for a sealed origin are computed, set:

```text
failure_phase = "PER_ORIGIN_FEASIBILITY"
failure_code  = "NO_VALID_COST_LEQ_25" when leq_count=0
                otherwise "NO_VALID_COST_GT_25" when gt_count=0
failure_type = null
failure_message_sha256 = null
valid_candidate_leq_25_count = count(J[k] <= 25 for k=0..5)
valid_candidate_gt_25_count  = count(J[k] > 25 for k=0..5)
```

The two counts are integers in `0..6`, sum to six, and exactly one is zero.
`failed_seed_ordinal` is the zero-based registered role ordinal;
`failed_origin_ordinal` is the zero-based canonical origin ordinal `0..23`;
`completed_registered_seed_count == failed_seed_ordinal`;
`registered_seed_execution_count == failed_seed_ordinal + 1`; and the candidate
digest is the sealed digest for that origin. The artifact is finalized
immediately, no later origin/seed is opened, no resampling/drop/retry occurs,
and no primitive vector, summary, check, or favorable conjunction is fabricated.

This schema covers caught exceptions and the deterministic feasibility failure.
It deliberately does not claim that an uncatchable process death can serialize
JSON; preservation/classification of empty, partial, or corrupt bytes belongs to
the V5 durable-witness/hard-interruption closure.

## 8. No scientific change and closure verdict

[Definition] V5 retains without change the generator and RNG equations,
dimensions, 96-record wake bank, scoped real-only memory semantics, residual-only
dream proposal, fixed cells and controls, H20 action set, costs, calibration
objectives including symbolic `REJECT_ALL`, per-origin threshold 25, metric
denominators, 55 performance checks, 17 hard-zero checks, 12 resource checks,
paired inference, all thresholds, and claim boundary. V5 must use fresh roles
selected by the seed/transport lane because executable bytes change, but that is
not a scientific parameter change.

[Derivation] The 10/5 recipe now has one possible cardinality; both inherited
recall strings name one real typed boundary; every control reaches one sealed
candidate surface without a label or truth reference; the arena/capability
seams are constructor- and ownership-total; and early failure bytes have one
field set and one non-self-referential hash recipe. Therefore no executable/API
P0 from audit 37 remains within this draft's assigned scope.

Gate recommendation: PASS for the V5 executable/API lane after the exact text is
encoded in the V5 amendment/registration and an independent auditor confirms the
recursive merge, callable manifest, object-graph poison tests, arena view tests,
and both terminal failure routes. Until that audit, V4 remains
`BLOCKED_PRE_IMPLEMENTATION` and V5 remains unauthorized to implement or run.

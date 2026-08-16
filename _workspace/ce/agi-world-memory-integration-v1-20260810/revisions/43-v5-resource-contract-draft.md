# G9-CBM V5 resource and registered-call-graph closure draft

Status: COMPLETE

Gate recommendation: PASS for the resource/call-graph lane after the conflicts
with draft 41 identified below are repaired in the frozen V5 bytes and an
independent static audit passes. This draft is not authority to implement V5 or
to open a registered seed.

## Scope

This draft closes only V4 findings P0-R1 and P0-R2 in
`36-v4-independent-prereg-audit.md`. It defines the observed source of every row
of the inherited 29-entry budget vector, makes every persistent ledger array an
actual destination, closes the condition-local NumPy owner set, fixes the
immutable common-input exemption, and freezes the V5 registered-path call graph.
The inherited generator equations, RNG streams, cells, controls, thresholds,
metrics, 29 values, 36-row allocation ledger, 393,216-byte total, and scientific
claim are unchanged.

No V4/V5 registered seed, RNG, scientific world, calibration, validation, test,
or scientific runner was opened while preparing this document. V4 remains
`BLOCKED_PRE_IMPLEMENTATION`.

### Required recursive-merge edits

V5 must delete the inherited path
`["implementation_closure_v4","resource_ownership"]`; retaining it beside a
new V5 rule would leave two authorities. It must replace the existing leaf
`["resources","inactive_shadow_execution","counter_ownership"]` with the
three-source construction in section 1. The following newly allowed paths must
then contain the exact definitions in this draft:

```json
[
  ["resources", "registered_budget_observation_recipe"],
  ["resources", "common_input_owner_manifest"],
  ["resources", "condition_owner_classification"],
  ["resources", "resource_evidence_v5"],
  ["implementation_closure_v5", "resource_ownership"]
]
```

The resource lane also replaces, rather than coexists with, the API lane's
`implementation_dependency_manifest.callable_boundaries` and
`registered_path_call_graph` values after the cross-lane corrections in
section 5 are incorporated. The final recursively merged object must contain
one callable list and one call graph.

The terms below have exact meanings:

- `event_counter`: starts at integer zero and changes only by a registered
  operation boundary after that operation completes. It is never initialized
  from an expected value.
- `owner_measurement`: is recomputed from a live registered arena owner, its
  shape, or its contents by the stated formula. `ActualCounterV5.event` cannot
  write such a row.
- `registered_limit`: is copied from the named V5 registration leaf. It is a
  limit, not claimed observed work. `ActualCounterV5.event` cannot write it.
- `owner`: the root NumPy array reached by following `.base` until `.base is
  None`. A zero-copy view is not an owner and contributes zero additional
  bytes.
- `condition resource interval`: begins immediately before construction of one
  condition's `ConditionArenaV5` and ends only after its candidate results,
  resource evidence, and metadata have been sealed and its `TempArenaV5` has no
  live lease. The inherited equal-budget claim is per such interval.

## 1. Exact 29-row observed-source table

The zero-based row code is the row's position in
`resources.registered_budget_vector`. The three source classes are disjoint and
exhaustive. `ActualCounterV5.event(row_code, delta=1)` accepts only the 20 event
row codes shown below and requires `type(delta) is int and delta == 1`; all other
row codes or deltas raise before mutation.

| row | name | source class | exact observed formula / operation boundary | required value |
|---:|---|---|---|---:|
| 0 | `N_wake_records` | `event_counter` | increment after one canonical wake record has had all 12 residual rows written directly to `ltm_trace_payload` and its ledger codes validated | 96 |
| 1 | `N_wake_transitions` | `event_counter` | increment after each residual row for `(wake_record, phase)` is written; `96*12` calls | 1,152 |
| 2 | `U_core` | `event_counter` | increment only after a fitted core parameter is changed inside a condition interval; there is no registered validation/test or post-calibration call site | 0 |
| 3 | `P_core` | `owner_measurement` | `ConditionArenaV5.array("learned_core").size` | 20 |
| 4 | `N_origins` | `event_counter` | increment once when one canonical origin is opened for its eight candidate rollouts, after its completed view/audit exists | 24 |
| 5 | `K` | `owner_measurement` | `predictions.shape[1]` | 8 |
| 6 | `H` | `owner_measurement` | `predictions.shape[2]` | 20 |
| 7 | `N_rollout_calls` | `event_counter` | increment once on entry to one origin/candidate recursive-or-persistence rollout | 192 |
| 8 | `N_predicted_transitions` | `event_counter` | increment after one finite predicted state row is written directly to `predictions[o,k,h,:]`; invalid inferred actions still run all 20 rows | 3,840 |
| 9 | `N_planner_score_calls` | `event_counter` | increment after one candidate cost is written directly to `inferred_costs[o,k]` | 192 |
| 10 | `B_ltm_trace_bytes` | `owner_measurement` | `ltm_trace_payload.nbytes` | 73,728 |
| 11 | `Q_ltm_call_slots` | `event_counter` | increment at each positive, lure, or cross-port recall-slot dispatch; canonical blocks are `24+24+24` | 72 |
| 12 | `N_scoped_distance_rows` | `event_counter` | increment after one 24-coordinate masked cosine row is actually evaluated and its scalar score stored or discarded; positive and lure slots execute 12 rows each, cross-port slots execute zero | 576 |
| 13 | `N_schema_key_slots` | `owner_measurement` | `schema_payload.shape[0]` | 72 |
| 14 | `N_ordered_pair_enumerations` | `event_counter` | increment after the next canonical `(context,prefix_slot,suffix_slot)` pair is written to `ordered_pair_indices` | 288 |
| 15 | `N_component_port_checks` | `event_counter` | increment after the component/port equality predicate for that pair is evaluated and its reason path chosen | 288 |
| 16 | `N_same_component_pairs` | `event_counter` | increment only on a true component/port equality result | 72 |
| 17 | `N_observed_keys` | `owner_measurement` | before any dream write, `sum(1 for i in range(72) if bool(schema_occupancy[i]))` | 48 |
| 18 | `N_join_candidates` | `event_counter` | increment for a same-component pair whose key is not observed, immediately before its two endpoint computations | 24 |
| 19 | `N_scalar_endpoint_join_values` | `event_counter` | increment after each left or right finite scalar is written directly to `endpoint_join_values[slot,side]` | 48 |
| 20 | `N_dream_output_slots` | `event_counter` | increment after each of the 24 valid-missing output slots has received its complete proposal bytes in `dream_residual_buffer` | 24 |
| 21 | `U_dream_update_slots` | `event_counter` | increment after each output slot's active-or-shadow update decision completes, whether accepted or rejected | 24 |
| 22 | `N_lesion_nonobserved_pairs` | `event_counter` | during the same 288-pair traversal, increment for every key not among the 48 observed keys | 240 |
| 23 | `N_lesion_accepted_slots` | `event_counter` | increment after each of the first 24 canonical nonobserved pairs is written directly to the lesion audit buffer | 24 |
| 24 | `N_lesion_capacity_padding` | `event_counter` | increment for each later nonobserved pair rejected by the fixed lesion capacity | 216 |
| 25 | `N_dream_passes` | `event_counter` | increment once only after the complete shared constrained/shadow traversal and lesion postclassification have completed | 1 |
| 26 | `persistent_numeric_payload_bytes` | `owner_measurement` | sum `nbytes` over the 36 unique roots owned by the current `ConditionArenaV5`, counted once in registered ledger order | 393,216 |
| 27 | `persistent_byte_cap` | `registered_limit` | integer value at `resources.persistent_byte_cap`; serialized with provenance `registration:resources/persistent_byte_cap` | 524,288 |
| 28 | `temporary_workspace_byte_cap` | `registered_limit` | integer value at `resources.temporary_workspace_byte_cap`; serialized with provenance `registration:resources/temporary_workspace_byte_cap` | 1,048,576 |

The exact event-row set is
`{0,1,2,4,7,8,9,11,12,14,15,16,18,19,20,21,22,23,24,25}`;
the owner-measurement set is `{3,5,6,10,13,17,26}`; and the
registered-limit set is `{27,28}`. Their sorted union must be `range(29)` and
their pairwise intersections must be empty.

`actual_budget_vector` is initialized with zero only in event rows and `-1` in
the other nine rows. Event calls mutate their own vector element in place.
`ActualCounterV5.snapshot` fills the seven owner rows from live owners and the
two limit rows from registration, compares all 29 integer elements to
`registered_budget_vector`, then makes both arrays read-only. No API exists to
set an event row to a chosen value or to assign the expected vector after work.

For event provenance, the counter also updates a rolling SHA-256 after every
successful event with the exact bytes
`row_code:uint8 || global_event_ordinal:uint64_le || row_event_ordinal:uint64_le`.
The digest is evidence of call order, not a substitute for the 29 integer
values. Each registered event call site is source-hashed and statically mapped
to one row; dynamic `getattr`, `eval`, callback injection, and counter access by
candidate code are forbidden.

## 2. `ConditionArenaV5`: exact persistent destinations

`ConditionArenaV5` calls `np.zeros(shape,dtype,order="C")` exactly once for each
of the inherited 36 ledger entries and in their listed order. Each returned
array must be C-contiguous, `OWNDATA=True`, `.base is None`, have the registered
dtype/shape/bytes, and share memory with no other persistent owner. A second
allocation, late replacement, resize, memmap, object dtype, or post-hoc result
copy is a resource failure.

The following is the exhaustive writer/consumer map. “Direct” means the named
owner or a basic-slice view of it is the destination of the operation that
produces the value.

| registered owner | sole direct writer and scientific use |
|---|---|
| `ltm_trace_payload` | `_make_store_schema_action_v5` writes each encoded wake residual/fingerprint row directly; active recall and disabled shadow recall read it |
| `ltm_occupancy` | setup writes the queryable-record mask; active cells use it, disabled cells expose no queryable record |
| `ltm_provenance` | setup writes only the registered empty/real codes; the LTM facade validates it before recall |
| `scoped_storage_indices` | setup writes the eight canonical 12-row scopes; recall reads only its selected row view |
| `schema_payload` | setup writes 48 observed residual schemas directly; accepted dream writes update missing residual slices directly; rollout reads it |
| `schema_occupancy` | setup writes observed occupancy and active accepted dream writes its missing slots |
| `schema_provenance` | setup/dream write only codes `0/1/2`; schema lookup reads it to return source `1/2/3` |
| `predictions` | `execute_candidate_v5` writes `predictions[o,k,h,:]` directly and the sealed result exposes only its arena view |
| `inferred_valid` | candidate token/schema validation writes every `[o,k,h]` directly |
| `resolved_schema_keys` | schema lookup writes every key directly, including `-1` only for invalid token padding |
| `schema_sources` | schema lookup writes every source code directly; zero-synthetic changes source 2 to 3 before rollout |
| `q_hat` | candidate q projection or zero-q policy writes `q_hat[o,:]` directly |
| `inferred_costs` | planner writes every `inferred_costs[o,k]` directly |
| `selected_indices` | canonical argmin writes the one `int64` result for each origin directly |
| `learned_core` | condition initialization copies the locked 20 common calibration scalars directly once; all prediction arithmetic uses zero-copy arena slices, never a copied `CoreModelV2` |
| `codec_mean_scale` | initialization writes locked `mu_codec` and `sigma_codec` directly; recall/schema conversion uses the two arena rows |
| `state_mean_scale` | initialization writes locked `mu_x` and `sigma_x` directly; the planner uses the two arena rows |
| `completed_positive_cues` | `prepare_recall_batch_v5` writes fallback/clamped accepted completion bytes directly for all 24 origins before q projection |
| `dream_residual_buffer` | the common 24-slot dream traversal writes every complete proposal directly, including shadow proposals; occupancy controls whether it enters schema |
| `lesion_residual_buffer` | the same 288 traversal writes the first 24 canonical nonobserved lesion proposals directly; it is diagnostic-only |
| `dream_occupancy` | traversal writes accepted-active status per output slot; disabled/shadow or rejected is false |
| `dream_provenance` | traversal writes `1` only for accepted hypothetical outputs and `0` otherwise |
| `lesion_occupancy` | traversal writes true for exactly the first 24 nonobserved pairs in every condition/control |
| `lesion_provenance` | traversal writes exactly three code-1 valid-missing and 21 code-2 cross-port entries |
| `recall_accepted` | recall batch writes positive, lure, then cross-port Boolean blocks directly |
| `recall_identity` | recall batch writes physical ledger index or `-1` directly in the same 72-slot order |
| `recall_confidence` | recall batch writes finite scope-1 confidence or the finite `-2.0` sentinel directly |
| `recall_scope` | recall batch writes exact scope code `0/1/2` directly |
| `ordered_pair_indices` | the sole dream/lesion traversal writes all 288 canonical pairs directly |
| `pair_check_flags` | the traversal marks exactly the 288 checks it actually completes |
| `pair_reason_codes` | the constrained path writes the registered reason code for every pair; lesion classification cannot overwrite it |
| `endpoint_join_values` | the traversal writes the 24 left/right scalar pairs directly |
| `actual_budget_vector` | event methods increment event rows in place; `snapshot` alone fills measured/limit rows |
| `registered_budget_vector` | initialization copies the 29 registered integers directly once; it is never used to drive work |
| `hard_provenance_leak_counts` | violation-capability boundaries alone increment their fixed rows in place; scientific success requires all 17 zero |
| `inactive_padding` | initialization writes zeros; no scientific or selection operation may read it |

`CoreArenaViewV5`, `CodecArenaViewV5`, and `CostArenaViewV5` must therefore be
non-owning frozen views backed respectively by `learned_core`,
`codec_mean_scale`, and `state_mean_scale`. Their scalar/`*_into` kernels write
to registered destinations or temporary leases. Constructing inherited
`CoreModelV2`, `CodecSpecV2`, or `CostSpecV2` inside a condition interval is
forbidden because their current `_f64` validation copies arrays and
`CoreModelV2.predict` creates result owners. Draft 41's request types must be
changed accordingly before V5 is frozen.

Likewise, `CandidateResultTargetV5` fields are basic-slice views of the seven
named result owners. `CandidateResultV5` validates with `np.shares_memory` and
never invokes the copying V3 `CandidateResultV2` constructor. At sealing,
`hashlib.sha256.update(memoryview(view).cast("B"))` hashes each C-order view
incrementally; `np.ascontiguousarray`, concatenation, and a hash-staging array
are forbidden.

## 3. Exact immutable common-input owner inventory

Common generation occurs once per seed before any condition resource interval,
takes no `ConditionDispatchV5`, and cannot observe a condition name or policy.
It must publish exactly the following 17 NumPy owners. They are shared by
identity across every factorial cell and non-oracle control; no condition may
copy or mutate them. All are C-contiguous, `OWNDATA=True`, `.base is None`, and
`WRITEABLE=False` at common-input seal.

| ordinal | semantic owner | shape | dtype | bytes |
|---:|---|---:|---|---:|
| 0 | `wake_states` | `(96,13,4)` | `float64` | 39,936 |
| 1 | `wake_actions` | `(96,12,2)` | `float64` | 18,432 |
| 2 | `wake_signatures` | `(96,12,4)` | `float64` | 36,864 |
| 3 | `wake_codes` | `(96,9)` | `int16` | 1,728 |
| 4 | `positive_cue_values` | `(24,12,8)` | `float64` | 18,432 |
| 5 | `lure_cue_values` | `(24,12,8)` | `float64` | 18,432 |
| 6 | `cue_masks` | `(24,12,8)` | `bool` | 2,304 |
| 7 | `origin_codes` | `(24,9)` | `int16` | 432 |
| 8 | `candidate_numeric_actions` | `(8,20,2)` | `float64` | 2,560 |
| 9 | `candidate_action_token_codes` | `(24,8,20)` | `int16` | 7,680 |
| 10 | `public_goals` | `(24,20,4)` | `float64` | 15,360 |
| 11 | `common_evaluator_noise` | `(24,20,4)` | `float64` | 15,360 |
| 12 | `true_futures` | `(24,8,20,4)` | `float64` | 122,880 |
| 13 | `generator_valid_sequences` | `(8,)` | `bool` | 8 |
| 14 | `calibration_core` | `(20,)` | `float64` | 160 |
| 15 | `calibration_codec_mean_scale` | `(2,96)` | `float64` | 1,536 |
| 16 | `calibration_state_mean_scale` | `(2,4)` | `float64` | 64 |
|  | **exact common NumPy payload** |  |  | **302,168** |

`wake_codes` columns are, in order,
`context,port,prefix,suffix,sign,context_token_code,prefix_token_code,
suffix_token_code,action_token_code`. `origin_codes` columns are
`context,port,prefix,sign,context_token_code,prefix_token_code,
suffix_token_code,cross_suffix_token_code,target_ledger_index`.

The random opaque strings live in one first-occurrence-ordered Python token
table. The two code arrays index that table. Candidate code receives equality-
and-hash-only `OpaqueTokenV5` references, never the numeric codes, ordering, or
generator structure. Cross-port cues reuse the same positive-value and mask
owners by reference and differ only in their opaque suffix token. Hidden cue
cells use one registered quiet-NaN bit pattern; the candidate-facing cue proxy
cannot return them.

The ordered common-owner record schema is exactly
`semantic_name,shape,dtype,nbytes,raw_sha256,writeable,ordinal`. `raw_sha256` is
SHA-256 of the C-order bytes read through a memoryview after dtype/shape checks.
The canonical compact JSON SHA-256 of the 17 ordered records is
`common_input_manifest_sha256`. At runtime a private map from root `id` to
ordinal proves identity; nondeterministic pointer values are not serialized.
Each condition serializes the exact ordinal vector `0..16` plus the manifest
digest, and the harness asserts that every referenced root is the same object as
the common manifest root. A copied array has a different root and fails even if
its bytes hash equally.

Only these common owners are exempt from the per-condition 393,216-byte ledger
and temporary peak. V5 generator construction transients must be released
before common-input seal and cannot be reachable from the returned common
object. Common generation and later split aggregation are distinct registered
phases; neither may accept a condition policy or allocate an object later
reachable from a candidate except one of the 17 owners or the bounded opaque
token metadata above. This prevents moving condition-local work into the common
phase to evade accounting.

## 4. `TempArenaV5` and closed owner classification

Within each condition resource interval, `TempArenaV5` creates exactly one
temporary root owner:

```text
semantic_name = condition_workspace
shape         = (1048576,)
dtype         = uint8
nbytes        = 1048576
allocation_ordinal = 0
```

It is created once by `np.empty((1048576,),dtype=np.uint8,order="C")` and is
the sole temporary owner. `lease(owner_code,shape,dtype)` returns an aligned
`np.ndarray(...,buffer=condition_workspace,offset=...)` view with
`OWNDATA=False`; it records owner code, dtype, shape, byte interval, lease
ordinal, generation, and release ordinal. Live leases may not overlap. A phase
reset is legal only after all leases in that phase are released, increments the
generation, and makes every earlier handle unusable. Raw lease views cannot be
stored in a result, common object, arena metadata, Python closure, or later
phase. `close_seed` releases the one root only after the live-lease set is empty
and records final live bytes zero.

The measured temporary peak is the maximum `nbytes` of live temporary roots,
not the sum of views. It is therefore exactly 1,048,576 bytes for every complete
condition interval and satisfies the registered cap by equality. Every lease
must be completely initialized before read; an off-range poison test fills the
workspace with two different byte patterns and requires bit-identical sealed
candidate and audit bytes.

At every registered boundary, recursively inspect all reachable NumPy arrays
in arguments, returned records, arena state, facades, and live closures. Follow
each `.base` chain to its root and classify it as exactly one of:

1. one of the current `ConditionArenaV5`'s 36 registered roots;
2. the current live `TempArenaV5.condition_workspace` root; or
3. one of the 17 sealed common roots by identity and raw hash.

Membership in zero or more than one class hard-fails. The walker has cycle
protection, handles dataclass/slot/tuple/list/dict fields, forbids object-dtype
arrays in candidate reachability, and treats Python scalar/container overhead
uniformly as non-NumPy metadata. Result views must classify as class 1;
temporary leases as class 2; candidate-readable common inputs as class 3.

## 5. Registered call graph and allocator closure

The condition path is a closed direct-call graph. No dynamic dispatch,
monkeypatch-selected helper, callback, `eval`, `exec`, alternate runner, or
unlisted callable is allowed. The required high-level edges are:

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

Private scalar/`*_into` kernels used by these nodes must either be included as
separate callable-boundary records or their complete enclosing class/function
source must contain them; an un-hashed imported helper is forbidden. The exact
node/edge/phase table is serialized in registration and its canonical compact
JSON digest is carried through the implementation lock and all later artifacts.

### 5.1 V3/G7-M calls

The V3 module and G7-M V2/V1 modules remain raw-hashed scientific dependencies,
but no V3/G7-M NumPy-producing callable may execute inside a condition resource
interval. In particular these runtime calls are forbidden:

```text
agi_world_memory_integration_v3.scoped_hard_recall_v3
episodic_ltm_dream_bridge_v2.hard_cue_anchored_recall
agi_world_memory_integration_v3.codec_residual_view_v3
agi_world_memory_integration_v3.constrained_residual_completion_v3
agi_world_memory_integration_v3.execute_candidate_v3
agi_world_memory_integration_v3.fit_shared_core_v3
agi_world_memory_integration_v3._generate_seed_world_v3
agi_world_memory_integration_v3._make_store_schema_action
agi_world_memory_integration_v3.build_evaluation_cues_v3
agi_world_memory_integration_v3.planning_cost_v3
agi_world_memory_integration_v3.paired_interval_v3
agi_world_memory_integration_v3.factorial_effects_v3
agi_world_memory_integration_v3.run_agi_world_memory_integration_v3_gate
```

The committed V3/G7-M recall route is not resource-safe: it creates owners via
`np.zeros`, Boolean advanced indexing, `np.stack`, `.copy()`, and vector norm
intermediates, then returns an owning reconstruction. Registering that call
would recreate P0-R2 even if its final reconstruction were copied into an arena.
Therefore `scoped_hard_recall_v5` must reproduce the inherited 12-row masked-
cosine, first-index tie, strict-threshold, fallback/clamp, and physical-identity
semantics using common/arena views and temporary leases. Handcrafted off-range
differential tests must compare every audit scalar and completed byte to the
hashed V3/G7-M reference outside the registered path. This is an implementation
change only, not a scientific rule change.

Draft 41 currently requires
`scoped_hard_recall_v5 -> scoped_hard_recall_v3 -> hard_cue_anchored_recall`.
That edge and the two inherited callable records must be removed from its exact
registered call graph before freeze. Retaining their raw dependency paths and
using them only as an off-range test oracle is allowed.

### 5.2 Forbidden condition-local NumPy owner creators

Except for the 36 calls inside `ConditionArenaV5.__init__` and the single call
inside `TempArenaV5.__init__`, the measured call graph may not invoke an
owner-producing form of:

```text
np.array, np.asarray, np.empty, np.zeros, np.ones, np.full,
np.empty_like, np.zeros_like, np.ones_like, np.full_like,
np.stack, np.concatenate, np.repeat, np.tile, np.from*, np.copy,
ndarray.copy, ndarray.astype, np.linalg.*, fancy/Boolean indexed reads,
reshape/ravel/flatten when memory sharing is not proved, or a ufunc/reduction/
matmul that returns an ndarray rather than writing to a registered destination
or TempArena lease.
```

Basic slices are allowed only when `np.shares_memory(slice,registered_root)` is
true. Scalar indexing, Python `math` operations, in-place operations on a lease,
and NumPy operations with an explicit registered/lease `out=` are allowed.
Boolean cue gathering and schema selection use scalar loops into a lease, not
advanced indexing. The AST test resolves aliases/imports and rejects an unknown
call target; the runtime owner walker is a second, independent check.

## 6. Exact active and shadow work

Every condition/control first fills the identical wake payload, observed schema,
scope indices, locked core/normalizers, and padding owners. LTM availability is
represented only by occupancy/provenance and the sealed dispatch policy; payload
capacity and common inputs remain equal.

For each of 24 positives, 24 lures, and 24 cross-port diagnostics, the recall
slot counter increments once. Cross-port invalidity is resolved before store
availability, returns scope 2, and performs no distance row. Each positive and
lure executes 12 masked-cosine rows. With LTM enabled, the winner/threshold may
affect only the positive completed view or diagnostic audit. With LTM disabled,
the same 12 rows are evaluated against the condition's unqueryable trace payload
in TempArena leases, every scalar result is discarded, and the external result
is the scope-0 sentinel. Thus `48*12=576` is observed work in both routes; no
shadow value reaches q, rollout, recall audit, or selection.

Dream and lesion auditing share exactly one canonical 288-pair traversal. For
all conditions it writes pair indices/checks/reasons, evaluates 288 component
checks, visits 72 same-component pairs, recognizes 48 observed keys, constructs
24 valid-missing proposals, writes 48 endpoint scalars and 24 proposal slots,
and executes 24 update decisions. With dream writing enabled, only a threshold-
accepted proposal updates the registered schema; otherwise the decision is
shadow-only. The same traversal classifies all 240 nonobserved pairs into the
first 24 lesion slots and 216 capacity-padding decisions. No second enumeration,
join, endpoint evaluation, proposal pass, or dream-pass increment is allowed.
Only the unconstrained-lesion report exposes the already frozen 3/21 audit; the
lesion buffer is unreachable from schema/action/candidate objects.

## 7. Hard provenance counts and capability ownership

`hard_provenance_leak_counts` uses the inherited exact 17-name order. It starts
as the zero-filled persistent owner and is writable only through fixed violation
methods owned by the stage/arena capability. Each method has one compile-time
row and increments exactly once before denying or failing the attempted action:

| hard row(s) | instrumented boundary |
|---|---|
| synthetic identity/tag/insert rows 0--4 | synthetic binding validator and real-only LTM insertion facade; an actual successful synthetic insert additionally increments row 4 before terminal failure |
| `nonledger_real_record_in_ltm` | physical ledger-index/hash validator at real insertion |
| `observed_record_overwrite_or_hash_change` | schema observed-slot write guard and pre/post observed hash check |
| splice/constraint rows 7--8 | dream acceptance validator before occupancy/provenance write |
| `accepted_cross_context_recall` | recall identity/scope validator before accepted completion write |
| truth-read rows 10--12 | evaluator-only handle accessors; no such handle/type is reachable from candidate code |
| `masked_cue_coordinate_reads` | `MaskedCueViewV5.visible_value(row,col)`, which increments and raises when mask is false; no raw cue-value owner is reachable from the request |
| `test_path_reads_before_unlock` | artifact I/O capability before any test-path open/read/hash |
| `cell_cross_write_or_shared_mutation` | arena owner-token check on every write handle and common-input write denial |
| `nonfinite_outputs` | scalar write/seal boundary before a nonfinite value can enter a scientific output |

Zero is established by absence of violation calls, not by overwriting the vector
at the end. Handcrafted tests must deliberately trigger each denial boundary and
observe the corresponding row increment in an off-range arena. They must also
show that ordinary off-range execution leaves all 17 rows zero.

Draft 41's raw `PartialCue` field cannot instrument a masked-coordinate read and
must be replaced by the non-owning `MaskedCueViewV5` proxy (or an exactly
equivalent proxy with no raw values attribute). Its `CoreModelV2`/`CodecSpecV2`/
`CostSpecV2` request fields must likewise become the arena-backed views defined
in section 2. These four view/proxy class sources must be included in the
callable manifest. Without these repairs, the V5 API lane still has an open
resource/leakage P0.

## 8. Resource evidence and exact replay

Each complete condition serializes one `ResourceEvidenceV5` with exactly these
top-level fields, in order for schema validation (canonical JSON still sorts
object keys for bytes):

```text
condition_code
budget_row_evidence
actual_budget_vector_sha256
registered_budget_vector_sha256
event_trace_sha256
condition_owner_manifest_sha256
condition_owner_total_bytes
common_input_manifest_sha256
common_reference_ordinals
temp_owner_record
temporary_peak_owned_bytes
temporary_final_live_owned_bytes
temp_lease_log_sha256
hard_provenance_leak_counts_sha256
metadata_utf8_bytes
environment_evidence
resource_checks
```

`budget_row_evidence` is an ordered 29-record array with exact fields
`name,value,source_class,source_ref,formula_id`. The source reference is a fixed
event row code, owner name, or JSON pointer. The two vector hashes cover the raw
little-endian `int64[29]` arena bytes. `condition_owner_manifest_sha256` covers
the ordered 36 records `name,shape,dtype,nbytes,raw_sha256,ordinal`, and its byte
sum must be 393,216. `temp_owner_record` contains exactly
`semantic_name,shape,dtype,nbytes,allocation_ordinal,release_ordinal`; peak is
1,048,576 and final live bytes is zero. `environment_evidence` contains the
observed CPU/NumPy/process/network/GPU/external-file/wall fields needed to replay
the inherited 12 resource checks.

`metadata_utf8_bytes` is the length of canonical compact UTF-8 JSON for the
condition's token dictionary, scalar recall/dream/lesion audits, owner/counter
evidence, and control-independent metadata, excluding numeric array payloads
already charged by `nbytes`. The measured bytes must be at most 32,768. The
measurement cannot omit a field merely because its value is unfavorable.

The resource replayer ignores stored conjunction Booleans. It rehashes current
registration/allocation/call-graph bytes, rechecks all 36 owner records, all 17
common identities, every 29-row formula, all 17 hard counts, temporary closure,
metadata bytes, and environment observations, then constructs the inherited
exact 12-key `resource_checks` object. `budget_vector_exact` is true only when
all 29 recomputed values equal their registered values. `resource_passed` is
`all(value is True for value in resource_checks.values())` after exact-keyset
validation.

## 9. Required zero-seed/off-range tests

Before an implementation lock, the two registered V5 test files must exercise
all of the following without a registered/development scientific seed:

1. recompute the three disjoint 29-row source-class sets and every formula;
2. reject arbitrary event deltas, owner/limit row events, direct vector writes,
   and expected-vector assignment as evidence;
3. allocate all 36 persistent owners, verify exact order/dtype/shape/bytes/total,
   and prove pairwise non-aliasing across simultaneously live condition arenas;
4. verify every result field shares memory with its assigned arena owner and
   that importing/constructing V3 `CandidateResultV2`, `CoreModelV2`,
   `CodecSpecV2`, or arena-unaware scientific functions on the measured path is
   rejected;
5. freeze the 17 common owners, verify the exact 302,168-byte inventory, pass
   the identical roots to every condition, and make an equal-byte copy fail;
6. walk the complete candidate request graph and prove no truth owner, raw
   masked cue values, arena/counter object, condition code, or lesion buffer is
   reachable;
7. trigger each of the 17 capability violations independently and verify its
   one-row counter mapping; ordinary execution leaves every row zero;
8. statically traverse the complete registered call graph, resolve every call,
   reject all forbidden V3/G7-M runtime edges and owner allocators, and compare
   the graph digest to registration;
9. run V5 recall/dream/rollout kernels on handcrafted fixtures against the
   hashed inherited reference and require equal audits, keys/sources, selected
   index, and candidate payload bytes; the inherited reference is test-only;
10. verify active/disabled LTM both execute 72 slots and 576 distance rows,
    active/shadow dream both execute `288/288/72/24/48/24/24/240/24/216/1`, and
    shadow values cannot change candidate bytes;
11. exhaust TempArena allocation, overlapping lease, use-after-release,
    double-release, escaped-view, uninitialized-read poison, extra-owner, and
    nonempty-close failures; require peak 1,048,576 and final live zero;
12. inject owning results from `np.stack`, `np.concatenate`, Boolean indexing,
    `.copy()`, and `np.linalg` into each boundary and require owner
    classification failure;
13. replay `ResourceEvidenceV5` after independently changing each row, owner
    hash, common ordinal, cap, peak, metadata byte count, environment field, and
    resource-check key; every mutation must fail;
14. prove all 24 permutations of the four factorial cell execution order yield
    bit-identical per-cell scientific and resource evidence.

The implementation lock must store exact collected/passed/failed/error/skipped/
xfailed/xpassed/deselected counts and raw stdout/stderr hashes for these tests;
a literal success Boolean is not evidence.

## 10. Closure and no-go boundary

[Definition] The persistent owner set, common exemption, temporary owner, call
graph, source class of every budget row, counter call site, shadow operation,
and evidence replay above are finite and exhaustive.

[Derivation] Under these boundaries, the final three budget entries are no
longer misrepresented as operation counts; only event rows can be incremented;
all persistent result/core/schema/audit bytes are actual destinations; every
condition-local NumPy root is classified; common owners have a closed identity
manifest; and arena-unaware V3/G7-M functions cannot hide owners in the measured
path. This removes P0-R1 and P0-R2 without changing a scientific factor or
threshold.

Gate recommendation: PASS for this resource/call-graph lane only after draft
41 removes its inherited runtime recall edge and replaces its copying/raw
request fields as specified above, the exact rules are incorporated into the V5
amendment/registration, and an independent resource/call-graph audit reports no
open P0. Until then the combined V5 preregistration gate remains REVISE.

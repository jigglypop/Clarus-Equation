# G9-CBM V2 implementation blueprint

Status: COMPLETE

## 0. Scope and inspection result

This is a read-only implementation reconnaissance result for
`revisions/00-contract-v2-draft.md`. No registered train, validation, test, or
development seed was generated or scored while preparing it. No runner was
executed.

The smallest safe implementation is a **standalone V2 module** with three
explicit reuse boundaries:

1. Import and byte-lock G7-M V2's `EpisodicRecord`, `PartialCue`,
   `PersistentEpisodicStore`, and `hard_cue_anchored_recall` together with the
   V1 provenance and component semantics on which they depend.
2. Reuse the G9-CB V5 discipline of one recursive H20 rollout, an H5 view made
   only by slicing that array, and a predictor API that cannot receive an
   episode future or hidden state. Do **not** inherit the G9-CB V4/V5 generator,
   parent artifact, residual filter, or `BridgeModel`: their state equation,
   data roles, and single-origin protocol do not match this contract.
3. Implement the action-conditioned generator, 20-parameter core, scoped recall
   facade, 72-slot residual schema, residual-only dream pass, fixed action set,
   planner, evaluator seal, and exact common-budget ledger in the new module.
   In particular, do not call G7-M's eight-coordinate dream join: fingerprint
   columns `4:8` would enter its endpoint RMS and violate V2.

The existing runner pattern should be retained: the file under `examples/agi`
is a two-line import of `main` plus `SystemExit(main())`. Scientific logic and
state transitions stay in the library module.

Three serialization details still need an explicit contract sentence before
the scientific lock. The recommended resolutions are:

1. Freeze `schema_sources` codes as `observed=0`, `synthetic=1`, `fallback=2`,
   and `unresolved=255` (`uint8`).
2. Freeze 72-key integer order by learner-visible first occurrence: context,
   then component, then prefix, then suffix. Do not sort randomized opaque
   string spellings.
3. Treat `test_lock.validation_artifact_sha256` inside the test report as the
   unlock record and require the validation bytes to be tracked at `HEAD`
   before test entry. This satisfies the current exact path list without
   inventing a sixth artifact. If a separate unlock file is desired instead,
   it must be added to the contract and integrity file set before train.

## 1. Files and dependency boundary

Create exactly these scientific source files before any registered seed is
opened:

```text
experiments/preregistration/agi_world_memory_integration_v2.json
reality_stone/python/reality_stone/clarus/agi_world_memory_integration_v2.py
examples/agi/agi_world_memory_integration_v2_gate.py
tests/test_agi_world_memory_integration_v2.py
tests/test_agi_world_memory_integration_integrity_v2.py
```

The implementation lock must name the raw SHA-256 of all five. It must also
name and verify these inherited dependencies before calibration and again
immediately before validation or test world generation:

```text
experiments/preregistration/episodic_ltm_dream_factorial_v2.json
reality_stone/python/reality_stone/clarus/episodic_ltm_dream_bridge_v2.py
reality_stone/python/reality_stone/clarus/episodic_ltm_dream_bridge.py
```

Locking the V1 module is necessary because the V2 store, record types,
standardizer, co-occurrence components, and provenance aliases delegate to it.
The new module must import those dependencies under explicit aliases, for
example `g7m_v2` and `g7m_v1`; copying their source into the new module would
destroy the dependency proof.

The implementation lock should additionally carry normalized source hashes for
the exact callable boundaries below, even though the whole module is already
locked:

```text
TrajectoryCodecV2.encode
scoped_hard_recall_v2
ResidualSchemaTableV2.lookup_residual_raw
residual_only_constrained_dream_v2
rollout_action_candidates_v2
score_action_candidates_v2
```

Use a single deterministic callable-source recipe: UTF-8 `inspect.getsource`,
normalize CRLF to LF, require one terminal LF, then SHA-256. Store the recipe
identifier in the lock so the symbol hashes are reproducible.

## 2. Preregistration JSON structure

Use one standalone, fully materialized JSON document. Do not use the recursive
`overrides` merge used by `sparse_causal_bridge_v4/v5`; that pattern is useful
for a parent-preserving extension, whereas G9-CBM V2 has a new generator and a
fresh train calibration. The top-level identity is:

```text
schema_version = 2
status = "locked_pre_implementation"
experiment = "agi_world_memory_integration_v2"
roadmap_stage = "G9-CBM/V2"
runner = "agi_world_memory_integration_v2"
standalone = true
extends = null
```

Materialize seed arrays as JSON integers, not range strings:

```text
data_roles.train.seeds      = 86100..86139
data_roles.validation.seeds = 87100..87139
data_roles.test.seeds       = 88100..88159
```

Unit tests should use handcrafted fixtures wherever possible. If an RNG fixture
is indispensable, record a small `data_roles.unit_fixture_seeds` list far
outside every seed in all preregistration files and make the test suite scan all
preregistrations to prove disjointness. The unit suite must install an autouse
guard around `_generate_seed_world_v2` that raises before any registered seed
is dispatched.

The JSON should have the following non-overlapping top-level sections. A value
should have one authoritative location and other sections should refer to its
field path rather than duplicate a conflicting literal.

| field | required contents |
|---|---|
| `preregistration_integrity` | all V2 seed roles unopened; no V2 pilot; registered results cannot amend the contract; implementation must precede train |
| `historical_boundary` | V1 formal `BLOCKED`, no V1 seed opened; G7-M V2 PASS is hard exemplar evidence only; G9-CB V5--V7 validation failures remain counterevidence |
| `claim_boundary` | exact conditional software claim plus the forbidden biological, open-world, consciousness, dream/sleep, and AGI claims |
| `data_roles` | the three exact arrays, role descriptions, and unit-seed prohibition |
| `spaces_and_tokens` | `d=4`, `m=2`, `H=20`, contexts 2, ports 4, locals 3, opaque-token equality-only rule |
| `bindings` | ordered `O=[[0,0],[0,1],[1,1],[1,2],[2,2],[2,0]]`, ordered `M=[[0,2],[1,0],[2,1]]`, and invalid composition rules |
| `generator` | literal `D`, `B`, `G`; residual phase rule; all 15 stream IDs; primitive scales; antithetic wake rule; common evaluation-noise coupling; state bound |
| `learned_core` | exact five-column design, source map `[3,0,1,2]`, ridge `1e-6`, unpenalized intercept, ordered 20 parameters, four solves, no clipping/refit |
| `wake_ledger` | 96 records, 13 states, 12 actions/signatures, canonical `c,p,i,j,h` order, immutable real IDs, 73,728 trace bytes |
| `codec` | raw residual/signature `12x8`, residual columns `0:4`, fingerprint columns `4:8`, population standardizer floor `1e-8`, raw/internal-standardized boundary |
| `evaluation_origins_and_cues` | 24 lexicographic origins; exact 12-transition anchor prefix; phase-zero start; 24-cell mask split `10/4/10`; noise; 24 lures and 24 cross-port diagnostics |
| `episodic_ltm` | physical bank 96, eight 12-row zero-copy scopes, strict threshold, storage-order tie, fallback semantics, 72 total call slots |
| `slow_schema` | 72 preallocated keys, 48 observed entries, standardized storage, inverse transform before rollout, raw residual decode only |
| `dream` | one immutable-snapshot pass; 288 enumerations/checks, 72 same-component, 48 observed rejection, 24 missing joins, 48 endpoint values, at most 24 writes; residual-only RMS |
| `actions_and_planning` | literal `A0..A2`; exact eight sequences and lure leads; wake-only action index; goal formula; normalizer floor `0.05`; cost, penalty `10000`, success `25`, tie rule |
| `candidate_capability` | exact request/result schemas, chronology, forbidden inputs, result-hash-before-unseal rule, no token emissions |
| `factorial_design` | exact M00/M10/M01/M11 resource differences and all other byte-equality requirements |
| `controls` | persistence, frozen core, schema-only, shuffled binding, zero-q, unconstrained lesion, zero-synthetic, and post-hash diagnostic oracle |
| `metrics` | `E_all`, `E_uv`, recall, regret, success, three separate invalidity rates, exact scalar denominators and nonfinite behavior |
| `paired_inference` | factorial signs, ratio-of-means rule, `ddof=1`, exact n40/n60 t values, strict wins and ties |
| `all_of_gate` | every prediction, planning, recall, dream, attribution, no-antagonism, stability, provenance, leakage, and resource threshold from the contract |
| `calibration` | only core, state/codec standardizers, scoped recall threshold, residual join threshold; exact population counts and unique selection rules |
| `provenance_and_leakage` | the four semantic tuples, all 17 hard-zero counters, poison/cell-order/permutation tests, pre-unlock denial |
| `resources` | exact registered operation vector, ordered 393,216-byte ledger, 524,288 persistent cap, 1,048,576 temporary cap, 32,768 metadata cap, CPU/NumPy/no-network rules |
| `test_lock` | exact five artifact paths, source/dependency hashes, run-once state transitions, validation unlock requirements, byte-identical calibration reuse |
| `failure_rules` | stop-before-train, stop-before-validation, validation FAIL/test unopened, no overwrite/retune/reseed/drop/epsilon rules |
| `dimensionless_audit` | raw and standardized variables are dimensionless representations; normalized cost/join/error; dimensional consistency is not empirical evidence |
| `literature_safety` | motivation-only citations and prohibited biological extrapolations |

Within `generator`, store matrices as row-major nested arrays and freeze this
exact stream map:

```text
token:0, initial_state:1, prefix_primitive:2, connector:3,
suffix_primitive:4, irreducible_interaction:5, episode_drift:6,
entity_fingerprint:7, wake_noise:8, evaluation_prefix:9,
evaluation_noise:10, cue_mask:11, cue_noise:12,
lure_fingerprint:13, presentation_order_test:14
```

Within `test_lock`, use the exact artifact paths from the contract and include
these Boolean requirements:

```text
open_test_only_after_validation_all_of_pass = true
implementation_lock_before_train = true
train_calibration_written_once = true
validation_written_once = true
test_written_once = true
require_identical_registration_sha256 = true
require_identical_implementation_lock_sha256 = true
require_identical_train_calibration_sha256 = true
require_identical_source_and_dependency_sha256 = true
early_test_read_is_hard_invalid = true
failed_validation_must_be_preserved = true
```

## 3. Module data model and public API

### 3.1 Imported G7-M surface

Expose explicit aliases only for the inherited interface being tested:

```python
Provenance = g7m_v1.Provenance
EpisodicRecord = g7m_v2.EpisodicRecord
PartialCue = g7m_v2.PartialCue
RecallResult = g7m_v2.RecallResult
PersistentEpisodicStore = g7m_v2.PersistentEpisodicStore
CoordinateStandardizer = g7m_v2.CoordinateStandardizer
REAL_PROVENANCE = g7m_v2.REAL_PROVENANCE
RECALLED_PROVENANCE = g7m_v2.RECALLED_PROVENANCE
SYNTHETIC_PROVENANCE = g7m_v2.SYNTHETIC_PROVENANCE
FALLBACK_PROVENANCE = g7m_v2.FALLBACK_PROVENANCE
```

Do not alias `SlowSchemaTable`, `constrained_missing_binding_dream`, or
`update_missing_slow_binding` as active V2 algorithms. Their semantics may be
hash-checked, but V2 requires a separately typed residual-only adapter.

### 3.2 Frozen public dataclasses

Keep public dataclasses small, frozen, and shape-validated in `__post_init__`.
Arrays received by a candidate must be copied once into their registered owner,
made C-contiguous, and marked read-only.

```python
@dataclass(frozen=True)
class CoreModelV2:
    intercept: np.ndarray       # (4,)
    diagonal: np.ndarray        # (4,)
    bridge: np.ndarray          # (4,), source order (3,0,1,2)
    action: np.ndarray          # (4,2)

    def predict(self, x: np.ndarray, a: np.ndarray) -> np.ndarray: ...

@dataclass(frozen=True)
class StateNormalizerV2:
    mean: np.ndarray            # (4,)
    scale: np.ndarray           # (4,), each >= 0.05

@dataclass(frozen=True)
class TrainCalibrationV2:
    core: CoreModelV2
    state_normalizer: StateNormalizerV2
    codec_standardizer: CoordinateStandardizer
    recall_threshold: float
    join_threshold: float
    registration_sha256: str
    implementation_lock_sha256: str
    sha256: str

@dataclass(frozen=True)
class CandidateRequestV2:
    cue: PartialCue                         # raw (12,8), 24 visible cells
    anchor_state: np.ndarray                # (4,)
    numeric_actions: np.ndarray             # (8,20,2)
    opaque_action_tokens: np.ndarray        # (8,20), immutable strings
    public_goal: np.ndarray                 # (20,4)
    core: CoreModelV2
    state_normalizer: StateNormalizerV2
    codec_standardizer: CoordinateStandardizer
    action_index: "WakeActionIndexV2"
    schema: "ResidualSchemaTableV2"
    episodic_store: "ScopedEpisodicFacadeV2 | None"
    action_cost_weight: float               # exactly 0.02
    invalid_penalty: float                  # exactly 10000

@dataclass(frozen=True)
class CandidateResultV2:
    predictions: np.ndarray                 # (8,20,4), float64
    inferred_valid: np.ndarray              # (8,20), bool
    resolved_schema_keys: np.ndarray        # (8,20), int16
    schema_sources: np.ndarray              # (8,20), uint8
    inferred_costs: np.ndarray              # (8,), float64
    selected_index: np.int64
    recall_audit: "RecallAuditV2"           # fixed-shape only
    provenance_audit: "ProvenanceAuditV2"   # fixed-shape only
```

The request contains no cell label. M00/M10/M01/M11 differences are expressed
only by the content/capability of `episodic_store` and hypothetical occupancy in
`schema`. The result contains no emitted token and no evaluator truth.

`WakeActionIndexV2` is an immutable mapping built only from real wake records:

```text
opaque action token ->
  (context_token, inferred_component_id, exact numeric action, suffix_token,
   resolved schema-local suffix index)
```

Construction must hard-fail on zero or multiple mappings. It must not retain a
public port integer or generator-validity label.

### 3.3 Public functions

The test-visible public surface should be limited to:

```python
def fit_shared_core_v2(wake_records: Sequence[WakeRecordV2]) -> CoreModelV2: ...

def scoped_hard_recall_v2(
    store: PersistentEpisodicStore,
    cue: PartialCue,
    scope_index: "ScopedRecallIndexV2",
    *,
    audit: "RecallAuditV2",
) -> RecallResult: ...

def residual_only_constrained_dream_v2(
    real_snapshot: Sequence[WakeRecordV2],
    schema: "ResidualSchemaTableV2",
    join_threshold: float,
    *,
    write_enabled: bool,
    audit: "DreamAuditV2",
) -> tuple["ResidualDreamBindingV2", ...]: ...

def execute_candidate_v2(request: CandidateRequestV2) -> CandidateResultV2: ...

def evaluate_factorial_seed_v2(
    master_seed: int,
    calibration: TrainCalibrationV2,
    registration: Mapping[str, object],
) -> dict[str, object]: ...

def prepare_implementation_lock_v2(config_path: Path) -> dict[str, object]: ...
def run_train_calibration_v2(config_path: Path) -> dict[str, object]: ...
def run_agi_world_memory_integration_v2_gate(
    config_path: Path, *, split: str = "validation"
) -> dict[str, object]: ...
def verify_artifact_chain_v2(config_path: Path) -> dict[str, object]: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

`execute_candidate_v2` must have exactly one argument. Signature tests should
reject names containing seed, split, world, episode, future, truth, latent,
validity, outcome, reward, cost-realized, target ID, or evaluator handle. The
runner signature follows the existing G7-M V2 pattern, while artifact-writing
state transitions are explicit in the CLI.

### 3.4 Scoped recall implementation

`hard_cue_anchored_recall` first checks component validity but then scores every
record in the supplied store. Therefore the new wrapper must:

1. Reject an invalid/cross-component cue before constructing a view or calling
   a distance routine.
2. Resolve one prebuilt `int16[12]` scope in frozen first-storage order.
3. Construct a read-only facade whose `.records` are references to exactly those
   12 existing records and whose standardizer/threshold are the frozen ones.
   It must not copy trace payload bytes or call `insert_real`.
4. Call `g7m_v2.hard_cue_anchored_recall(facade, cue)` exactly once.
5. Convert an accepted result to the same raw-codec reconstruction already
   returned by G7-M. On rejection, discard its all-zero reconstruction and use
   the registered raw-mean fallback path; the zeros must never enter `q_hat`.

The wrapper must be bit-equivalent to direct inherited recall when the supplied
bank already contains only the same 12 records. A mixed-bank test must show that
no record outside context/component can be returned.

### 3.5 New residual schema adapter

`ResidualSchemaTableV2` owns fixed arrays in the condition ledger, not a Python
dictionary that grows after initialization. Its canonical key index follows
learner-visible first-occurrence order for context, inferred component/port,
prefix token, and suffix token; randomized opaque string spellings are never
sorted. It stores:

```text
payload             (72,12,8) float64 standardized
observed_occupancy  (72) bool
synthetic_occupancy (72) bool
provenance_code     (72) uint8
```

Columns `4:8` are standardized zero for every schema entry. Every mean, join,
fallback, decode, and rollout reads only `0:4`. `lookup_residual_raw` performs
the inverse transform and returns a `(12,4)` raw residual template plus key and
source codes. Observed keys are immutable.

The dream function enumerates exactly `2*12*12=288` context-local prefix/suffix
pairs in frozen first-occurrence order, records 288 component/port checks,
reaches 72 same-component pairs, rejects 48 observed keys, computes two
four-coordinate endpoint RMS values for 24 missing keys, and owns 24 output and
24 update slots whether or not the threshold accepts them. `write_enabled=False`
executes the same arrays/counters but discards the write.

The unconstrained lesion uses the same enumeration order, excludes the 48
observed keys, copies the first 24 of 240 remaining objects into its separate
audit buffer, marks the remaining 216 `capacity_padding`, and never calls the
valid schema update method or action-index builder.

## 4. Internal pipeline

Keep generator and evaluator-only types private. A practical internal split is:

```text
_rng / _normalize_local_family
_generate_primitives_v2
_generate_wake_record_v2
_generate_evaluation_origin_v2
_generate_seed_world_v2
_fit_codec_and_state_normalizers_v2
_calibrate_scoped_recall_threshold_v2
_calibrate_residual_join_threshold_v2
_build_condition_ledger_v2
_build_candidate_request_v2
_hash_candidate_result_v2
_unseal_and_score_v2
_aggregate_split_v2 / _build_all_of_checks_v2
```

`_GeneratedSeedV2` should separate `wake_records`, public prefixes/actions/goals,
and a private `_EvaluatorOnlyHandleV2`. The handle owns true future arrays, true
schema/q, generator-validity labels, true costs, stream metadata, and outcomes.
It is created as a local value, never a module global, and is not accepted by
any candidate function. All non-oracle candidate results are converted to
canonical bytes and hashed before `_unseal_and_score_v2` receives the handle.

For every origin, generate the prefix as an independent complete 12-transition
episode using `A[i]`, anchor `(i,i)`, the selected stored entity's same `q`, a
fresh initial state, and fresh prefix innovations. Store states `x[0:13]`; pass
only the codec cue and `x[12]`; start every rollout at phase zero. The evaluator
future is a separate allocation.

`TrajectoryCodecV2.encode` uses the full five-coefficient fitted expression:

```text
r_hat[t] = x[t+1] - core.predict(x[t], a[t])
T_raw[t,0:4] = r_hat[t]
T_raw[t,4:8] = signature[t]
```

Recall standardizes internally and returns raw coordinates. The drift estimate
is exactly the mean of completed raw residual columns minus the inverse-
standardized observed anchor schema. Never subtract a standardized template
from a raw reconstruction.

`rollout_action_candidates_v2` fills the single ledger-owned `(8,20,4)` array.
For each next row it calls the 20-parameter core, adds `q_hat`, and adds the raw
schema row. The previous prediction is the next input. It always executes all
160 candidate steps, including incompatible-token lures. `H5` is only
`predictions[:, :5]`; there is no second rollout call.

The candidate uses its wake-only action graph to produce inferred-valid flags,
resolved key indices, and source codes. It assigns `10000` to an incompatible
token; otherwise it computes the registered trajectory/action cost from its
prediction and public goal. It selects by `(cost, canonical_index)`. The
evaluator separately uses generator truth and true futures to compute regret,
success, and invalid-selected metrics after hashes are frozen.

## 5. Common condition allocation

Implement one `ConditionLedgerV2.allocate()` factory. It must allocate every
array in the contract's exact ordered table, assert each `shape`, `dtype`,
`C_CONTIGUOUS`, and `nbytes`, and assert the ordered total is exactly `393216`.
The ledger exposes views to algorithms; helpers may not replace owned arrays.
The fixed padding array closes the subtotal and is allocated in all factorial
cells and every non-oracle control. The persistent cap is `524288`.

Use the same factory independently for:

```text
M00, M10, M01, M11,
persistence, frozen_core, schema_only, shuffled_binding, zero_q,
unconstrained_lesion, zero_synthetic
```

The diagnostic oracle runs after candidate hashes and is explicitly outside
this equality set. Immutable generator inputs may be shared by read-only
reference; any condition-local copy is charged to that condition. Serialize
the ordered allocation SHA, every array byte count, operation vector, temporary
workspace maximum, and metadata byte count per condition.

The exact registered operation vector and the ledger byte table should be read
from preregistration and compared against module constants at load time. If
either differs, stop before train; do not silently prefer one source.

## 6. Artifact state machine

Use canonical UTF-8 JSON with sorted keys, two-space indentation, no NaN, and
exactly one terminal LF. `_locked_json` must reject BOM, CR, missing/double LF,
noncanonical key/indent bytes, and nonfinite values. Artifact creation should
write a sibling temporary file and atomically replace only a previously absent
target; an existing scientific artifact is never overwritten.

The CLI should expose explicit stages:

```text
--stage implementation-lock
--stage calibration
--stage validation
--stage test
--stage integrity
--stage verify
```

The transition table is:

| stage | preconditions | registered execution | output / postcondition |
|---|---|---:|---|
| preregistration | V1 remains blocked, no V2 artifact | 0 | raw JSON SHA copied into module/test constants |
| implementation lock | source/tests/dependencies frozen; handcrafted integrity tests pass; scientific artifacts absent | 0 | `agi_world_memory_integration_implementation_lock_v2.json`, written once |
| calibration | exact implementation lock exists and matches; calibration/validation/test absent | train 86100..86139 once | calibration artifact with frozen core, normalizers, thresholds, counts, hashes |
| validation | calibration raw bytes and every lock match; validation/test absent | validation 87100..87139 once | validation artifact, PASS or FAIL preserved |
| test | committed validation raw bytes are canonical, self-consistent, all-of PASS, and name matching calibration/source locks; test absent | test 88100..88159 once | test artifact names validation raw SHA and reuses calibration bytes exactly |
| integrity | no scientific execution; inspect current byte chain and repository provenance | 0 | integrity ledger for validation-FAIL/test-unopened, or complete validation/test chain |
| verify | read-only hashes and self-consistency only | 0 | no file changes |

Calibration may not be generated implicitly as a side effect of validation.
Likewise, a second invocation must verify or refuse; it must never rerun a seed
because an output file happens to contain matching bytes. Deletion of an
artifact is not authorization to repeat a registered stage.

The implementation-lock payload minimally contains:

```text
schema_version, experiment, status="implementation_locked_pre_train",
registration_sha256,
implementation_sha256 {module, runner, both test files},
inherited_dependency_sha256,
callable_source_sha256 and recipe,
registered_seed_executed_count=0,
ordered_budget_ledger_sha256,
handcrafted_integrity_recipe/version
```

The calibration payload minimally contains:

```text
schema_version, experiment, status="train_calibration_frozen",
source_split="train_only", train_seed_count=40,
registration_sha256, implementation_lock_artifact_sha256,
implementation_sha256, inherited_dependency_sha256,
core coefficients/vector SHA,
mu_x, sigma_x, codec_mu, codec_sigma,
recall_threshold, join_threshold,
selector pools/ties and exact population counts,
ordered budget ledger SHA
```

The validation and test reports minimally contain:

```text
experiment, roadmap_stage, split, seed_count,
registration_sha256, implementation_lock_artifact_sha256,
implementation_sha256, inherited_dependency_sha256,
train_calibration_sha256, core_sha256, budget_ledger_sha256,
environment, seed_results, cell_means, control_means,
factorial_effects, attribution_effects, planning_summary,
all denominator/counter summaries,
checks, hard_zero_checks, resource_checks,
performance_passed, integrity_passed, resource_passed, passed,
test_lock {test_opened_after_validation_pass, validation_artifact_sha256}
```

`passed` must be recomputable as the conjunction of the three component flags,
and each component flag must be recomputable from its complete Boolean mapping.
The test unlock validator must reject an incomplete or self-inconsistent PASS.

The integrity ledger follows the G7-M V2 pattern: raw hash, size, LF/BOM
contract, first locked commit, current repository commit, lock-chain hashes,
exact outcomes and seed counts, and any later audit-only test transition with
both source hashes and explicit assertions that model/gates/artifacts did not
change. It does not execute a registered seed or regenerate an artifact.

## 7. Unit-test groups

`tests/test_agi_world_memory_integration_v2.py` should contain these groups.
Every test uses handcrafted arrays or explicitly off-range fixture seeds; an
autouse monkeypatch rejects all `86100..86139`, `87100..87139`, and
`88100..88159` generation.

### A. Raw registration and roles

- Exact preregistration SHA, canonical LF, identity/status/standalone fields.
- Exact three seed arrays, pairwise disjointness, and disjointness from all
  prior G7-M/G9-CB/V1 ranges.
- Exact D/B/G, stream map, O/M order, action sequences, thresholds, gates,
  denominators, budget vector, and artifact paths.
- No registered seed is used by any unit fixture or prelock proof.

### B. Imported dependency and scoped recall boundary

- Exact inherited raw hashes and alias identities.
- Store rejects synthetic insertion; attempt and success counters are separate.
- Scoped facade has 12 storage-order-preserving references and zero trace copy.
- Scoped wrapper equals inherited hard recall on an already scoped bank.
- Mixed-bank/cross-context/cross-component queries cannot return an outside
  identity and invalid scope rejects before a distance call.
- Strict `confidence > tau`, first-index tie, visible clamp, masked poison, raw
  reconstruction, and rejection-to-raw-mean fallback behavior.

### C. Generator, core, and codec

- Stream isolation and deterministic draw order on handcrafted/off-range cases.
- Fingerprint/drift independence; antithetic wake q/noise; 96 records and exact
  canonical order; common evaluation noise across cells/candidates.
- The exact evaluation prefix has 12 transitions, correct `A[i]`, anchor key,
  same q, fresh noise, and phase-zero next lead.
- Ridge design retains four intercepts and produces ordered 20 parameters;
  expected train row count is 46,080.
- Codec subtracts the full core including intercept/action, is raw at its API,
  standardizes internally only, and never mixes fingerprint columns into q.
- Generator and valid predictions enforce `max(abs(x))<=2`.

### D. Residual schema and dream/lesion accounting

- Fixed 72-slot table, 48 observed and 24 empty valid keys, immutable observed
  hash, and standardized-zero fingerprint columns.
- Raw inverse decode reads only `0:4`; poisoning `4:8` cannot change joins,
  accepted bindings, schema outputs, or rollout.
- Exact 288/288/72/48/24/48/24/24 counters and rejection reasons.
- Synthetic provenance has null episode ID and never becomes observed/recalled
  or enters LTM/action graph.
- Shadow pass has identical calls/arrays; M01 snapshot is nonqueryable/destroyed
  at evaluation.
- Unconstrained lesion deterministically selects 24 of 240, pads 216, and
  cannot create a legal schema edge.

### E. Candidate API, chronology, and leakage

- Exact dataclass field and function signatures; forbidden names/types absent.
- Poison future states, true q/schema, validity, outcomes/costs, seeds/streams,
  and masked cue cells independently; candidate bytes and selection remain
  bit-identical while evaluator scores may change.
- All 24 factorial cell execution orders produce identical per-cell bytes.
- Input/core/observed-schema hashes are equal across cells; no shared mutation.
- Candidate results are hashed before evaluator unseal.
- Pre-unlock test-path reads are denied; candidate cannot import or receive the
  evaluator handle.

### F. Action rollout and planning

- All eight candidates execute exactly 20 steps; incompatible token lures still
  fill finite padded predictions.
- H5 is bit-exact `H20[:5]` from one call; no true state is reread.
- Wake-only token index validates context/component/numeric action/schema key;
  same-valued wrong-port/context tokens fail despite byte-identical numeric
  actions.
- Public goal is computed independently with literal D/B/G and q=b=noise=0.
- Valid cost bound is below 10000; invalid cost is exactly 10000; lower-index
  exact tie wins.
- Candidate presentation permutation maps back to the same canonical selection.

### G. Metrics, inference, and gates

- Assert denominators 11,520; 2,880; 3,840; 960; 1,728; and 2,880 from explicit
  index arrays, including fallback/rejection cases.
- Action lures are excluded only from the valid predicted-transition metric,
  never from their required padded rollout/audit.
- Factorial signs, ratio-of-means, relative 2% margins, exact t critical values,
  strict wins/ties, and zero-denominator/nonfinite hard failures.
- Synthetic reports that fail each individual gate cannot be hidden by a
  favorable average or joint contrast.

### H. Budget and operation equality

- Every factorial cell/control ledger has the exact ordered arrays, dtypes,
  byte counts, total 393,216, cap 524,288, and identical allocation SHA.
- Exact registered call/update/counter vector, no early stopping, 576 scoped
  distance rows, inactive shadow work, temporary/metadata caps.
- Oracle is post-hash and cannot satisfy a treatment gate.

### I. Artifact state machine

- Canonical JSON writer/parser round trip and tamper rejection.
- Implementation lock includes source, both tests, inherited modules,
  callable-source, and budget hashes before any train seed.
- Calibration artifact shape/count/lock round trip using a handcrafted payload.
- Validation refuses absent/mutated calibration/lock and refuses overwrite.
- Test refuses absent, failed, incomplete, uncommitted, or self-inconsistent
  validation and any byte/hash change; valid handcrafted chain returns the
  exact calibration and validation SHA.
- A validation FAIL remains preserved and makes every test-open attempt fail.
- `verify` and `integrity` paths invoke no generator.

`tests/test_agi_world_memory_integration_integrity_v2.py` is read-only. Before a
final integrity ledger exists, it should verify that no locked test artifact is
present and validate canonical/hash chains for whatever earlier state is
legitimately present; it must not generate missing artifacts. Once the ledger
exists, it checks:

- the exact scientific file set, raw hashes, sizes, LF/BOM transport, and commit
  provenance;
- registration -> implementation lock -> calibration -> validation -> optional
  test links;
- validation seed count 40 and test seed count 60 only when unlocked;
- every serialized check/resource/hard-zero result and recomputed `passed`;
- validation-FAIL implies no test file, while a test file implies an earlier
  passing committed validation and exact validation SHA;
- any post-test audit-only source transition names base/current SHA and asserts
  no scientific artifact regeneration.

## 8. Recommended implementation order

1. Materialize the standalone JSON and freeze its raw SHA.
2. Implement immutable types, generator fixtures, core/codec, and unit seed
   guard; do not expose a split runner yet.
3. Implement scoped recall and its inherited-equivalence/poison tests.
4. Implement the preallocated residual schema, constrained/shadow/lesion passes,
   and provenance counters.
5. Implement candidate request/result, recursive rollout, wake action index,
   planning, candidate hashing, and evaluator seal tests.
6. Implement metric index arrays, exact denominators, effect reports, gates, and
   common-budget ledgers.
7. Implement canonical locks and the explicit artifact state machine; run only
   handcrafted/off-range unit tests.
8. Freeze the implementation lock. Only then may the parent workflow invoke the
   registered calibration stage, followed by exactly one validation run. A
   validation FAIL ends V2 and leaves test unopened.

This order preserves the useful G7-M and G9-CB interfaces without importing
their scientific conclusions or quietly changing the V2 estimand.

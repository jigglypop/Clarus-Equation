# G9-CBM V5 executable, resource, and irreversible-state amendment

Status: DRAFT_PRE_LOCK

Gate: REVISE_PENDING_INDEPENDENT_AUDIT

## Scope

[Definition] G9-CBM V5 is the next preregistration candidate for the same
world-model, episodic-memory, constrained offline recombination, and planning
experiment described by the recursively merged V2--V4 contract. V5 changes the
executable boundary, resource instrumentation, and one-shot evidence protocol.
It does not change the generator equations, factorial cells, controls, costs,
metric denominators, thresholds, 55 performance checks, 17 hard-zero checks, 12
resource checks, or scientific claim boundary.

[Derivation] V4 remains `BLOCKED_PRE_IMPLEMENTATION`. Independent audits found
that its candidate API exposed ambiguous allocation paths, its resource ledger
did not uniquely identify the source of every value, and its filesystem-only
opening witness could be deleted before a durable record existed. Those are
complete counterexamples to V4's implementation authorization, not empirical
results against the scientific hypotheses.

[Definition] This amendment is the single human-readable V5 amendment. The
three lane drafts `41-v5-api-contract-draft.md`,
`42-v5-state-machine-draft.md`, and `43-v5-resource-contract-draft.md` are
preserved as nonnormative design history. If their prose differs from the V5
registration, the strict JSON registration is authoritative.

No V4 or V5 registered seed, RNG stream, scientific world, calibration,
validation, locked test, or scientific runner was opened while preparing this
amendment.

## Locked predecessor

[Definition] The immutable V4 witnesses are:

- registration SHA-256
  `f414e7f357a4640039520255889362f3389af5b684f88f410217b201a0be2897`;
- amendment SHA-256
  `47ba50da367cf4fea73b2525c988ba813f101a9d2f197b67d3fe9b4e4b7b1ec3`;
- recursive V2 -> V3 -> V4 canonical merge SHA-256
  `8cd745c17fba41924f9774002b6a7b850611f0eb757909e7c70d1bd9032a6e0e`;
- V4 registered train, validation, and test execution counts all equal zero;
- no V4 implementation-lock, calibration, validation, test, or integrity
  artifact exists.

[Axiom: transport] Every recursively hashed registration, amendment, source,
test, runner, marker, and artifact is UTF-8 without BOM, LF-only, with exactly
one terminal LF where it is a text file. A fresh-checkout raw-byte check is a
precondition for implementation lock; a clean Git status alone is not evidence
of the physical EOL bytes.

## Scientific inheritance and fresh roles

[Definition] V5 inherits the effective V4 scientific object after recursive
hash verification. The only registered role change is fresh, disjoint seed
ownership:

- calibration/train: `101100..101139`, exactly 40 seeds;
- validation: `102100..102139`, exactly 40 seeds;
- locked test: `103100..103159`, exactly 60 seeds.

The three ranges are pairwise disjoint and must have zero collision with every
other repository preregistration. Handcrafted tests use no seed in a registered
range. No development pilot may use a V5 registered seed.

[Prediction] If the integrated mechanism is useful under the inherited C1--C5
gates, V5 validation will pass all 55 performance, 17 provenance, and 12
resource checks. This is a preregistered prediction, not a current result.

## Recursive-merge closure

[Definition] V5 deletes and re-adds complete top-level authorities for
`preregistration_integrity`, `data_roles`, `candidate_api`,
`implementation_dependency_manifest`, `artifact_state_machine`, `test_lock`,
and `failure_rules`. It deletes the complete `implementation_closure_v4` and
adds one complete `implementation_closure_v5`. No V4 intent, V3-runtime
permission, stale 8/3 manifest recipe, V2 candidate record, or V4 unlock record
may survive the merge.

[Definition] A missing parent added as one override is one new JSON path. Its
children are not separately listed as new paths. The observed set of newly
created paths must equal, not merely be a subset of,
`merge_semantics.allowed_new_override_paths`. Arrays replace and never
concatenate; null is a value and never deletion.

[Definition] Existing scientific leaves are changed only where the V5 boundary
requires a new name: the episodic recall function, train recall wrapper, and
per-origin failure route point to V5. Resource evidence leaves are added under
the existing `resources` object. Every other scientific leaf remains inherited.

## Candidate isolation

[Axiom: no raw array in candidate request] Candidate code receives no NumPy
array, memoryview, arena object, store object, truth object, seed, split, cell
name, evaluator handle, or object whose public or private field reaches such an
object. A NumPy slice is not a safe boundary because `.base` can expose the full
owner.

[Definition] Module-private registries owned by the stage capability map opaque
unguessable handles to arrays. Candidate-facing frozen proxies contain only a
handle token and a module-private seal. Their exact scalar methods are frozen in
the registration. They expose neither raw storage nor buffer protocol,
`__array__`, arbitrary indexing, reflection helpers, pickle/copy support, nor a
public registry reference.

[Definition] The proxy family separates:

- opaque token equality/hash;
- one-origin public state, action, action-token, goal, and completed-view reads;
- core, codec, cost, episodic, action-index, and residual-schema scalar reads;
- one-origin bounded writes to prediction, validity, key, source, q, cost, and
  selected-index destinations.

Hidden cue values are accessible only through a mask-enforcing recall proxy.
An attempted hidden-coordinate read increments its evaluator-owned provenance
counter before raising. Truth handles are held in a separate evaluator registry
which no candidate handle can address.

[Definition] Candidate execution writes through its bounded write proxy and
returns only a scalar receipt containing its own recall audit and payload hash.
After all 24 origins in a condition finish, the harness freezes all seven result
roots, verifies every per-origin digest, constructs evaluator-only read-only
views, and then unseals truth. Candidate code is never re-entered after truth is
unsealed.

[Derivation] Shuffled binding can therefore change only the registered completed
view handle. It cannot use another origin's target, audit, goal, actions, or
truth. Zero-q, zero-synthetic, persistence, and lesion controls likewise differ
only through the fixed dispatcher codes.

## Callable and call-graph closure

[Definition] The registration contains one ordered callable-boundary list and
one ordered registered call graph. A function record hashes the exact
`inspect.getsource` segment after CRLF/CR normalization to LF and exactly one
terminal LF. A class record hashes the entire class segment and covers every
method in that class; no method is separately treated as an unlisted callable.
The exact record fields, source paths, line bounds, source digest, canonical
compact-JSON recipe, and graph digest are registered.

[Axiom] Unlisted registered-path callables are forbidden. V3 and G7-M files are
raw-hashed provenance and off-range differential-test references only. No V3 or
G7-M function, constructor, method, runner, RNG, or artifact path executes on a
V5 registered path.

[Definition] The list includes the merge and strict-JSON boundaries, all proxy
and record classes, arena/counter/temp ownership classes, world construction,
calibration generation and fit, recall, dream, rollout, planning, aggregation,
resource replay, canonical writer, Git state transition, pytest parser, raw-role
parser, terminal-state classifier, unlock checker, integrity builder, read-only
verifier, public runner, and CLI entry point. Every exact evidence schema carries
the source-record digest and registered-call-graph digest.

## Resource closure

[Definition] Every factorial cell and non-oracle control owns one
`ConditionArenaV5` containing the inherited 36 C-contiguous NumPy owners in
registered order and totaling exactly 393,216 bytes. These owners are the direct
destinations of scientific results; post-hoc result copies are forbidden.

[Definition] Every condition owns one `uint8[1048576]` `TempArenaV5` root.
Temporary typed workspaces are nonowning, aligned, nonoverlapping leases into
that root. Use after release, overlapping leases, uninitialized reads, escaping
leases, an additional condition-local NumPy owner, or a nonempty close is a
resource failure.

[Definition] The 17 immutable common NumPy owners are generated once per seed
without a condition policy, sealed read-only, and shared by identity. Their
exact ordered inventory totals 302,168 bytes. Runtime identity/nonaliasing is a
runtime attestation; an artifact verifier can recheck bytes, records, and
digests but does not claim to reconstruct past Python object identity.

[Definition] The 29 budget rows are partitioned into 20 event counters, seven
live-owner measurements, and two registered limits. Arena initialization writes
zero to event rows and `-1` to measurement/limit rows before work. Only one
fixed event call site can increment each event row; snapshot fills the remaining
nine rows from live owners or registered JSON pointers and then freezes both
vectors. Expected-vector assignment is never evidence.

[Definition] Each of the 29 resource evidence records has a literal name,
required value, source class, source reference, and formula identifier in the
registration. Condition owner records, common owner records, budget records,
hard counts, lease summary and digest, environment evidence, and canonical
metadata-measurement payload are serialized, not replaced by favorable Boolean
claims.

[Definition] `metadata_utf8_bytes` is computed from one exact pre-check payload
using compact sorted ASCII-safe JSON with no terminal LF. The payload explicitly
excludes `metadata_utf8_bytes`, `resource_checks`, and every check derived from
that count, removing all direct and indirect cycles.

[Definition] Environment evidence records the exact Python/NumPy/platform
identity, process identity, child-process attempts, numeric-backend attempts,
network attempts, GPU attempts, external-trajectory reads, monotonic start/end
nanoseconds, and derived finite wall seconds. The 12 resource checks are
recomputed from these primitive records and the arena/counter evidence.

[Axiom: equal work] LTM-disabled routes still execute 72 recall slots and 576
masked distance rows but discard shadow values. Dream-disabled routes still
execute the single shared 288-pair traversal and the registered
`288/288/72/48/24/48/24/24/240/24/216/1` checks, endpoints, outputs, updates,
lesion classifications, padding decisions, and pass. Shadow data cannot enter
q, schema, action indexing, rollout, or selection.

## Irreversible scientific stages

[Definition] The four one-shot roles are implementation lock, calibration,
validation, and locked test. Each public invocation first creates canonical
consumed-marker bytes with exclusive creation, fsyncs them, commits the one-path
addition on the registered research branch, and creates the role's write-once
protected Git ref. Only after that ref verifies may the same process create its
opening envelope and launch pytest or claim a registered seed ordinal.

[Derivation] Deleting a worktree marker, pending file, final file, index entry,
or branch path cannot restore an unopened role because its consumed commit is
reachable from the protected ref. Ref/object-store destruction is outside the
failure model and makes verification invalid rather than reopening science.

[Definition] A normal or caught-failure terminal object is written to the exact
pending descriptor, fsynced, and atomically renamed without replacement to its
final path. The final path is committed once and bound to a final protected ref.
There is no resume, retry, amend, promotion by another process, alternate output,
dry run, or deserializable stage capability.

[Definition] Raw role bytes are parsed by one strict parser into an exact parse
record before the precedence classifier runs. Missing, empty, partial,
noncanonical, wrong-key, payload-hash-invalid, committed PASS/FAIL, dangling
pending, uncommitted final, deletion/mutation, and multiple-attempt states have
one registered classification. A hard process death is represented by the
durable consume witness plus surviving raw bytes; it is never misreported as a
caught Python exception.

[Definition] Integrity is a zero-seed derived terminal ledger written directly
to its exact final path with exclusive create and fsync. A partial integrity
write is terminal invalid and is never deleted, repaired, promoted, or
regenerated. The read-only verifier needs no integrity artifact: it recomputes
current role classifications and returns exact `verified`, `mismatches`, and
`terminal_states` fields without writing.

## Implementation-lock and test unlock

[Definition] The implementation-lock subprocess uses Python `3.14.4`, pytest
`9.0.1`, NumPy `2.3.5`, disables plugin autoload, user site, bytecode, color,
warnings, and pytest's cache provider, fixes repository root/config discovery,
and freezes every pytest-affecting environment variable. Its parser strictly
decodes UTF-8, normalizes CRLF, accepts one full output grammar, rejects duplicate
or unmatched summaries, and records raw stdout/stderr hashes and all outcome
counts. Literal or mocked success Booleans are forbidden.

[Definition] Before locked test consumption, validation primitive vectors
independently recompute the exact 55, 17, and 12 Boolean maps, all four
conjunctions, every source/callable/graph/NumPy/allocation/current-artifact hash,
and the committed validation bytes. The test path receives only an existence
check until the exact `UnlockRecordV5` exists in memory. The record contains no
future test marker, test artifact, or final commit value.

## Failure and claim boundary

[Definition] Caught exceptions and per-origin feasibility failures use one exact
non-self-referential failure schema. Per-origin validation/test failure is
evaluated only after all six generator-valid true costs for a sealed origin
exist; it preserves counts and the pre-unseal candidate digest, stops immediately,
and never resamples or fabricates primitive vectors.

[Incomplete] V5 has not yet passed an independent recursive-merge, executable
API, resource, or irreversible-state audit. It has no implementation lock and no
empirical result. The current status remains `DRAFT_PRE_LOCK` until those audits
report no P0 and the exact registration/amendment bytes are committed and
fresh-checkout verified.

[Axiom: claim boundary] Even a future all-of PASS would support only the narrow
claim that this fixed synthetic benchmark integrates observed-only episodic
completion and constrained offline missing-binding augmentation with a frozen
world-model/planning loop under the registered controls. It would not prove AGI,
human long-term memory, biological dreaming or sleep, hippocampal mechanisms,
consciousness, general causal discovery, or general planning.

# G9-CBM V4 implementation-closure amendment

Status: LOCKED_PRE_IMPLEMENTATION

## Disposition of V3

G9-CBM V3 is **BLOCKED_PRE_IMPLEMENTATION**. Its immutable registration is
`experiments/preregistration/agi_world_memory_integration_v3.json`, with raw
SHA-256
`bb98be25d60484f0f477b052e97e66be0aa416ebf33712ca269c09f7bfa3758b`.
No V3 implementation lock, registered train/calibration seed, validation seed,
or locked test seed was opened. The V3 prototype is retained only as a negative
implementation-boundary witness.

The stop was required because five outcome- or integrity-determining boundaries
were not executable enough to admit a unique implementation:

1. the direction and reject-case totalization of shuffled episodic binding;
2. whether the unconstrained-lesion objects affected rollout;
3. the returned schema-source code when a synthetic slot is replaced by the
   component fallback;
4. whether the registered persistent arrays were the actual write destinations,
   and how transient ownership was released and measured;
5. exact stage paths, durable opening intent, failure finalization, history, and
   deletion-after-opening behavior.

V4 closes only those boundaries and uses fresh roles and paths. No V3 result is
reinterpreted and no registered V3 seed is reused.

## 1. Shuffled-binding direction and totalization

Within every fixed `(context, port)` stratum, index the three canonical positive
origins by `i=0,1,2` in the already registered `prefix_local` order. Run the
ordinary M10 scoped recall independently for all three origins first. For each
origin `i`, construct a finite completed raw `(12,8)` view `V_i` by the ordinary
R1 rule:

- clamp the visible cells from origin `i`'s own noisy cue;
- if origin `i` accepts, fill its hidden cells from its accepted real exemplar;
- otherwise fill its hidden cells with `reshape(mu_codec,(12,8))`.

There is therefore a `V_i` even when recall rejects. The shuffled control for
origin `i` uses **exactly** `V_(i+1 mod 3)` and no other direction or conditional
skip. It computes

```text
q_hat_shuffled_i = mean(
  V_(i+1 mod 3)[:,0:4] - schema_anchor_raw_i[:,0:4], axis=0)
```

where `schema_anchor_raw_i` remains origin `i`'s own observed anchor schema.
Origin `i` retains its own anchor state, action candidates, goal, schema lookup,
and ordinary M10 `OriginRecallAudit`; audits are neither rotated nor replaced.
The next origin's acceptance affects only how its own `V` was completed. All
three `V` objects are completed before any shuffled `q_hat` is projected, so a
partial-rejection pattern cannot change traversal or permutation direction.

## 2. Lesion and zero-synthetic rollout semantics

The unconstrained-lesion control is rollout-byte-identical to M00. It has empty
queryable LTM, zero accepted hypothetical writes into the active schema, and the
same fallback lookups, predictions, inferred validity, schema keys/sources,
costs, and selected indices as M00. The 24 common lesion-audit objects remain in
the separate lesion buffer, never enter the 72-key schema or action index, and
never affect `q_hat`, rollout, or planning. Only this control exposes the common
audit's registered diagnostic `invalid_splice_rate=21/24=0.875`; all other
scientific primitive values are the corresponding M00 values. Every condition
still computes and hashes the same common shadow lesion audit.

The zero-synthetic control performs the same constrained dream calls, decisions,
and audit writes as M01. At lookup time, every valid missing key that would read
an accepted source-2 hypothetical in M01 instead reads the ordinary
same-component fallback and returns the same resolved key with schema source
`3=component_fallback`. Missing keys that were rejected also return source 3.
Observed keys remain source 1; invalid action/token padding remains key `-1`,
source 0. No returned zero-synthetic slot may carry source 2. The replacement is
made before rollout and is reflected in predictions and planning.

## 3. Actual owner, counter, and temporary arenas

For each factorial cell and non-oracle control, `ConditionArenaV4` allocates the
registered allocation ledger once in listed order. Every listed persistent
array is C-contiguous, owns its base allocation, has the registered dtype/shape,
and is the actual destination used by the scientific path. Candidate results
are views of their assigned arena slices; computing into an unregistered owner
and copying into the ledger afterward is forbidden. Views are not counted as
owners and may not outlive their registered owner. `inactive_padding` is a real
allocated owner and is never read by scientific arithmetic.

`ActualCounterV4` is capability-instrumented at the operations that perform the
work. It alone increments the 29 registered counters. Candidate constants,
expected values, post-hoc assignment, or self-reported counts cannot satisfy a
resource check. Every resource Boolean is derived from observed counters and
owner measurements only.

All nonpersistent NumPy owner arrays used by the scientific path are allocated
through `TempArenaV4`. It records the unique owning base, dtype, shape, bytes,
allocation ordinal, and release ordinal. A release removes the owner from the
live set only after all registered views are gone. The measured peak is the
maximum sum of live unique-owner `nbytes`; persistent owners, immutable common
generator inputs, and zero-copy views are excluded exactly once. Each
origin/candidate work buffer is released at its registered phase boundary before
the next reusable buffer is allocated. At seed return the temporary live set is
empty, and the observed peak must be at most 1,048,576 bytes. Any untracked
temporary owner, persistent copy, owner alias across conditions, nonempty live
set, or cap excess is a hard resource failure. UTF-8 metadata bytes are measured
separately against 32,768 bytes.

The callable-hash manifest contains the complete class sources of
`ConditionArenaV4`, `ActualCounterV4`, and `TempArenaV4` in addition to the
V4 wrappers, evaluator, and runner. It also contains every inherited V3
callable that V4 invokes. The ordered path manifest includes the complete raw V3
registration and module, so private helper and constant bytes are covered even
when their source is not a separate callable record. No alternate helper may
open a registered role, construct a registered world, bypass these arenas, or
synthesize a favorable counter.
Internal registered-seed evaluators require a module-private stage capability
created only by `run_agi_world_memory_integration_v4_gate` after the stage-open
checks. Handcrafted fixtures never receive that capability.

## 4. Calibration selector and per-origin feasibility

The recall-threshold candidates are every sorted unique finite train
positive/lure confidence plus one nonnumeric symbolic `REJECT_ALL` candidate.
Every candidate, including `REJECT_ALL`, is actually evaluated against all 960
positive and 960 lure train queries. `REJECT_ALL` accepts none. Retain only
candidates satisfying pooled false-lure rate `<=0.025` and pooled positive
wrong-all rate `<=0.025`, then select by this complete lexicographic maximum:

1. positive correct-identity count divided by 960 (abstention is wrong);
2. negative lure-accept count divided by 960;
3. on equality of the first two coordinates, symbolic `REJECT_ALL` ranks above
   every finite threshold, while finite thresholds rank by larger numeric value.

The candidate table, feasibility bits, objective tuples, complete tie set, and
winner kind are serialized. If the unique winner is `REJECT_ALL`, calibration
is infeasible and no finite `tau_recall` is persisted. Replacing the symbolic
candidate by a numeric infinity, omitting it from comparison, or considering it
only after finite selection is forbidden.

For every origin of every legally opened validation or test seed, the sealed
evaluator computes true `J` for all six generator-valid candidates `0..5` under
the registered common-noise truth. Before candidate scoring is unsealed, it
asserts that at least one of those six has `J<=25` and at least one has `J>25`.
Action-lure candidates 6 and 7 cannot witness either side. A failed origin
assertion terminates the already opened split as immutable FAIL; it never causes
resampling, seed dropping, or threshold changes. The assertion result, true
costs, and validity remain unavailable to candidate code until candidate return
bytes and harness audits are hashed.

## 5. Exact stage paths and irreversible two-phase opening

Every mutating public call requires an explicit `output_path` whose resolved
repository-relative path is byte-equal to the path registered for that stage.
Omission, an alternate path, a symlink escape, or a pre-existing path hard-fails
before opening. `verify` alone requires no output and is repeatable/read-only.
There is no `dry_run` parameter, CLI flag, environment switch, or helper API for
a registered role.

Calibration, validation, and test each use a distinct registered intent-marker
path. Opening is two-phase:

1. `<stage>_intent` creates its marker with `os.open(..., O_CREAT|O_EXCL|O_WRONLY)`
   and canonical LF JSON, flushes and `fsync`s it, and performs zero registered
   world/RNG construction. The marker must then be Git-tracked, committed, clean
   at its exact path, and byte-identical to `git show HEAD:<intent_path>`.
2. `<stage>` verifies that committed marker and every predecessor/hash gate,
   then exclusively creates the exact registered stage artifact path. Before
   the first registered RNG or world is constructed, the still-open descriptor
   receives and `fsync`s a canonical nullable FAIL envelope with phase
   `opened_unfinished`. This exclusive creation is the irreversible scientific
   opening and sets that role's append-only run count to one. The same descriptor
   is finalized exactly once to the canonical success/ordinary-gate-FAIL schema,
   or to the canonical exception FAIL schema; after close it is immutable.

The exception/crash FAIL schema has exactly: `experiment`, `stage`, `status`,
`failure_phase`, `failure_type`, `failure_message_sha256`,
`completed_registered_seed_count`, `expected_registered_seed_count`,
`registration_raw_sha256`, nullable predecessor raw hashes, intent raw SHA,
`registered_seed_execution_count`, and nullable `canonical_payload_sha256`.
`failure_message_sha256` hashes UTF-8 `type(exception).__name__ + ':' +
str(exception)` and no traceback, path, seed value, latent, or future is stored.
For `opened_unfinished`, the type/message hash and canonical-payload hash are
null. On normal completion, the inherited full artifact schema is used and
`status` is exactly `PASS` or `FAIL`.

An existing intent marker or stage artifact is never deleted, replaced, or used
to authorize another run. Before either phase, the runner checks the worktree,
Git index, and `git log --all -- <exact path>` for every V4 marker/artifact. If a
path is absent but the index or Git history shows it ever existed, the
experiment is terminally invalid and rerun is refused. Deleting, moving,
truncating, or committing deletion does not restore authorization. A dangling
committed intent or an `opened_unfinished` artifact is a terminal FAIL, not a
retry opportunity. Stage history is reconstructed in fixed stage order from
the immutable intent/artifact raw hashes. Every history record has exactly
`stage`, nullable `intent_raw_sha256`, nullable `artifact_raw_sha256`, nullable
`status`, `run_count`, and `registered_seed_execution_count`. Scientific stage
status is `INTENT`, `PASS`, or `FAIL`; a wholly unopened later stage has null
hashes/status and zero counts. Implementation lock has null intent, zero counts,
and its artifact hash; integrity uses a null self-hash to avoid self-reference.
Missing later stages are explicit records, never silently omitted.

Implementation lock and integrity also use their exact registered output paths,
exclusive creation, and no alternate outputs. The implementation lock may not
store a literal handcrafted-test success Boolean. It runs exactly

```text
sys.executable -m pytest -q --disable-warnings --maxfail=1
  tests/test_agi_world_memory_integration_v4.py
  tests/test_agi_world_memory_integration_integrity_v4.py
```

and stores the exact argv tail, both test-file raw hashes, collected/passed/
failed/error/skipped/xfailed/xpassed/deselected counts, integer exit code, and
raw stdout/stderr SHA-256 values. The lock is valid only when exit code is zero,
collected count is positive and equals passed count, all other counts are zero,
and the file hashes equal the ordered source manifest. A literal `true`, mocked
subprocess, omitted collection count, or manually assigned evidence is invalid.

## 6. Test unlock

Before the test intent may be created, validation must be Git-tracked,
committed, clean, and HEAD-identical. Unlock recomputes from validation primitive
vectors—not stored summaries—exactly 55 performance checks, 17 hard-zero checks,
and 12 resource checks, including their three conjunctions and final `passed`.
It requires exact registered key sets, exact equality with every stored Boolean,
and all 84 checks true. Resource checks use observed arena/counter evidence only.

The unlock then rehashes current registration/amendment bytes, every ordered
source path, every registered callable source, the exact NumPy version string, and the
ordered allocation ledger with its unsorted object-key recipe. It compares these
to implementation lock, calibration, and validation, also rechecking their raw
hash chain. Only after every recomputation and current-byte comparison passes is
the in-memory `UnlockRecordV4` constructed and the test intent path touched. No
test artifact byte is read or hashed before that point; mere existence checks
are allowed only for refusal. The test role then follows the same two-phase
opening and runs exactly once.

## 7. Fresh roles, paths, and inheritance

Registered roles are replaced, not extended:

```text
train/calibration: 95100..95139 (40)
validation:        96100..96139 (40)
locked test:       97100..97159 (60)
```

A static scan of all existing preregistration JSON files found no collision with
these integers before this amendment was written. Every V4 registration,
module, runner, test, artifact, and intent-marker path carries suffix `v4`.

V4 may import the committed V3 prototype as an explicitly hashed implementation
dependency. It may call only the V3 symbols listed by the V4 callable manifest;
their current `inspect.getsource` hashes are carried through every artifact.
Importing V3 does not authorize its runner, V3 artifact paths, V3 registered
roles, or any V3 stage-state helper. The V4 gate, evaluator, arenas, source
codes, controls, and stage capability remain authoritative.

All other generator equations, stream IDs, action/state space, cue chronology,
96-record wake ledger, residual-only dream restriction, R1 equations, 2x2
factorial estimands, common noise, controls, metrics, thresholds, 29-field
budget, 393,216-byte persistent allocation total, provenance rules, and claim
boundary are inherited recursively from V3 and V2. The V4 JSON registration is
the executable authority: it verifies raw base/amendment hashes, applies exact
segment-array deletions, recursively merges only declared overrides, and then
assigns its own top-level records. Any further scientific or integrity-relevant
change requires V5 with fresh seeds and paths.

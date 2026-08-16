# G9-CBM V4 independent resource and state-machine audit

Status: COMPLETE

Gate: REVISE

Scope: read-only audit of the frozen V4 amendment and registration, recursively
merged with V3 and V2. No V4/V3 scientific runner, registered seed, RNG, world,
calibration, validation, or test was opened. This audit does not predict an
empirical PASS or FAIL.

## Audited bytes and mechanical results

- V4 amendment: 15,468 bytes, raw SHA-256
  `47ba50da367cf4fea73b2525c988ba813f101a9d2f197b67d3fe9b4e4b7b1ec3`.
- V4 registration: 30,676 bytes, raw SHA-256
  `f414e7f357a4640039520255889362f3389af5b684f88f410217b201a0be2897`.
- Recursive V2 -> V3 -> V4 canonical merge: 75,088 bytes, SHA-256
  `8cd745c17fba41924f9774002b6a7b850611f0eb757909e7c70d1bd9032a6e0e`.
- The merged resource vector has 29 unique names; the allocation ledger has 36
  unique owners and sums to exactly 393,216 bytes.
- The registered leaf key sets contain exactly 55 performance checks, 17 hard
  zeros, and 12 resource checks. The ordered source and callable manifests have
  10 paths and 22 symbols respectively.

These are [산출] from the locked bytes. They establish byte and table
consistency, not that the corresponding runtime evidence can yet be produced
without an implementation choice.

## P0 findings

### P0-SM1: deletion-after-opening is not detectable for an uncommitted artifact

The V4 state-machine claim says deletion never restores authorization and says
the runner reconstructs the witness from the worktree, Git index, and
`git log --all -- <exact path>`. That evidence is insufficient for the interval
between scientific opening and the artifact's first Git commit.

A complete counterexample is:

1. a committed calibration/validation/test intent exists;
2. the runner creates the registered artifact with `O_EXCL`, fsyncs
   `opened_unfinished`, and begins registered RNG/world work;
3. the still-untracked artifact is deleted before it is added or committed;
4. the worktree path is absent, the index has no entry, and Git history has no
   entry for that artifact path.

The observable repository state is then byte-for-byte indistinguishable from
the legal pre-opening state, while the registered role has in fact run. A new
process can therefore open the same role again. The committed intent cannot
distinguish “not opened yet” from “opened and untracked artifact deleted.” This
is a complete counterexample to the append-only run-count and no-rerun parent
claim, and it applies most critically to the locked test role.

Required V4 fix before implementation: introduce a durable consumed/open witness
whose transition is committed to Git (or another specified append-only store)
inside the same authorized invocation and before the first registered RNG/world
operation. The invocation must start only from the pre-consumption state,
perform the durable transition once, retain an in-memory capability, and then
continue; a later invocation starting from the consumed state must refuse. One
workable contract is to commit the exact `opened_unfinished` artifact bytes at
their registered path before RNG, then finalize that same path after science;
Git history will retain the opening even if the final worktree file is deleted.
The exact commit message/author policy, clean-tree precondition, failure rule,
and history fields must be registered. Merely staging the file is not durable
across index reset or worktree removal.

### P0-SM2: the crash/finalization state is not total

The amendment calls the 15-field object an “exception/crash FAIL schema,” but
hard process termination can occur (a) after `O_EXCL` and before the first
envelope write, or (b) after seek/truncate begins finalization and before the
replacement JSON is completely written and fsynced. The surviving path can be
empty or partial JSON rather than either the exact `opened_unfinished` envelope
or the exact exception schema. The registered integrity terminal alternatives
do not define this byte state, and a killed process cannot serialize its own
exception type/message hash.

Required V4 fix before implementation: distinguish caught exceptions, which can
use the exact 15-field envelope, from hard interruption/corrupt finalization.
Register how integrity classifies any existing empty, noncanonical, partial, or
schema-invalid artifact as a terminal FAIL with run count one, how its raw bytes
are preserved and hashed without parsing, and the exact history status/fields.
The durable pre-RNG witness from P0-SM1 must remain available as the proof that
the role opened. Do not claim that an uncatchable crash writes a canonical
exception object.

### P0-R1: the 29-vector has no total observed-source recipe

V4 says `ActualCounterV4` alone increments all 29 entries at operations that
perform work and forbids constants or post-hoc assignment from satisfying the
resource gate. The inherited vector, however, mixes event counts, owner
measurements, and registered limits. Its final entries are:

```text
persistent_numeric_payload_bytes = 393216
persistent_byte_cap              = 524288
temporary_workspace_byte_cap     = 1048576
```

The two caps are configuration literals, not operations that can be observed or
incremented. Several other entries (`B_ltm_trace_bytes`, `P_core`) are naturally
owner/shape measurements rather than operation counts. Consequently one
conforming implementation may initialize those values from registration while
another may reject that as the expressly forbidden expected-value assignment.
That choice changes `budget_vector_exact`, one of the 12 hard resource checks.

Required V4 fix before implementation: register an ordered 29-row observation
recipe with exactly one source class and formula per name, for example
`event_counter`, `owner_measurement`, or `registered_limit`. Restrict
`ActualCounterV4.increment` to event rows; derive byte/shape rows from actual
arena owners; and copy the two caps only as explicitly classified registered
limits. Define the exact construction of `actual_budget_vector` from those
three source classes and require every row's provenance to be serialized. Map
the inactive LTM and dream operations to their exact counter rows so the
72/576 and 288/48/24/1 shadow-work claims are executable evidence rather than
end-of-run assignments.

### P0-R2: `TempArenaV4` cannot presently prove absence of untracked owners

The contract requires every nonpersistent NumPy owning base on the scientific
path to be allocated through `TempArenaV4`, while excluding common immutable
generator inputs. It does not register the exact exempt owner inventory or an
exhaustive allocation-observation mechanism. This matters because the permitted
V3 callable manifest includes arena-unaware implementations such as
`execute_candidate_v3`, `fit_shared_core_v3`, `_generate_seed_world_v3`, and
`build_evaluation_cues_v3`. The committed V3 source directly creates owners via
`np.array`, `np.empty`, `np.stack`, `np.concatenate`, `np.repeat`, and
`np.linalg.solve`; `CandidateResultV2.__post_init__` also copies result arrays.
Those paths cannot simultaneously yield arena-slice result views and satisfy
“all owners allocated through TempArenaV4” unless V4 forbids those calls or
registers a different tracked boundary.

Required V4 fix before implementation: freeze the actual registered-path call
graph, not only an allowed callable superset. For every invoked inherited
callable, either prove that it creates no condition-local owner, replace it with
an arena-aware V4 boundary, or specify how each owner is registered at creation.
Add an exact common-input owner manifest (semantic name, shape, dtype, bytes,
read-only flag, identity/hash rule shared by every condition) and a closed
classification test requiring every observed NumPy owner to be exactly one of
`ConditionArenaV4`, live `TempArenaV4`, or that manifest. Explicitly test that
candidate result fields share memory with their assigned persistent slices and
that no V3 result constructor performs a hidden copy.

## Boundaries that are otherwise closed

- [정의] The lesion audit is separate from the active 72-key schema/action
  index. Its 3 valid plus 21 invalid accepted audit objects account for
  `21/24 = 0.875`, reuse the common 288 traversal, and do not add a dream pass.
  The inherited contract expressly excludes shadow-lesion bytes from constrained
  provenance counts, so the 17 hard-zero gate need not count those 21 diagnostic
  objects.
- [정의] Disabled LTM and dream routes have exact shadow targets: 72 facade
  slots/576 distance rows and 288 enumerations/48 endpoints/24 update slots/one
  pass. The missing issue is observed-source routing in P0-R1, not the target
  numbers.
- [정의] The zero-synthetic route keeps M01 dream calls/audits and changes a
  valid missing returned source from 2 to 3 before rollout; the lesion route is
  otherwise primitive-identical to M00.
- [정의] Intent, opened-failure, and history records have exact field counts
  11, 15, and 6. Validation/test leaf check key sets are also exact.
- [정의] The recursively inherited six-field unlock record can remain compact
  only if the test-intent and test verifiers independently recompute all 84 leaf
  checks plus the four conjunction Booleans and replay the callable, NumPy,
  allocation, source, amendment, and artifact-chain comparisons. The literal
  `test_unlocked=true` is not evidence by itself.
- [정의] Implementation lock uses the exact two-file pytest argv and structural
  counts/hashes with zero registered-seed execution; a literal handcrafted-test
  success Boolean is forbidden.

## Gate decision

The frozen byte tables and scientific control meanings are internally
consistent, but the durable one-shot transition has a complete deletion
counterexample and the resource evidence has two unresolved ownership/source
boundaries. These are pre-implementation integrity failures, not empirical
failures. They are repairable without opening a registered seed, so the proper
decision is `Gate: REVISE`, not `BLOCKED`.

After the four P0s are closed in amended V4 bytes, rerun independent byte/merge,
state-machine, and resource audits before committing preregistration. Do not
create the implementation lock or open any V4 registered role under the current
bytes.

# G9-CBM V4 executable-contract red-team

Status: COMPLETE

Gate: REVISE

## Scope

This is an independent, read-only audit of the frozen V4 amendment, raw
registration, recursively merged V2 -> V3 -> V4 contract, and the committed V3
prototype interfaces. It tests whether two conforming implementers are forced
to build the same outcome- and integrity-determining procedure. It does not
test whether the empirical gate will pass.

No registered seed, scientific world, calibration, validation, test, V3
runner, or V4 runner was opened. The V4 module does not yet exist, as expected
at this pre-implementation gate.

## Audited bytes and reconstruction

- V4 registration: 30,676 bytes; raw SHA-256
  `f414e7f357a4640039520255889362f3389af5b684f88f410217b201a0be2897`.
- V4 amendment: 15,468 bytes; raw SHA-256
  `47ba50da367cf4fea73b2525c988ba813f101a9d2f197b67d3fe9b4e4b7b1ec3`.
- Strict recursive merge: 75,088 canonical bytes; SHA-256
  `8cd745c17fba41924f9774002b6a7b850611f0eb757909e7c70d1bd9032a6e0e`.
- The merged manifest contains 10 ordered source paths, 5 ordered dependency
  paths, and 22 callable boundaries.

## P0 findings

### P0-1: the source-manifest record cardinality is contradictory

The V4 override serializes 10 `ordered_source_paths` and 5
`ordered_dependency_paths`. The recursively inherited recipe nevertheless says
that the path records form an ordered **8-record** array, and that the **final
three** records correspond exactly to `ordered_dependency_paths`. No array can
simultaneously have the registered 10/5 cardinalities and the inherited 8/3
cardinalities.

This is not cosmetic: implementation lock, calibration, validation, test, and
the unlock record must copy and compare this exact array. Choosing 10/5 or 8/3
can therefore change whether validation is considered byte-locked and whether
the test may open.

Required repair: override both
`implementation_dependency_manifest.ordered_path_raw_sha256_recipe` and
`implementation_dependency_manifest.ordered_dependency_subset` to name the
exact 10-record source array and final-five dependency subset, then include
those override paths in the allowed-new/override audit as applicable.

### P0-2: the registered recall boundary names mutually incompatible APIs

The merged paths `episodic_ltm.recall_function` and
`calibration.recall_threshold.wrapper` still name
`scoped_hard_recall_v2`. There is no such implementation symbol on the V3
branch. The callable manifest instead requires both
`agi_world_memory_integration_v4.scoped_hard_recall_v4` and the committed
`agi_world_memory_integration_v3.scoped_hard_recall_v3`; the latter has the
typed signature
`(store, cue, scope_index, *, enabled=True) -> RecallResult` and delegates once
to the inherited G7-M V2 hard-recall boundary.

Following the merged function/wrapper fields requires an absent callable or an
invented adapter; following the callable manifest violates the two inherited
fields. That choice controls calibration confidences, acceptance identities,
and treatment recall, so it can change both the learned threshold and the
scientific result.

Required repair: explicitly replace both merged recall fields with one V4
boundary and state its exact delegation to the hashed V3/G7-M call, including
scope resolution, disabled/invalid sentinels, call count, and typed signature.

### P0-3: the V4 candidate/control and capability APIs are not totalized

The merged typed API still defines only `CandidateRequestV2` and
`CandidateResultV2`, while the manifest introduces new V4 callables and classes
without registered signatures or exact field schemas:

- `execute_candidate_v4`;
- `evaluate_factorial_seed_v4`;
- `ConditionArenaV4`;
- `ActualCounterV4`;
- `TempArenaV4`;
- the module-private stage capability consumed by registered-seed evaluators.

This omission is outcome-determining for the new controls. A
`CandidateRequestV2` contains one cue and no completed-view, preserved recall
audit, q-policy, schema-source substitution policy, arena-slice handle, or
capability field. The committed `execute_candidate_v3(request)` consequently
cannot by itself express all of (a) origin-i audit plus next-origin completed
view for shuffled binding, (b) zero-q, and (c) source-2-to-source-3 replacement
before rollout. The contract also forbids passing a cell label, evaluator
truth, master seed, or split to candidate code. Without a V4 typed boundary, a
conforming implementation can reasonably add control flags, mutate shared
schema, bypass `execute_candidate_v4`, or expose evaluator-owned state. Those
choices change candidate bytes, leakage behavior, and arena/counter evidence.

Required repair: register exact V4 constructor/function signatures, exact
request/result/control records and dtypes, ownership and lifetime of every
array field, the sole dispatcher sequence for every cell/control, and the exact
capability argument or closure boundary. The repair must show how shuffled,
zero-q, lesion, and zero-synthetic executions reach the same sealed candidate
surface without a cell label or evaluator truth.

### P0-4: the deletion-history mechanism does not make implementation lock irreversible

Scientific stages have committed intent markers, so an absent artifact after a
committed intent can be recognized as a dangling terminal FAIL. Implementation
lock has no intent marker. Its only durable probes are current worktree
existence, Git index, and `git log --all -- <path>`. If the O_EXCL-created lock
is deleted before it is added to the index or committed, all three probes are
again empty. The runner can then create a second lock, potentially after a new
clean source commit and a different handcrafted-test result. This directly
contradicts `deleting_an_artifact_never_authorizes_rerun` and permits selection
of the implementation before train opens.

Required repair: give implementation lock its own two-phase, committed intent
or an equivalently durable registered witness that exists before the lock
attempt. Define an absent-lock/present-intent state as terminal and include the
new path/stage in the fixed history and deletion probes. Integrity's zero-seed
rerun policy should be made consistent at the same time, although it does not
alter scientific seed outcomes.

### P0-5: opened-stage failure serialization is not uniquely executable

The registered exception schema ends with nullable
`canonical_payload_sha256`, but no recipe states whether a finalized exception
hashes the canonical object with that field null, omits the field, or uses some
other non-self-referential representation. Intent and calibration payloads use
the distinct, explicit field name
`canonical_payload_sha256_excluding_this_field`; that rule is not assigned to
the exception field.

The per-origin feasibility failure is also described only as an immutable
split FAIL. The contract does not choose whether this assertion uses the exact
exception schema or the inherited normal full split schema, whose primitive
vectors and summaries cannot yet be complete at an early failed origin.
Different choices yield different terminal artifact bytes, different history
hashes, and different integrity-verification behavior after the same registered
execution.

Required repair: define one exact non-self-referential hash recipe for the
exception payload and assign per-origin infeasibility (and every other
pre-report hard assertion) to one named failure schema. Fix the values/nulls and
completed-seed count semantics for that route.

## Boundaries that are sufficiently determined

- **Shuffled binding:** the fixed `(context, port)` strata, canonical origin
  order, next-origin direction, accepted-or-fallback completed views, own noisy
  visible clamp, own anchor, and own unrotated audit remove the V3 partial-reject
  ambiguity.
- **Unconstrained lesion:** rollout is byte-identical to M00; lesion objects are
  diagnostic-only and cannot enter schema/action lookup; only the registered
  `21/24` invalid-splice primitive differs.
- **Zero synthetic:** M01 dream calls and audits are retained, while every valid
  missing lookup substitutes component fallback before rollout and returns the
  same key with source 3, never source 2.
- **REJECT_ALL:** it is a real nonnumeric candidate evaluated with the 960
  positives and 960 lures. The inherited `wrong_all` is accepted-wrong/all, so
  rejection does not make the symbolic candidate infeasible; abstention is
  correctly counted as wrong in the primary correct-identity objective. A
  symbolic winner closes validation and persists no finite recall threshold.
- **Per-origin scientific condition:** validation/test truth must contain at
  least one valid candidate on each side of `J=25`; candidates 6 and 7 cannot
  witness it; no resampling, dropping, or threshold change is permitted.
- **Handcrafted evidence:** the exact two-file pytest argv, raw test hashes,
  structural collection/outcome counts, exit code, and stdout/stderr hashes
  exclude a literal success Boolean. This evidence becomes usable after P0-1
  makes its enclosing source manifest satisfiable.
- **Leakage intent:** the inherited forbidden-input list, masked-cue poison
  rules, pre-unseal candidate hashing, and V4 evaluator-private truth rule are
  coherent. P0-3 must supply the enforceable typed boundary before this can be
  implementation-locked.

## Verdict

The new scientific control semantics close the principal V3 ambiguities, but
five open P0s remain in the executable and irreversible boundary. Because at
least one open P0 forbids `Gate: PASS`, V4 must be revised and re-frozen before
any implementation lock or registered role is opened. The repair may retain
the current fresh seed ranges because no V4 registered seed has been executed;
all raw and recursively merged hashes must then be recomputed and independently
audited.

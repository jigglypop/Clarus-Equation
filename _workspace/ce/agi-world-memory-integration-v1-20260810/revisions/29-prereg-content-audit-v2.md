# G9-CBM V2 patched preregistration content audit

Status: COMPLETE

Gate: **REVISE**

## Scope

This audit rechecks the current patched
`experiments/preregistration/agi_world_memory_integration_v2.json` against the
unchanged locked contract and every P0-1 through P0-7 in
`revisions/28-prereg-audit-v2.md`. It is static only. No scientific module,
runner, calibration, development seed, registered train seed, validation seed,
or locked-test seed was executed.

Per the audit instruction, the registration's currently ignored/untracked Git
state is treated as a deliberately pending provenance step, not as a content
mismatch. It still must be closed before the implementation lock exactly as the
patched state machine now requires.

Audited byte snapshot:

```text
contract bytes: 58,768
contract SHA-256: 842512a55764e20a1b1f11c50c708b89bd8a8fe33b5c82f88a143f0cb36f7e70
patched preregistration bytes: 66,180
patched preregistration SHA-256: 6e059c8f74737fb61c42e43b1b8944d5994400bdb0b9854cdbd18beff295d2e4
```

The JSON is strict and finite: UTF-8 without BOM, LF only, exactly one terminal
LF, no duplicate key, and no nonfinite number. Its embedded contract SHA equals
the actual raw contract SHA.

## P0-1 through P0-7 crosswalk

| Prior finding | Current content gate | Re-audit |
|---|---|---|
| P0-1 tracked registration | **PENDING PROVENANCE; content PASS** | `artifact_state_machine.global_rules` and the preregistration/implementation-lock preconditions now require tracked, clean, committed, HEAD-identical registration bytes. Actual force-add/unignore is intentionally deferred until content PASS and is not scored here. |
| P0-2 generator and append order | **REVISE** | The `b` phases, signed `q_episode`, 96-record append order, constant `A[j]`, and common-noise key are fixed. Exact lure/cross-port/cue-mask construction is still missing; see P0-A below. |
| P0-3 learner/planner equations | **REVISE** | `q_hat`, inverse schema, recursive rollout, cost, selection, regret, success, lead indexing, and negative-regret rule are now present. The evaluator's generator-invalid cost override is still absent; see P0-B. |
| P0-4 metrics/gates/calibration | **REVISE** | Exact principal contrast formulas, `E_recall`, invalid-transition union, reports, and residual join statistics are now present. One calibration hard-failure rule remains attached to the wrong selector; see P0-C. |
| P0-5 codebooks/lesion | **PASS** | Key/source/scope/reason/dream/lesion domains, sentinels, left-failure priority, typed result fields, canonical 288 traversal, deterministic first 24, and 3/21 provenance are serialized. |
| P0-6 executable allocation | **REVISE** | Composite pseudo-dtypes were correctly split and byte arithmetic/hash are exact, but `bool8` itself is not an executable dtype name in the locked NumPy-2 environment and inactive shadow work remains unstated; see P0-D/P0-E. |
| P0-7 artifact state machine | **REVISE** | Ordered stages, no overwrite/rerun, lock payloads, conjunction recomputation, exact unlock record, and read-only integrity/verify are substantially fixed. A hash-field name changes without a producer and later artifacts omit mandatory direct hash references; see P0-F. |

## Repairs that are mechanically correct

- The canonical wake order is now `context, port, one displayed O binding,
  sign`, so it yields exactly 96 records rather than the prior accidental 288.
- `r1_completion_and_rollout` now fixes raw fallback, `q_hat`, schema inverse,
  phase/key selection, recursive state rows, invalid-token padding, and all
  `8*20` calls.
- All registered prediction/planning thresholds and principal paired vectors
  agree with contract Sections 8--9. The 11,520 / 2,880 / 3,840 / 960 / 1,728 /
  2,880 denominators remain exact.
- The registered budget vector has exactly 29 unique names in contract order and
  exact values.
- The allocation ledger now has 36 unique, scalar-dtype entries. Every
  shape-times-itemsize equals its declared bytes; subtotal before padding is
  `312,584`, padding is `80,632`, and total is `393,216`.
- Recomputing the declared canonical allocation recipe gives
  `0c7b95e4f63bc88f4d7b8bf5776e4bacf9ec5c1d8a8d078615f736be54a51d44`,
  exactly the stored expected SHA.
- Train `86100..86139`, validation `87100..87139`, and test `88100..88159`
  remain complete, unique, pairwise disjoint, and collision-free against other
  preregistration seed-valued fields.

## P0-A — false-lure, cross-port, and cue-mask generation remain non-executable

The patched generator records the lure cosine, formula, attempt cap, stream, and
counts, but it still omits contract lines 296--324 that determine the actual
recall calibration and safety queries:

- take the first admissible perpendicular lure attempt in canonical origin
  order, rejecting `norm(v_perp)<=1e-12` and requiring strict
  `max_other(abs(dot(f_lure,f_other)))<0.95`;
- after acceptance, draw the fresh normalized drift, initial state,
  innovations, signature noise, and generate the 12-step prefix from the same
  anchor schema, all from stream 13 in the contract's order;
- keep identical valid tokens for the unstored lure and never store its
  fingerprint or drift;
- construct the cross-port diagnostic by retaining the prefix token and
  replacing the suffix token by the next port's suffix token;
- select the `10/4/10` visible cue cells uniformly without replacement inside
  the three registered row slots, in canonical origin order, from mask stream
  11. A count vector alone does not specify the mask draw.

These choices change the 960 lure confidence pool, recall threshold, false-lure
gate, and cross-port hard-zero diagnostic. They cannot remain implementation
choices in a `standalone=true` preregistration.

**Exact fix:** add ordered `lure_generation`, `cross_port_diagnostic`, and
`cue_mask_sampling` fields containing the above contract algorithms, strict
inequalities, retry/failure behavior, streams, and canonical origin order.

## P0-B — true invalid sequences do not receive the registered penalty

`actions_and_planning.candidate_cost` correctly sets `J_hat=10000` when an
inferred sequence is invalid. However, `true_cost` only serializes the finite
state/action formula. It never states contract line 534:

```text
for a generator-invalid sequence, J(o,k)=P_invalid=10000 exactly.
```

Consequently the JSON permits the evaluator to score same-valued opaque-token
lures 6 and 7 by their numeric copied trajectories, which changes true regret,
success auditing, and the meaning of invalid-action detection.

**Exact fix:** add a separate literal
`true_invalid_sequence_cost="J(o,k)=10000 exactly when generator-valid is false"`
and state that the finite formula is used only for generator-valid sequences.
Keep generator validity evaluator-only and unseal it only after candidate hashes.

## P0-C — the recall selector's non-unique hard failure is not registered

Contract lines 969--979 hard-fail any empty/nonfinite/count-mismatched pool or
non-unique selector. The patched JSON adds
`empty_nonfinite_count_mismatch_or_nonunique_selector_hard_fail` only inside
`join_threshold`, where the fixed NumPy quantile is deterministic. The
`recall_threshold` object still lacks the corresponding selector assertion.

**Exact fix:** add to `recall_threshold` that empty/nonfinite/count-mismatched
pools and any non-unique result after the complete lexicographic objective
hard-fail calibration. Preserve sorted-unique finite candidates, symbolic
`REJECT_ALL`, strict `confidence>tau`, and the current objective order.

## P0-D — `bool8` is semantic prose, not an executable NumPy-2 dtype string

The repository lock selects NumPy 2.x (`uv.lock` resolves 2.3.5 for Python
3.11+), in which `np.dtype("bool8")` is not a valid alias. The JSON nevertheless
uses literal `bool8` in API dtypes and seven allocation entries. Therefore an
implementation that calls `np.dtype(entry["dtype"])` cannot construct the exact
ledger it is required to hash.

This is not a change to the contract's one-byte Boolean semantics; it is an
executable serialization repair.

**Exact fix:** use executable allocation dtype string `"bool"` for every
Boolean array and add a frozen dtype codebook such as
`{"contract_semantic":"bool8","numpy_dtype":"bool","itemsize":1}`.
Use the same executable name/mapping in API field dtypes, assert
`np.dtype("bool").itemsize==1` before train, then recompute the ordered allocation
SHA. Shapes, byte counts, padding, total, and the 29-vector remain unchanged.

## P0-E — common-budget shadow execution is not defined

The JSON simultaneously registers 576 scoped distance rows for every condition,
scope 0 as disabled/no-queryable LTM, and identical calls/capacity across all
cells and controls. Without the contract's shadow rule, a disabled scope returns
before distance access and produces zero rows. Likewise, no-dream cells lack the
explicit shadow enumeration/update rule even though they claim the same
288/48/24/one-pass budget.

Contract lines 1074--1084 resolve this: inactive LTM conditions execute
equal-shaped shadow facade/distance slots and discard results; inactive dream
conditions execute 288 checks, 48 endpoint values, and 24 update slots and
discard writes. External recall audit remains scope 0, and no shadow value may
affect rollout.

**Exact fix:** serialize these two shadow paths, their ownership/discard rules,
and their exact counters under `resources`. Require actual counters—not
self-reported constants—to equal the 29-vector in every factorial cell and every
non-oracle control.

## P0-F — the artifact hash chain changes field names without a producer

The implementation lock requires `ordered_source_raw_sha256`, backed by
`implementation_dependency_manifest.ordered_source_paths`. The calibration,
validation, test, and unlock record then switch to
`ordered_dependency_raw_sha256`, which is never defined, given a recipe, or
produced by the implementation-lock required fields. This makes the first
calibration precondition impossible to verify literally.

Moreover, contract lines 1179--1191 require validation/test artifacts to name
the exact registration, implementation lock, core, calibration, inherited
module, runner, and test SHAs. The patched validation required fields omit
`core_payload_sha256`, `ordered_callable_sha256`, `numpy_version`, and
`ordered_allocation_ledger_sha256`; the test's inherited phrase does not replace
direct serialized references.

**Exact fix:** define one ordered path-to-raw-SHA mapping with one canonical
recipe and one field name, and carry that exact mapping byte-for-byte through
implementation lock, calibration, validation, test, unlock, and integrity.
Alternatively define explicit disjoint source/dependency maps, but then produce
and verify both at every stage. Add direct core, callable, NumPy, allocation,
runner, and both-test references to validation/test required fields. Hash
aggregation may supplement, not replace, the contract's named dependency
references.

## Decision

**REVISE.** The large patch faithfully closes most defects in audit 28, and its
transport, contract identity, equations already present, codebooks, seed roles,
29-vector, allocation arithmetic, and allocation SHA are internally sound.
Content is not yet implementation-lockable because exact diagnostic generation,
the true invalid-action penalty, recall-selector failure semantics, executable
Boolean dtype mapping, inactive shadow work, and the artifact hash-chain field
names remain unresolved.

Do not run the implementation lock or open train. Apply only contract-faithful
serialization repairs, recompute the preregistration and allocation SHAs, and
repeat static content audit. Once content passes, complete the deliberately
pending Git provenance step from P0-1 before implementation lock.

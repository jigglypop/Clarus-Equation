# G9-CBM V2 final preregistration content audit

Status: COMPLETE

Gate: **PASS**

## Scope and meaning

This is the final static content audit of the current
`experiments/preregistration/agi_world_memory_integration_v2.json` against the
unchanged locked `revisions/00-contract-v2-draft.md` and the unresolved items in
audits 28 and 29. It checks exact scientific literals/formulas, transport,
NumPy dtype executability, lure/mask/cross-port construction, invalid true
cost, calibration failure semantics, shadow budgets, the unified source hash
chain, unlock/pass mappings, and allocation arithmetic/hash.

No implementation module, runner, calibration, development seed, registered
train seed, validation seed, or locked-test seed was executed. The only runtime
library operation was a mechanical `numpy.dtype` construction check; it did not
generate a world or evaluate scientific code.

Audited byte snapshot:

```text
contract bytes: 58,768
contract SHA-256: 842512a55764e20a1b1f11c50c708b89bd8a8fe33b5c82f88a143f0cb36f7e70
preregistration bytes: 77,134
preregistration SHA-256: b336fed11bf964512d1a2d50dd6c103a9593b426a986d4fe3b26e0bafa1338c2
```

The embedded contract SHA is byte-equal to the actual locked contract SHA.

## Final gate table

| Boundary | Gate | Static result |
|---|---|---|
| Strict transport | **PASS** | Strict JSON parse succeeds; no duplicate key or nonfinite number. UTF-8 has no BOM, contains no CR/CRLF, and has exactly one terminal LF. |
| Locked scientific literals | **PASS** | `d=4`, `m=2`, `H=20`, `K=8`, `O/M`, streams `0..13`, `D/B/G`, primitive distributions/scales, 20-parameter learned core, wake/evaluation counts, action order, goals, cost, metrics, controls, and gates match the contract. |
| Generator completion | **PASS** | Phase-specific `P/C/S+E`, signed `q_episode`, 96-record append order, constant wake action, and `(seed,origin,lead)` common-noise coupling are explicit. |
| Lure, mask, and cross-port diagnostics | **PASS** | The first admissible strict-cosine lure attempt, canonical origin order, 10,000-attempt failure, stream-13 fresh drift/state/innovation/signature draws, same anchor schema/action/tokens, unstored status, slot-local stream-11 permutations `10/4/10`, and next-port same-local suffix replacement are frozen. |
| LTM/R1/dream | **PASS** | Scoped 12-row recall, raw/standardized boundary, fallback, `q_hat`, recursive rollout, residual-only schema/dream, key/source/reason/provenance codebooks, deterministic 288-to-240-to-24 lesion route, and all denominators agree. |
| True and inferred planning costs | **PASS** | The finite cost applies only to generator-valid sequences; evaluator-invalid sequences receive exactly `J=10000`. Candidate-inferred invalid sequences receive exactly `J_hat=10000`; generator truth stays sealed until candidate bytes are hashed. |
| Calibration failure semantics | **PASS** | Recall and join pools/counts are exact; empty, nonfinite, mismatched-count, or non-unique complete selectors hard-fail. `REJECT_ALL` remains symbolic, a winning value closes validation, and persisted thresholds must be finite. |
| Executable dtypes | **PASS** | All allocation dtype strings construct under the current NumPy 2 environment: `bool`, `float64`, `int64`, `int16`, and `uint8`. Boolean `bool` has itemsize one; removed alias `bool8` is forbidden and retained only as the Markdown semantic name. |
| Common work/resource budget | **PASS** | The 29-vector has 29 unique fields in exact order/value. Disabled LTM and dream routes execute equal-shaped shadow calls/counters, discard results/writes, and cannot affect `q_hat`, schema, or rollout. |
| Allocation ledger | **PASS** | 36 unique scalar-dtype arrays each satisfy shape-times-itemsize; subtotal plus padding is exactly `393,216` under the `524,288` cap. The canonical ledger SHA recomputes exactly. |
| Unified source/dependency chain | **PASS** | Five primary paths plus three inherited dependency paths form one ordered, unique eight-record `{path,raw_sha256}` chain copied unchanged through lock, calibration, validation, test, unlock, and integrity. Callable, NumPy, core, and allocation hashes are also direct required fields where the contract requires them. |
| Unlock record | **PASS** | Exact six-field order is validation SHA, registration SHA, implementation-lock SHA, calibration SHA, ordered eight-path SHA records, and literal `test_unlocked=true`. Test still requires a committed, clean, HEAD-identical recomputed validation PASS. |
| Split pass mapping | **PASS** | Performance/resource/hard-zero key sets contain respectively 55/12/17 unique keys. The 17 hard-zero keys exactly equal the canonical provenance order. Missing, extra, or duplicate keys hard-fail; JSON object insertion order is correctly non-scientific because artifact keys are canonically sorted. `passed` is exactly the conjunction of the three recomputed component flags. |
| Seed/state chronology | **PASS** | Train 40, validation 40, and locked test 60 seeds are complete, unique, pairwise disjoint, run once, and protected by the no-overwrite/no-rerun state machine. |

## Exact mechanical recount

The current allocation ledger has 36 entries and no duplicate semantic name.
Every dtype is executable and gives the registered itemsize:

```text
bool     1 byte
uint8    1 byte
int16    2 bytes
int64    8 bytes
float64  8 bytes
```

The ledger bytes sum to `393,216`. Canonical serialization under the registered
recipe recomputes:

```text
7f5c52b1b4aa01f8141ce821ed1bf4164e3fdf131ae828f08b20a8280f3079b4
```

which is exactly `resources.allocation_ledger_expected_sha256`. The Boolean
dtype change altered only the executable spelling and the ledger hash; shapes,
one-byte semantics, padding, total, caps, and 29-vector are unchanged.

The source chain is exactly:

```text
5 primary paths + 3 inherited dependency paths = 8 unique ordered paths.
```

`ordered_source_paths` is byte-for-value the concatenation of the registered
primary and dependency arrays. There is no remaining
`ordered_source_raw_sha256` / `ordered_dependency_raw_sha256` name split.

## Closure of audits 28 and 29

- Prior P0-2 generator/append-order defect: **closed**.
- Prior P0-3 missing learner/planner and invalid-cost equations: **closed**.
- Prior P0-4 metric/gate/calibration failure ambiguity: **closed**.
- Prior P0-5 codebook and lesion traversal ambiguity: **closed**.
- Prior P0-6 pseudo-dtypes and inactive-budget ambiguity: **closed**.
- Prior P0-7 artifact hash/state-machine ambiguity: **closed**.
- Audit-29 P0-A through P0-F: **all closed** by the current bytes.

No executable content blocker remains in the audited preregistration snapshot.

## Provenance boundary and decision

**PASS** is a content and static-consistency decision only. It is not an
implementation-lock PASS, calibration result, validation result, test result,
or empirical support for C1--C5.

The JSON is intentionally still ignored/untracked while content review closes.
That pending repository action is not a content failure in this audit. Before
implementation lock, the state machine must now be followed literally: add the
exact approved bytes to Git, commit them, require a clean path byte-identical to
`git show HEAD:<registration-path>`, and then recompute the registration raw SHA
used by source/test constants and the implementation-lock artifact. Until that
provenance postcondition and all handcrafted off-range integrity tests pass, no
registered train seed may be opened.

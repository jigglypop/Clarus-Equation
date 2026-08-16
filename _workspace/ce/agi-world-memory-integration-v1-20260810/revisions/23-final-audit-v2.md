# G9-CBM V2 final pre-implementation audit

Status: COMPLETE

Gate: **PASS**

## 1. Final disposition

The latest `00-contract-v2-draft.md` is sufficiently closed to begin
implementation. All findings P0-a through P0-d from `20-audit-v2.md` and all six
mechanical contradictions from `21-consistency-v2.md` are now contractual,
typed, and count-compatible.

This PASS means **implementation is permitted**. It is not a prediction that
calibration, validation, or locked test will pass, and it is not empirical support
for C1--C5. Implementation lock, off-range integrity tests, one train/calibration
opening, validation, and conditional test unlock remain future state transitions.

No code, test, calibration, benchmark, development seed, or registered seed was
run in this audit.

## 2. Prior P0-a--d closure

| Finding | Gate | Contract evidence | Final audit |
|---|---|---|---|
| P0-a, residual-only adapter | **PASS** | `00-contract-v2-draft.md:305-396` | The two-record observed means, prefix/connector/suffix aggregation, component fallback, canonical candidate order, preallocated empty-key resolution, and `J_left <= tau`, `J_right <= tau` equality rule are unique. Fingerprint columns never enter joins or rollout. |
| P0-b, planning anchor and typed cost inputs | **PASS** | `:406-409,487-535,537-582` | `x_origin=x_prefix[12]`, public goal rows and prediction rows share the same `ell=0..19` convention, and `CostSpecV2`/`CodecSpecV2` supply the exact frozen normalizers. |
| P0-c, lure construction | **PASS** | `:119-130,275-303` | PCG64/SeedSequence ownership, NumPy-version lock, orthogonal perturbation, fixed target cosine `0.85`, other-bank cosine bound, draw order, and 10,000-attempt hard failure are fixed. |
| P0-d, split feasibility versus lock | **PASS** | `:1057-1108` | Feasibility runs only inside the legally opened split. Validation/test seeds and futures are unavailable during pre-train and train; a split failure cannot authorize the next split. |

## 3. Mechanical-consistency closure

| `21-consistency-v2` item | Gate | Resolution |
|---|---|---|
| P0-1, truth prefix / public goal | **PASS** | `:193-198` excludes `f_hat` from truth generation; `:489-517` fixes `x_origin`, `(20,4)` goal shape, lead indexing, and cost input. |
| P0-2, raw/standardized symbols | **PASS** | `:218-232,275-322,411-413` makes `T_raw` the episodic payload, `T_std` a derived view, uses `sigma_codec` for cue noise, and names the residual-only V2 fallback. |
| P0-3, typed action/return schema | **PASS** | `:458-485` freezes key order, source codebook, sequence validity, and invalid cost. `:550-582` separates 24 per-origin `CandidateResultV2`/`OriginRecallAuditV2` objects from the harness-owned `(72,) SeedRecallAuditV2`; lure and cross-port cues never enter `execute_candidate_v2`. This preserves exactly 72 LTM calls and the allocation ledger. |
| P0-4, `U_s` and denominators | **PASS** | `:654-750` uses `ell=0..H-1`, an H-dependent `U_s(H)`, fixed scalar denominators, and only the explicit `max(accepted,1)` recall totalization. |
| P0-5, duplicate 216 counters / lesion | **PASS** | `:609-624` separates constrained component rejection from lesion capacity padding, fixes lesion composition `3+21`, asserts invalid rate `21/24`, and retains all 48 endpoint calculations. |
| P0-6, allocation schema | **PASS** | `:947-1055` gives two named `(29,) int64` vectors, removes the undefined 32-counter row, uses one-dimensional `(80632,)` padding, totals exactly `393,216` bytes, and stays below the `524,288` cap. |

## 4. A1--A10

| Action | Gate | Basis |
|---|---|---|
| A1 factorial estimand | **PASS** | C1 uses the marginal LTM contrast; simple effects and `RR_joint` are distinct (`:756-810`). |
| A2 metrics | **PASS** | `E_all`, `U_s(H)`, `E_uv`, H5 slicing, nonfinite behavior, and every primary denominator are fixed (`:654-750`). |
| A3 planning | **PASS** | Spaces, do-equation, common noise, action order, opaque lures, goal, cost, regret, success, bounds, and ties are explicit (`:47-165,421-535`). |
| A4 leakage/API | **PASS** | Append-only chronology, minimal typed inputs/outputs, evaluator-after-hash order, poison/taint tests, cell-order invariance, and test-path denial are hard gates (`:537-582,896-945`). |
| A5 provenance | **PASS** | Four semantic tuples and the canonical 17 hard-zero integer counts are fixed; attempts and successes are distinct (`:896-945`). |
| A6 thresholds | **PASS** | Scoped recall and residual-only join calibration use exact train-only populations, selectors, inequalities, and failure rules (`:854-894`). |
| A7 common budget | **PASS** | Calls, capacities, counters, owned bytes, inactive padding, controls, caps, and equality hashes are numeric (`:947-1055`). |
| A8 comparisons | **PASS** | Benefit signs, ratio-of-means, positive denominators, Student-t constants, strict wins/ties, relative 2%, and no-synergy language are frozen (`:756-852`). |
| A9 controls/P1 | **PASS** | Shuffled binding, zero-q, deterministic lesion, zero-synthetic-slot, absolute caps, persistence, lure upper CI/max, and cross-context zero have decision roles (`:587-652,782-852`). |
| A10 source boundary | **PASS for V2** | Literature remains motivational only and the allowed conclusion excludes biological memory, dreaming, consciousness, and AGI (`:29-45,1128-1129`). The pre-existing Schapiro wording correction remains a separate publication-document P1 and does not alter this implementation gate. |

## 5. Split lock and unlock

The state machine is internally ordered:

1. Freeze preregistration, implementation, runner, tests, inherited modules, and
   codec SHAs before train (`:854-858,1094-1096`).
2. Open train/calibration seeds `86100..86139` once; write one calibration
   artifact or preserve failure (`:854-894,1067-1077`).
3. Open validation seeds `87100..87139` exactly once only with the frozen
   calibration and dependency hashes (`:1067-1098`).
4. Keep test seeds `88100..88159` inaccessible after any validation, integrity,
   control, or resource failure (`:1067-1108`).
5. At test entry, require a tracked, clean, HEAD-identical validation artifact,
   recompute all pass flags and SHAs, and create only then the in-memory unlock
   record (`:1094-1108`).
6. Any scientific post-lock change creates V3 with fresh seeds (`:1110-1114`).

There is no preflight path that legally reads a later split. Off-range handcrafted
fixtures may test implementation assertions but may not substitute for or inspect a
registered seed.

## 6. Formal claim boundary

- R1 is **[axiom: model selection]**, not a theorem or empirical result.
- The generator, codec, action set, costs, thresholds, budgets, and mappings are
  **[definitions/axioms]** of this finite software experiment.
- C1 marginal LTM benefit, C2 both matched dream benefits, and C3 integrated
  planning benefit are now well-formed **[predictions]**. Their truth remains open.
- C4 is a bounded **[safety prediction]** about the enumerated capability,
  provenance, constraint, and lock channels. It is not a theorem excluding every
  conceivable implementation defect.
- C5 is **[axiom: reporting rule]**. Metric-specific interactions are reported;
  generic synergy is not registered.
- The dimensionless PASS in `13-dimensionless-v2.md:5-37` establishes symbolic
  coordinate/unit consistency only. It supplies no performance or biological evidence.
- A future PASS may support only the conditional software-component statement in
  `00-contract-v2-draft.md:29-45`; it cannot support general world modelling,
  semantic consolidation, recurrent attractors, biological memory, dreaming,
  consciousness, continual learning, or AGI.

## 7. Audit counts and execution record

- Top-level claims audited: **5**
- Prior P0-a--d: **4 PASS / 0 open**
- Mechanical P0-1--P0-6: **6 PASS / 0 open**
- A1--A10: **10 PASS / 0 open for implementation**
- Established empirical theorems/outputs: **0 / 0**
- Registered/development seeds run by this audit: **0 / 0**
- Parent claims deleted: **0**

Final decision: implementation may begin within the frozen V2 scope. Before any
registered train seed is opened, the implementation, typed schemas, handcrafted
integrity tests, exact allocation ledger, and dependency hashes must satisfy the
contract. Any failure invokes the stated stop condition; it does not weaken this
preregistration or convert PASS into an empirical claim.

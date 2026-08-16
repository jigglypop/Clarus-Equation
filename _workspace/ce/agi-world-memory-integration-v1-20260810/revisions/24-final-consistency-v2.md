# G9-CBM V2 final mechanical recount

Status: COMPLETE

Gate: **REVISE**

## 1. Static scope

This audit rereads the latest `revisions/00-contract-v2-draft.md` against the six
findings in `revisions/21-consistency-v2.md`. No code, calibration, development seed,
registered train seed, validation seed, or locked-test seed was run.

## 2. Previous findings: closure state

| prior item | state | exact recount |
|---|---|---|
| P0-1 intercept / prefix / goal | PASS | Four intercepts remain in the 20-parameter core and every learned residual/rollout/control. Truth prefix generation excludes `f_hat`. Prediction and goal both start at `x_prefix[12]`; returned rows are lead states 1–20 and `g` is `(20,4)`. `CostSpecV2` now carries `mu_x,sigma_x`. |
| P0-2 raw/std boundary | REVISE | Raw episodic ownership and residual-only schema use are now stated correctly, but the two displayed raw/std formulas still combine `(12,8)` arrays with `(96,)` vectors without an explicit flatten/reshape operation. |
| P0-3 typed action/return | PASS, with one sentinel defect below | Key order `0..71`, unresolved `-1`, source codes `0..3`, sequence validity, invalid cost, per-origin result, seed-global 72 audit, audit order, identity and scope codebooks, and the 17 hard-count order are now explicit. |
| P0-4 lead index / denominators | PASS | Stored row `ell` is mathematical lead `ell+1`; `U_s(H)` uses `a[k,ell]`; the validity audit uses `ell mod 12`; the `wrong_given_accept` totalization is the sole named exception. |
| P0-5 lesion counts | PASS | The contract separates constrained rejection `216` from lesion padding `216`, freezes `240 -> 24`, and asserts 3 valid-missing plus 21 invalid cross-port accepted lesion objects, hence `21/24=0.875`. |
| P0-6 allocation | PASS arithmetically, REVISE operationally | The table now has valid one-dimensional padding and two 29-vectors, but lesion-specific runtime counts are still required of every condition without an equal shadow-lesion execution rule. |

## 3. Verified arithmetic and chronology

- `b_registered` has exactly **29** fields. Each int64 vector is therefore
  `29*8=232` bytes.
- The ordered allocation subtotal before padding is exactly **312,584** bytes.
  Adding `(80632,) uint8` gives exactly **393,216** bytes.
- `393,216 <= 524,288`; persistent-cap headroom is exactly **131,072** bytes.
- The temporary cap is separately fixed at `1,048,576` bytes and metadata at
  `32,768` UTF-8 bytes.
- Exact metric denominators remain correct: `11,520`, `2,880`, `3,840`, `960`,
  `1,728`, `2,880`, and origin-level denominators `24`.
- Exact calibration populations remain correct: core `46,080`, states `49,920`,
  codec records `3,840`, positive/lure queries `960/960`, and four-residual-coordinate
  endpoint joins `3,840`.
- Split chronology is closed: implementation lock precedes the one train opening;
  validation/test data are not generated during pre-lock or train; validation runs once;
  test requires a committed, clean, byte-identical passing validation artifact and a
  recomputed in-memory unlock record; any split failure forbids the next split.

## 4. Remaining executable contradictions

### P0-A — flatten/reshape is still missing at the raw/std boundary

`T_raw` and `S_std` are `(12,8)`, while `mu_codec` and `sigma_codec` are `(96,)`.
The current displayed expressions are not valid NumPy broadcasting. Replace both by:

```text
T_std = reshape(
    (vec_C(T_raw) - mu_codec) / sigma_codec,
    (12,8), order="C")

S_raw = reshape(
    mu_codec + sigma_codec * vec_C(S_std),
    (12,8), order="C")
```

`vec_C` must mean C-order flattening and must be the same operation used by inherited
masked recall, calibration hashes, schema construction, inverse standardization, and
recall-error scoring. This textual fix changes no count or allocation.

### P0-B — recall-confidence finiteness is incompatible with invalid precheck

The typed seed audit serializes `confidence:float64`; invalid cross-port scopes reject
before a distance call. No confidence value is specified for that case, while the
inherited empty/precheck result is nonfinite and Section 9.4 requires all outputs finite.
Freeze a finite audit sentinel without changing internal inherited semantics:

```text
When scope=0 (invalid_precheck): accepted=false, identity=-1,
serialized_confidence=-2.0. The inherited internal confidence may remain -inf but is
never serialized, thresholded, averaged, or exposed to the candidate.

When scope=1: confidence must be finite and in [-1,1].
```

Also state that the calibration's `+infinity` item is a symbolic `REJECT_ALL` candidate,
not a member of the finite confidence pool. If it wins, calibration is infeasible and
validation remains closed; a persisted operational `tau` must be finite.

### P0-C — the common 29-vector contains lesion operations absent from normal cells

`b_registered` requires `N_lesion_nonobserved_pairs=240`,
`N_lesion_accepted_slots=24`, and `N_lesion_capacity_padding=216` for every factorial
cell and every control. The prose executes those operations only in the unconstrained
lesion control; ordinary cells only run the constrained/shadow dream pass. Thus their
actual 29-vectors cannot all equal `b_registered` as written.

Preserve the 29-vector and allocation by inserting this exact rule:

```text
Every condition, including M00/M10/M01/M11 and every non-lesion control, executes the
same deterministic 240-to-24 lesion classification into its preallocated lesion audit
buffer after the 288 pair checks. In a non-lesion condition this is a shadow-only audit:
its payload and reason codes are hashed, never enter the 72-key schema/action index,
never affect rollout/selection, and are excluded from constrained provenance counts and
the condition's scientific invalid_splice_rate. The unconstrained-lesion control exposes
that same audit only to its registered diagnostic metric. All conditions therefore
record actual counts 240,24,216.
```

Because the single `(288,)` pair-reason array is needed for the active constrained pass,
the shadow lesion must not overwrite it. Either derive the three shadow counters and the
fixed 3/21 split directly into the already allocated lesion occupancy/provenance arrays,
or add a second reason array and rebalance padding before lock. The chosen ownership and
hash order must be stated explicitly.

## 5. Final disposition

The requested intercept, lead-index, field/codebook, lesion 3/21, denominator,
calibration, split chronology, 29-vector length, and `393,216`-byte arithmetic checks are
closed. The remaining raw/std shape error, finite-audit sentinel, and per-condition
lesion-operation mismatch are executable rather than empirical issues. Implementation
lock and train opening remain forbidden until those three textual contracts are closed;
no seed run is needed to repair or re-audit them.

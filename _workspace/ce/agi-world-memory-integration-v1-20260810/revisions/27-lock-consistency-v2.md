# G9-CBM V2 lock-consistency recount

Status: COMPLETE

Gate: **PASS**

## Scope

Static final recount of the latest `revisions/00-contract-v2-draft.md` against
`revisions/21-consistency-v2.md` and `revisions/24-final-consistency-v2.md`. No code,
calibration, development seed, registered train seed, validation seed, or locked-test
seed was run.

## Final checks

### Codec coordinate boundary — PASS

- `T_raw` is the owned episodic `(12,8)` payload.
- `vec_C` is explicitly C-order flattening.
- Forward standardization is exactly
  `reshape((vec_C(T_raw)-mu_codec)/sigma_codec,(12,8),order="C")`.
- Inverse standardization is exactly
  `reshape(mu_codec+sigma_codec*vec_C(S_std),(12,8),order="C")`.
- The same flatten/reshape convention is required for recall, schema, calibration hash,
  inverse decode, and recall-error scoring. Raw and standardized shapes are therefore
  executable and unambiguous.

### Recall sentinel and threshold — PASS

- Scope codebook is disjoint and complete:
  `0=disabled/no queryable scope`, `1=valid 12-row scope`,
  `2=invalid context/component precheck`.
- Scope `0` and `2` serialize `accepted=false`, `identity=-1`, and finite confidence
  sentinel `-2.0`; cross-port diagnostics always use scope `2`.
- Scope `1` confidence is finite in `[-1,1]`.
- Any inherited internal `-inf` is forbidden from serialization, thresholding,
  averaging, or candidate arithmetic. This is consistent with the all-output-finite
  gate.
- `REJECT_ALL` is symbolic, is outside the finite confidence pool, is never persisted,
  and makes calibration infeasible if it would win. Persisted operational `tau` must be
  finite.

### Typed result and lead indexing — PASS

- Per-origin `CandidateResultV2` and seed-global `SeedRecallAuditV2` ownership are
  separated. The seed harness alone aggregates 24 positive, 24 lure, and 24 cross-port
  audits in frozen order.
- Schema key indices `0..71`, unresolved `-1`, source codes `0..3`, recall field order,
  identity, scope, and the 17 hard-count serialization order are frozen.
- Returned prediction row `ell` is mathematical lead state `ell+1`; goal and evaluator
  arrays use the same convention.
- `U_s(H)` reads `a[k,ell]`, and validity auditing uses `ell mod 12`. All exact metric
  denominators remain consistent.

### Dream/lesion execution counts — PASS

- Every condition performs exactly one 288-pair constrained-or-shadow enumeration,
  288 checks, 24 join candidates, 48 endpoint values, 24 output/update slots, and one
  dream pass.
- The common lesion audit is a post-classification of already enumerated indices. It
  adds no enumeration, join, schema proposal/update, or dream pass.
- Every condition records lesion counts `240,24,216`; the separate lesion buffer freezes
  3 valid-missing and 21 invalid cross-port entries, hence `21/24=0.875`.
- Constrained pair reasons are not overwritten. Shadow lesion bytes are hashed but do
  not enter schema/action lookup, rollout, selection, constrained provenance, or a
  non-lesion condition's scientific invalid-splice metric.

### Budget and allocation — PASS

- `b_registered` contains exactly 29 fields; each actual/registered int64 vector is
  `29*8=232` bytes.
- Ordered allocation subtotal before padding is `312,584` bytes.
- Adding `(80632,) uint8` padding gives exactly `393,216` bytes.
- Persistent cap is `524,288`; exact headroom is `131,072` bytes.
- No second pair-reason array or extra pass was introduced, so the frozen allocation and
  29-vector remain unchanged.

### Calibration and split chronology — PASS

- Calibration populations remain exact: core transitions `46,080`, state rows `49,920`,
  codec records `3,840`, positive/lure queries `960/960`, residual endpoint values
  `3,840`.
- Implementation lock precedes the single train opening. Validation/test material is not
  generated during pre-lock or train. Test unlock requires a committed, clean,
  byte-identical passing validation artifact and recomputed dependency checks. Any split
  failure forbids the next split without resampling.

## Disposition

All previously identified executable consistency defects are closed in the contract.
The document is mechanically ready for implementation lock and handcrafted off-range
tests. This PASS is a static contract-consistency result only; it is not empirical
support and does not authorize opening a registered seed before the implementation and
integrity locks exist.

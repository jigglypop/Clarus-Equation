# G9-CBM V2 final mechanical-consistency audit

Status: COMPLETE

Gate: **REVISE**

## 1. Scope

This is a static cross-audit of `revisions/00-contract-v2-draft.md` against
`revisions/11-math-v2.md`. No code, development seed, registered seed, calibration,
validation, or test was run.

## 2. Items that are mechanically closed

- The learned core now has five coefficients per target, four targets, `P_core=20`,
  and `N_solve=4`. Codec residuals, rollout, and the frozen-core control include
  `c_hat`.
- Episodic traces and accepted recall reconstructions are raw `(12,8)` codec arrays;
  schema/dream payloads are standardized and are inverse-standardized before rollout.
  `q_hat` is formed in raw residual coordinates and never reads fingerprint columns.
- Dream joins use only four standardized residual columns and two endpoint RMS values
  per missing key.
- The evaluation prefix has 12 transitions, records `x[0]..x[12]`, and correctly makes
  the first future lead phase zero.
- The stated scalar denominators are arithmetically correct:
  `E_all20=11,520`, `E_all5=2,880`, `E_uv20=3,840`, `E_uv5=960`, recall hidden
  coordinates `=1,728`, and valid-transition decisions `=2,880`.
- The constrained enumeration arithmetic is correct:
  `288` ordered pairs, `72` same-component pairs, `48` observed pairs, `24` valid
  missing pairs, `24` join candidates, and `48` scalar endpoint joins.
- The lesion arithmetic is correct: `288-48=240`, then `24` accepted audit slots and
  `216` capacity-padding rejections.
- The calibration populations are mutually consistent: core `46,080`, state rows
  `49,920`, codec trajectories `3,840`, positive/lure recall queries `960/960`, and
  residual endpoint joins `3,840`.
- `b_registered` currently has exactly **29** fields; `(29)` int64 is therefore
  `232` bytes. The listed byte values sum to `393,216` only under the intended
  one-dimensional 80,608-byte padding interpretation, and the resulting total is below
  the `524,288` persistent cap by `131,072` bytes.

## 3. Required exact revisions

### P0-1 — truth-prefix wording and public-goal anchor/index are inconsistent

Section 3.3 says `f_hat` is used by “prefix generation,” while Section 4.2 correctly
generates the true evaluation prefix with the registered structural equation. Section
5.2 then starts the goal at undefined `x_0`, although prediction starts at `x[12]`, and
uses the ambiguous shape/index `g[1:20,4]` while the typed request says `(20,4)`.

Replace those sentences by:

```text
The fitted expression, including c_hat, is used for every wake/evaluation codec
residualization, every candidate rollout, and every learned-core control. It is never
used to generate evaluator-truth prefix or future states.

x_origin := x_prefix[12]. For lead ell=0..19, generate the public reference from
x_goal[0]=x_origin and store g[ell,:]=x_goal[ell+1]. Thus g has exact shape (20,4).
The reference uses literal D,B,G, action A[goal_id], and q=b=eta=0.
```

Add `mu_x` and `sigma_x` to an explicit typed `CostSpecV2` inside
`CandidateRequestV2`; “cost literals” alone does not provide the arrays needed to
compute `J_hat`.

### P0-2 — two raw/standardized boundary symbols remain wrong

`T_std` is said to be used “inside storage,” but inherited episodic storage owns raw
`EpisodicRecord.trajectory`. Also `codec_train_scale` is undefined; the frozen name is
`sigma_codec`.

Replace the relevant text by:

```text
The episodic store owns T_raw. T_std is a derived read-only view used by masked-cosine
scoring and by the V2 schema/dream adapter; it is not the stored episodic payload.

For masked cell (r,h), cue_noise[r,h] =
0.01 * reshape(sigma_codec,(12,8))[r,h] * Normal(0,1).
```

Replace “imported component-local schema fallback” with “the residual-only V2
adapter's component-local mean fallback”; the contract explicitly forbids calling the
inherited eight-column schema/join path.

### P0-3 — the typed action/return schema is not byte-complete

The 72 schema-key indices and `uint8` source codes have no frozen ordering/codebook,
and sequence-level invalid-cost aggregation is implicit. Exact fix:

```text
Schema keys use the frozen first-occurrence ordinal order
(context, component/port, prefix-local, suffix-local), flattened in that order to
int16 0..71; unresolved is -1.

schema_source uint8: 0=unresolved, 1=observed_real, 2=synthetic_hypothetical,
3=component_fallback. Values 4..255 are invalid.

inferred_sequence_valid[k] = all(inferred_valid[k,0:20]).
J_hat(k)=10000 iff inferred_sequence_valid[k] is false; otherwise use the registered
state/action cost.
```

Freeze the exact field order/dtype of the “fixed-shape recall/provenance audits” rather
than referring to them generically. The allocation ledger implies 72 entries of
`accepted:bool8`, `identity:int16`, `confidence:float64`, and `scope:uint8`; the typed
return must say exactly that and name the 17 hard-count fields in their serialized order.

### P0-4 — `U_s` has an action-index off-by-one and one denominator contradiction

The current `t<=H` expression can index action `a[k,20]` and does not define whether
`U_s` is H-dependent. Replace both error definitions by one lead convention:

```text
ell = 0..H-1; compare x_hat[o,k,ell+1] with x_star[o,k,ell+1].
U_s(H) = {(o,k,ell,j): k<6, ell<H,
          (i(o),j(a[k,ell])) in M, j in 0..3}.
```

The fixed denominators remain `3,840` and `960`.

`wrong_given_accept` uses `max(accepted,1)`, while Section 8.2 says no
zero-denominator convention is permitted. Keep the inherited totalization and replace
the latter sentence by:

```text
No zero-denominator convention is permitted except the explicitly registered
wrong_given_accept denominator max(accepted_positives,1); its coverage and identity
gates remain mandatory.
```

### P0-5 — the two distinct `216` counters and lesion composition must be frozen

For the constrained pass, `288-72=216` is the component/port rejection count. For the
lesion, `240-24=216` is the unrelated `capacity_padding` count. They must not share one
counter or reason code.

Under the contract's displayed `O` order and frozen first-occurrence traversal, the
first 24 non-observed lesion pairs contain exactly **3** same-port valid-missing pairs
and **21** cross-port invalid pairs. Add hard assertions:

```text
N_constrained_component_port_rejections = 216
N_lesion_capacity_padding = 216
N_lesion_valid_missing_accepted = 3
N_lesion_invalid_cross_port_accepted = 21
invalid_splice_rate(unconstrained) = 21/24 = 0.875
```

Every active, inactive, and lesion condition must compute exactly 48 residual endpoint
join scalars; the lesion ignores their threshold for acceptance but may not omit the
registered computations.

### P0-6 — the allocation ledger has one invalid shape and one undefined counter schema

`(80,608)` is a two-dimensional NumPy shape with 48,640 uint8 elements, not a
one-dimensional 80,608-byte payload. In addition, `operation counters (32)` has no
32-name schema and does not match the 29-field registered vector.

Use this exact correction while preserving the total:

```text
actual budget vector     | (29,) int64 | 232
registered budget vector | (29,) int64 | 232
fixed inactive/padding   | (80632,) uint8 | 80,632
```

Remove the undefined `(32)` operation-counter row. Hash-lock the 29 names/order for
both actual and registered vectors. With this replacement, the table still totals
exactly `393,216` bytes and remains below the `524,288` cap.

## 4. Final disposition

The intercept, residual-only join, evaluation-prefix phase, primary denominator
arithmetic, calibration populations, and headline budget arithmetic are closed. The six
items above are executable schema/index/allocation contradictions, so implementation
lock or train opening is not yet permitted. After textual repair and a fresh static
recount, this audit can move to `PASS` without running any seed.

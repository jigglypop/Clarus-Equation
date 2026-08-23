# G2 fixed-weight compressed metric-feature utility contract

Status: COMPLETE v4

## Claim boundary and no-go theorem

With `C=B B^T+lambda I` and `g=C^-1`, `g` is a deterministic reparameterization of `C` and a
deterministic function of `B`. Also, `B` and `-B` give the same `C,g` but opposite signed endpoints.
G2 therefore cannot show new information in `g`, unique geometry, or metric-only signed dynamics.
Its maximum claim is only:

> In the declared opt-in all-active BrainRuntime evaluation regime and chart, one precommitted compressed
> quadratic metric feature has held-out utility for a scalar path-access outcome over the complete
> frozen adverse family.

Even a pass is an inductive-bias/compression result, not `g-to-x` causation, mediation, physical
cortical geometry, biology, or consciousness.

## Runtime, environments, and independent units

Development seeds are `97601..97616`; confirmation seeds `99601..99632` remain sealed unless
development passes every gate and a hash-bound freeze manifest is written.

Each seed uses the G1 algorithm for seed-specific Gaussian `W0`, disjoint `S/T/N`, injection `U`, and
chart `F=U^T`, but a dedicated `G2Config`/fixture sets `active_threshold=0` and never calls or aliases
G1 `_frozen_protocol`. G2 never changes `W`. Torch CPU, no delay, Dale transform, automatic STDP,
hippocampal write/row, or replay; `active_ratio=1`.

BrainRuntime receives one default-off evaluation switch,
`force_all_active_selection: bool = False`. The default path and all predecessor behavior remain
unchanged. When true, `_select_active` returns the all-true mask after every native step. G2 alone
sets it true, and its fixture must assert `active_ratio=1`. The exact 48-bit mask is still logged
after the pulse and every free tick; any count other than 48 makes the circuit ineligible and forces
route `STOP`. This is an explicit simulator intervention, not a claim about natural cortical
recruitment, and all conclusions are restricted to this all-active evaluation regime.

Frozen environments `(external_gain,noise_sigma)` are
`(.40,0),(.40,.02),(.60,0),(.60,.02)`.
Every episode restores its environment snapshot and all transients. It then sets only
`runtime.step_index` to a collision-free native-noise start. For environment index `e` and local ID
`l` (calibration axes `0..2`, fit rows `3..18`, test rows `19..26`):

`global_id = seed*512 + e*128 + l`, `start = 8*global_id`.

An episode consumes `start..start+6`; stride 8 makes intervals disjoint across circuits,
environments, splits, and rows and keeps starts below `2e9`. Calibration `+/-` for one axis
deliberately shares an interval as common random numbers. The full interval schedule is
pairwise-checked and hash-logged. All models read the same realized rows. The seed, never a pulse or
tick, is the statistical unit.

## Calibration, fit, and held-out splits

For each seed/environment, six `+/-0.5 U e_j` axis probes followed by six zero-input WAKE ticks form
calibration only. Central differences at every free horizon give `B_h`, `h=1..6`; define
`B=B_6`, `C=B B^T+1e-3 I`, `g=C^-1`. Numeric G1 matrices are never reused.

A global float64 Torch generator seeded `97599` draws and unit-normalizes standard-normal
three-vectors. A row is redrawn if nonfinite/zero, if `max_j abs(q_j)>0.95` (too near an axis), or if
its absolute dot product with an earlier accepted row is at least `1-1e-10` (duplicate up to sign).
The first 64 rows are fit and the final 32 held out. `row_index mod 4` assigns environments, yielding
16/8 rows per environment. Within each environment's deterministic local ordering, fit amplitudes
alternate `.60,.75` and test amplitudes `.90,1.05`.
Direction, amplitude, split, and `(q,a)` byte hashes are logged and mutually disjoint.

An episode applies `a U q` at `t=0`, records present state `y0=F x0`, then records `y1..y6` after six
zero-input WAKE ticks. No future state enters a feature. The sole primary target is the nonmetric
path-access magnitude `L=mean_{h=1..6} abs(y_h[T])`. First passage is descriptive.

## Calibration-only features

Let `u:=y0` only; its pre-free-tick state/hash is the provenance of every feature.

- `m_g=u^T g u`;
- `m_C=u^T C u`;
- `z_Cterms=[C11*u1^2,C22*u2^2,C33*u3^2,2*C12*u1*u2,2*C13*u1*u3,2*C23*u2*u3]`;
- `z_Craw=[C11,C22,C33,C12,C13,C23]`;
- `m_E=u^T u`;
- `m_perm=u^T(Pi^T g Pi)u`, with `Pi` swapping named `S,T` axes;
- `m_Bpath=mean_h abs((B_h(aq))[T])`, a horizon-matched direct prediction;
- `Q_raw=(mean_r y_r y_r^T+1e-3 I)^-1` from the six signed `h=6` calibration endpoints, and
  `m_Qraw=u^T Q_raw u`.

`Q_raw` is raw endpoint precision, not an unrestricted/optimized SPD. `z_Cterms` only decomposes
the selected scalar `m_C`; it is not called all raw-C information. `z_Craw` separately supplies all
six independent raw-C entries.

For the no-repackaging invariant, independently copy `C`, recompute `g_from_C` with float64 CPU
`torch.linalg.inv`, build a non-aliased feature array, and independently fit the identical
standardizer/ridge on the same rows. Raw and standardized feature residuals must be `<=1e-10`, and
held-out prediction residual `<=1e-8`.

The chart audit rebuilds `B'=PB`, `R0'=P I P^T`, `C'=B'B'^T+lambda R0'`, `g'=(C')^-1`; with
`u'=Pu`, require `u'^Tg'u'=u^Tgu` within `1e-6` for every row.

## Predictors, budgets, and score

All heads use the same pooled 64 fit rows, fit-only standardization, scalar ridge `1e-4`, unpenalized
intercept, deterministic float64 solve, and frozen preprocessing. Base direct features are
`d=[u_S,u_T,u_N,external_gain,noise_sigma]`.

- `D=[1,d]`: 6 coefficients;
- `D+g`, `D+C`, `D+E`, `D+perm`, `D+Bpath`, `D+Qraw`: `[1,d,m]`, 7 each;
- `D+Cterms=[1,d,z_Cterms]` and `D+Craw=[1,d,z_Craw]`: 12 each;
- `D+g_from_C`: independent duplicate for the invariant only;
- `D2=[1,d]` plus all 15 upper-triangular products of standardized `d`: 21 coefficients;
- persistence `abs(u_T)`: zero coefficients;
- fit-row global mean: one coefficient;
- `raw_Bpath=m_Bpath`: zero coefficients.

`D+Bpath/raw_Bpath` directly see known `(a,q)` and are deliberately stronger causal baselines.
Only the six one-scalar-added heads have equal trainable readout budget; `Cterms`, `Craw`, `D2`, raw
`B_h`, and `Q_raw` are adverse models with explicitly different information/capacity.

The frozen scalar fixed-variance Gaussian score uses `s2=1e-4`:
`loss=.5*log(2*pi*s2)+(L-mu)^2/(2*s2)`. It is a proper score for that fixed family and equivalent to
scaled squared error for ranking, not an uncertainty-calibration claim. MSE is secondary.

## Integrity and simultaneous decision gate

A required fixture test proves G2 does not call/alias G1 `_frozen_protocol`, the new selection flag
defaults false in ordinary runtimes, snapshot/restore preserves it, and only the G2 fixture sets it
true. Every selected seed must
preserve exact W hash, dense/CSR equality, zero rows/STDP updates, finite state, 48/48 masks at all
seven ticks, fixed counts/order, disjoint noise intervals, and reset parity. All models see identical
rows; any model-specific exclusion makes the circuit ineligible. Schema, codebook, split,
standardizer, ridge, transform, noise schedule, and source hashes are recorded.

For adverse comparator `b`, define per seed
`Delta_i^b=mean_test(loss_b)-mean_test(loss_D+g)`. The family is
`D,D+C,D+E,D+perm,D+Bpath,D+Qraw,D+Cterms,D+Craw,D2,persistence,global_mean,raw_Bpath`.
Define one simultaneous contrast `Delta_i^min=min_b Delta_i^b`.

Development/confirmation GO requires:

1. every selected seed passes all integrity and no-repackaging invariants;
2. `Delta_i^min>0` on at least 80% of circuits;
3. a positive one-sided percentile-bootstrap 95% lower bound: resample all `n` seed units with
   replacement 10,000 times using generator seed `97598`, compute each mean `Delta^min`, and take
   the empirical `0.05` quantile;
4. confirmation independently repeats the frozen development decision.

`D+perm` is an ordinary member of the worst-adversary statistic and ties fail. Any nonfinite score
or eligibility failure forces `STOP`. Failure rejects only the chosen compressed metric-feature
utility in this experiment, not the algebraic SPD summary.

## Retired smoke quarantine

The pre-v4 apparatus smoke used seed `97501` under the earlier threshold-zero/no-force code and
source SHA-256
`1f1c46224f9100106dd9a4b44c624ae151485dc5a078a93c8ae62acdeb82efbe`. Its config otherwise matched
G2 v3. No artifact was written, but stdout exposed integrity failures and model losses. That entire
old development block `97501..97516` is retired and absent from both new seed ranges. Those exposed
losses may not tune features, comparator family, ridge, score, effect gate, or v4 parameters; v4
changes only the independently predicted active-mask apparatus defect.

Before execution, direct semantic tests must show: flag true returns a bool all-true mask for
arbitrary including all-negative salience before budget/eligibility logic; flag false is bit-for-bit
equal to the legacy selection result on a fixed fixture; the flag participates in config/source
hashes and survives snapshot round-trip.

# G3-D randomized-contingency response/recall diagnostic contract

Status: FROZEN — apparatus amendment and stable implementation audits PASS.

## Permanent identification boundary

Original G3 learned mediation is BLOCKED. A learning contingency changes recurrent `W`; both a
post-learning response summary `S(W)` and recall `R(W)` are downstream functions of that same `W`.
Randomizing contingency can identify only a joint `Z -> {S,R}` effect. Correlation, regression, or
cross-fitting cannot identify `S -> R`, mediation, or `g-to-x` without an independent intervention on
the summary that holds every other recall-relevant determinant fixed. G1 and G2 STOP additionally
remove any claim that the inherited response precision was validated or predictively sufficient.

G3-D is therefore a diagnostic falsification only. Its maximum positive statement is:

> In frozen M1 BrainRuntime circuits, the correct cue/value condition had a larger descriptive SPD
> response change and continuous zero-store recall change than every predeclared adverse condition,
> and the same-arm seed-level contrasts co-varied under this protocol.

Every outcome remains BLOCKED for mediation, unique metric, physical brain geometry, biological
memory, selfhood, or consciousness.

## Frozen predecessor and seeds

Only confirmed M1 fixed-clock delayed three-factor binding is used. Frozen source:
`runtime_alternative_memory.py`, SHA-256
`be708bac30bb4e7e681990f838159e70efb9ed36061cef602771e86c8248c27a`.
Its full default config remains `dim=48`, 12 epochs, 3 replay ticks, cue drive 5, `m1_lr=.8`, trace
decay `.95`, eligibility decay `.99`, LTP 1, LTD .2, max write 5, and threshold .20. G3-D does not
retune M1 or use M0/M2/M3.

The first development block `97701..97716` is retired as `APPARATUS_INVALID`: all scientific and
integrity outputs were inspected, and all 256 lesion installs had float32 reconstruction residuals
in `[1.64868e-7,1.88207e-7]`, above the frozen `1e-7` gate. It is never reused. Fresh replacement
development seeds are `97801..97816`; confirmation `99701..99732` remains unopened and sealed unless
the replacement diagnostic passes and a source/artifact manifest freezes. A circuit seed is the
sole unit. Public single-seed and range APIs reject any overlap with the retired block before
computation.

The G3 implementation duplicates the frozen M1 training body in a new module because the frozen
function does not return W snapshots. A parity test on excluded smoke seed `97699`, using the full
frozen 48-dimensional/12-epoch/3-replay-tick configuration, must reproduce the original condition's
W drift, association contrast, schedule counts, every continuous/discrete recall audit, and final W
hash exactly. The predecessor file itself is not edited.

## Learning and structural arms

Every seed starts from the identical zero-W M1 snapshot, codebooks, source manifest, episode order,
reset schedule, and block-end clock. Required arms are:

- `matched`: exact frozen `fixed_clock` pairing;
- `target_shuffled`: exact frozen cyclic target derangement;
- `no_replay`: exact frozen no-replay schedule;
- `weight_permuted`: run the exact zero-gate schedule on a fresh branch, then install
  `P W_matched P^T`, where `P` is the seed-fixed coordinate permutation from CPU generator
  `seed+97901`.

The last arm is a post-learning structural lesion derived from the matched arm, not a randomized
learning-contingency arm. It preserves matched W's Frobenius norm, singular spectrum, density,
diagonal-zero property, and multisets of row/column norms while breaking its relation to the
unpermuted codebook. Its zero-gate runtime must be restored independently, share no tensor/storage
alias with the matched runtime, and receive the additive delta `P W_matched P^T-W_zero_gate` exactly
once through the public bounded install with bound `10`. Any clipping, or
`||W_applied-P W_matched P^T||F>1e-7`, is integrity STOP. Log P, its hash, initial/final W hashes,
the requested/applied delta hashes and norms, and the no-alias check. All arms audit equal
event/reset/tick/block counts, finite W, exact applied W, and dense/CSR parity.

## Independent response procedure

The physical probe matrix `U0` is fixed for all seeds and arms before outcomes. Its three columns
are unit constant vectors on contiguous coordinate blocks `0:16`, `16:32`, `32:48`; `F0=U0^T`.
Its construction reads no cue, target, codebook, arm, W, decoder, or outcome, and its bytes are
hash-logged.

Pre-learning and each post-learning arm are physically sealed: replace hippocampus with an empty
object, disable writes, clear the temporal audit store, and reset all transients. Each calibration
probe restores a fresh sealed snapshot. For each axis, apply paired `+/-5 U0 e_j` for one WAKE tick,
then six zero-input WAKE ticks. Common deterministic zero-noise dynamics and matched ordering apply
to every arm. Record central-difference matrices `B_h`, `h=1..6`, and

`C = B_6 B_6^T + 1e-3 I`.

`C` is called only a descriptive SPD response summary. The primary response-change scalar is

`S_arm = AIRM(C_pre,C_arm)`.

Computationally, all matrices are float64 and

`AIRM(A,B) = ||log(A^(-1/2) B A^(-1/2))||F
           = sqrt(sum_k log(lambda_k)^2)`.

Construct `A^(-1/2)` by a symmetric eigendecomposition, symmetrize the whitened matrix once as
`(M+M^T)/2`, and obtain `lambda_k` with `eigvalsh`. Every input and generalized eigenvalue must be
finite and strictly greater than `1e-12`; otherwise integrity is STOP. No eigenvalue clamp is used.

`logdet(C)` and `||B_h-B_h_pre||F` are descriptive. There is no `g=C^-1` result and no transform or
predictive-utility inheritance. Calibration integrity requires zero stores, unchanged W, no STDP,
finite SPD eigenvalues, pulse active count exactly the frozen M1 WAKE budget 12, and nonzero `B_h`.

## Independent continuous recall

Recall uses separately restored sealed snapshots and never a calibration state. For each trained
source index `k`, run the frozen M1 cue plus six zero-input WAKE ticks for both clean cue and the
frozen 15% corruption (first coordinates zeroed). Define each continuous target margin

`r_k = cos(unit(x_H),v_k) - mean_{ell != k} cos(unit(x_H),v_ell)`

against the original unshuffled mapping. `R_arm` is the mean over source indices and both cue
conditions. Compute the common `R_pre` from zero W. Decoder labels and abstention thresholds never
enter `R`. Every probe must keep hippocampal/temporal rows zero, W unchanged, and use a fresh reset.

## Calibration-null lesion falsifier

Starting only from the sealed matched post-W snapshot, generate eight seed-fixed Gaussian dense
directions with CPU generator `seed+97999`, zero their diagonals, normalize each to Frobenius `.25`,
and include both signs for a 16-candidate bank. For each intended signed direction `d*`, construct
the native-representable target `W_target=fl32(W_matched+d*)`, then define the actual candidate
`d=W_target-W_matched` in float32. Require `| ||d||F-.25 |<=1e-6` and
`||d-d*||F<=1e-6`; log both hashes/norms. This is a numerical apparatus repair, not a response or
recall-dependent choice. For every candidate, independently restore that exact sealed snapshot,
require `||d||F<=.250001`, and install precisely one `d` through the bounded additive native install
with the fixed numerical-headroom bound `.250001`, calibrate
from separately restored sealed copies, and discard the branch. Candidates never accumulate. Any
clipping or `||W_applied-W_target||F>1e-7` is integrity STOP; never run recall during selection.

Select the first minimum of

`q_j = ||stack(B_1^j,...,B_6^j)-stack(B_1^M,...,B_6^M)||F
       / (||stack(B_1^M,...,B_6^M)||F+1e-12)`.

`stack` concatenates the six 3-by-3 matrices in increasing-horizon order. Select the first minimum
`q_j` in the fixed generation/sign order. Only after selection, score its fresh zero-store
continuous recall. A calibration-null falsifier is found if `q_j<=.02`,
`AIRM(C_j,C_matched)<=.02`, and `abs(R_j-R_matched)>=.05`. Any found falsifier forces diagnostic
STOP and shows that the declared response summary misses recall-relevant W directions. Failure to
find one is not sufficiency evidence.

## Seed-level contrasts and decision

For adverse conditions `b in {target_shuffled,no_replay,weight_permuted}`, define same-arm
contrasts

`DeltaS_i^b = S_i^matched-S_i^b`,
`DeltaR_i^b = (R_i^matched-R_i^pre)-(R_i^b-R_i^pre) = R_i^matched-R_i^b`.

Define `DeltaSmin_i=min_b DeltaS_i^b` and `DeltaRmin_i=min_b DeltaR_i^b` for simultaneous directional
gates only. The association endpoints are ordinary same-condition Pearson correlations
`rho_b=corr_i(DeltaS_i^b,DeltaR_i^b)`; contrasts from different adverse conditions are never paired.
If either same-condition vector is constant or nonfinite, status is `UNINFORMATIVE_STOP`. Bootstrap
entire paired seed rows 10,000 times with generator `97898`; each resample computes every
condition-specific mean contrast and `rho_b`, then records the minimum across the frozen condition
family. Zero-variance resamples receive that condition's rho `-1`. One-sided simultaneous 95% lower
bounds are empirical `.05` quantiles of these resampled minima.

`DIAGNOSTIC_PASS` requires:

1. every seed and arm passes all learning, cutoff, calibration, recall, W, source-hash, and schedule
   invariants;
2. both `DeltaSmin>0` and `DeltaRmin>0` in at least 80% of circuits;
3. simultaneous lower bounds for every condition-specific mean DeltaS, mean DeltaR, and same-arm
   rho are all positive;
4. no calibration-null falsifier occurs in any circuit;
5. sealed confirmation independently repeats the frozen diagnostic.

Ties, nonfinite values, any matched control winning, or any model-specific exclusion are STOP.
Regardless of `DIAGNOSTIC_PASS`, the separate field `mediation_status` is always
`BLOCKED_NOT_IDENTIFIED`.

Before any group bootstrap, the stage validator requires exactly the ordered, unique replacement
development seed block `97801..97816`, or exactly the ordered, unique confirmation block
`99701..99732` after a
frozen development-pass manifest. Duplicate, missing, foreign, mixed-stage, mixed-config,
mixed-source, mixed-schema, or mixed-probe rows are rejected rather than scored.

Both the library execution and summary entrypoints reject confirmation without that manifest. The
verifier parses the referenced development JSON and requires the G3-D schema, development mode,
exact development block/count, canonical result hash, current frozen source-hash map, a reproduced
summary, and `DIAGNOSTIC_PASS`; the manifest must bind both the whole-artifact hash and canonical
result hash. A self-declared verdict or filename/hash pair is insufficient.

The public single-seed and arbitrary-range diagnostic APIs reject every official confirmation seed
before computation. Only the manifest-verified stage entrypoint may call the private unchecked
runner for `99701..99732`.

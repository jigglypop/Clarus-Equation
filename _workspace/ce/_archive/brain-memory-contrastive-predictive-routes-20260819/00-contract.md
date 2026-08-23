# Contrastive and predictive recurrent-memory routes

Status: COMPLETE

PREDECESSOR: `_workspace/ce/brain-mechanism-alternative-routes-20260819`

## Scope

This light continuation preserves the predecessor's frozen M0/M1 binding evidence and opens a fresh
blind seed partition for the remaining learning questions. It tests exactly three items:

1. **T1 -- M1 factor transfer.** Reuse the frozen M1 learner, hyperparameters, fixed clock, native
   inter-phase reset, decoder threshold, and controls without revision. Train factorized combinations
   `(0,0)`, `(0,1)`, `(1,0)` and evaluate only held-out `(1,1)` after physical store cutoff.
2. **M2 -- lagged contrastive phase learning.** From the same base snapshot, collect a positive
   cue-to-replayed-target phase and a negative cue-to-zero phase. Apply one bounded update from the
   difference of row-post/column-pre lag correlations. Test binding first and factor transfer second.
3. **M3 -- teacher-forced replay-state residual with cached credit.** Fit one circuit-local linear
   predictor on exactly 64 adjacent native transitions and freeze it. During replay, compute error
   only from the true immediate predecessor, then combine the first residual with a separately named
   cached cue-credit trace. Test held-out transition prediction before binding and factor transfer.

No M0 rank search, M1 binding retuning, geometry, metacognitive control, SCC lesion, or biological
claim is part of this continuation.

## Seed and freeze boundary

- Development seeds: `97301..97316`.
- Confirmation seeds: `99301..99332`; they remain unopened until source, tests, endpoints,
  thresholds, controls, and a manifest are frozen.
- The predecessor's opened `99201..99232` seeds are prohibited for development or confirmation.
- The independent statistical unit is a seeded circuit. Cues, coordinates, neurons, and ticks are
  repeated measurements only.

## Common native evaluation

- Torch CPU, zero runtime noise, `axon_delay=False`, `dale_law=False`.
- Seed-fixed independent cue/value codebooks contain no pairing information.
- All scored states come from `BrainRuntime.activation` after cue plus six zero-input WAKE steps.
- Temporal memory and `HippocampusMemory` are physically replaced, hippocampal writes disabled,
  transient dynamics reset, and zero rows audited after every rollout.
- The decoder receives only final activation and the precommitted value codebook. Target, target ID,
  decoder score, old activation, goal, memory context, replay row, and hidden condition are forbidden
  evaluation features.
- Every arm starts from an identical snapshot, uses equal event/tick counts, and logs the actual
  applied dense-weight delta plus dense/sparse parity.

## T1 frozen factor protocol

- Factor codewords occupy disjoint cue-A, cue-B, value-A, and value-B quarters. Training presents the
  three complete combinations; `(1,1)` appears in neither cue nor value form during learning.
- The unchanged M1 signed eligibility object is the only phase bridge. Every cue/value phase is
  separated by a native transient reset; the fixed target-blind clock is `+1.0` at block end.
- M1 parameters remain: 12 epochs, 3 replay ticks, learning rate `0.8`, trace decay `0.95`,
  eligibility decay `0.99`, LTP `1.0`, LTD `0.20`, max installed norm `5.0`, decoder threshold `0.20`.
- Controls: zero clock, sign flip, time reversal, eligibility reset, no replay, target assignment
  shuffle. No M1 parameter or threshold revision is allowed even on development.
- Because the observed combinations expose factor value 0 twice and factor value 1 once, T1's
  maximum claim is transfer under this declared imbalanced schedule. Report a non-gating
  factor-frequency sensitivity and decoder-only/chance baseline; do not call it general composition.

## M2 frozen update

M2 begins from a deterministic projection fixed point. A diagonal-zero Gaussian draw fixes each
row's support and signs. The 16 largest off-diagonal entries receive signed amplitude `0.25`
(`4` entries of amplitude `0.5` in dimensions below 17), giving an exactly unit-norm binary row.
Apply `structural_projection` with `density=1.0`, `theta_on=1e-6`, and `theta_off=5e-7`, and verify a
second application changes it by at most `1e-7` in Frobenius norm. This verified matrix is the common
pre-block snapshot for every M2 arm, so a zero raw contrast has a zero proposed and applied delta.

This exact-amplitude construction is implementation revision 2. Repeated float32 normalization of a
dense projected Gaussian entered one-ulp cycles above the frozen aggregate tolerance for three
development seeds. The tolerance was not relaxed and no task outcome had been produced when the
construction was changed.

For one block, let the positive lag correlation be

$$
C^+=\frac1{T_+}\sum_{t=1}^{T_+}a_t^+(a_{t-1}^+)^{\mathsf T},
$$

where an external collector sequence consists of a cached cue followed, after a native transient
reset, by three replayed-target states. This is a virtual collector lag, not an uninterrupted native
runtime transition. The negative phase uses the same cached cue, reset, and three zero-replay ticks:

$$
C^-=\frac1{T_-}\sum_{t=1}^{T_-}a_t^-(a_{t-1}^-)^{\mathsf T}.
$$

There are exactly three lag terms: cached cue to first post-reset state, then two adjacent target or
zero-state transitions. The desired weight is

$$
W' = \Pi\left(W+0.8(C^+-C^-)\right),
$$

The implementation computes `raw=0.8(C+ - C-)`,
`proposed=structural_projection(W + raw)`, zeros the diagonal, forms the additive
`delta=proposed-W`, clips that delta to Frobenius norm `5.0`, and passes only `delta` to the native
install before CSR rebuild. Raw, proposed, and actual applied deltas are separately audited. M2 uses
decoder threshold `0.20`.
Controls are no-write, target shuffle, identical positive/negative phase, positive-only,
negative-only, and sign-reversed phase difference. Each uses identical block/tick counts.
Positive-only and negative-only still execute both phases and only zero the omitted collector term;
they are mechanism ablations rather than norm-matched adverse controls. Target shuffle changes only
positive-phase target assignment. Every arm logs raw, proposed, clipped, and actual applied deltas.

M2 is explicitly supervised contrastive phase learning because the positive training phase contains
the target. It is not STDP or biological sleep consolidation.

## M3 frozen predictor and update

- A single feature map is used in fitting, held-out scoring, and every replay residual. It concatenates
  activation, refractory, memory trace, adaptation, STP-u, STP-x, bitfield, lifecycle,
  inactive-steps, goal, the exact external-drive vector, the exact effective replay-drive vector
  `replay_mix(mode) * replay`, forced-mode one-hot, replay-present scalar, and bias. Its dimension is
  `12*d + 5`. Axon delay and automatic STDP are disabled; noise is zero.
- Each circuit fits deterministic ridge `1e-4` on exactly 64 adjacent transitions. Training actions
  are `[0,1,2,3] * 16` at unit drive and forced modes repeat `[WAKE,NREM,REM,WAKE]`. Output is the
  actual next native activation from `BrainRuntime.step`. Even rows contain no replay; odd rows use
  one of four seed-fixed orthonormal replay calibration vectors generated independently from the task
  cue/value codebooks. A single explicit hippocampal row makes the effective replay vector exactly
  known. Calibration contains no cue--value association and spans the unit-norm replay inputs and
  modes later used by learning.
- Held-out predictor scoring uses 16 independent forks of one frozen pre-score snapshot, reversed
  actions `[3,2,1,0] * 4`, the reversed mode schedule, unseen external drive `1.5`, and a separate
  seed-fixed set of independent replay calibration vectors on odd rows. The model freezes before
  those scores and all writes. Score rows never become write rows.
- Association learning begins from a fresh copy of the pre-fit recurrent snapshot. A cue phase stores
  a separately named cached credit vector
  $q_{\mathrm{credit}}=\operatorname{unit}(a_{\mathrm{cue}})$, then native transients reset. Its norm
  and exact bytes are audited; no other normalization is allowed. For
  replay tick 1, $s_0$ is the actual reset state immediately before the step and $a_1$ its actual
  replay-driven next activation:

  $$
  e_1=a_1-f_\theta(\phi(s_0,0,r_{\mathrm{eff},1},\mathrm{NREM},1)).
  $$

  The cached cue is not the predictor predecessor. For ticks 2 and 3, $s_{t-1}$ is the true immediate
  native predecessor and

  $$
  e_t=a_t-f_\theta(\phi(s_{t-1},0,r_{\mathrm{eff},t},\mathrm{NREM},1)),
  \qquad
  p_{t-1}=u_{t-1}\odot x_{t-1}\odot a_{t-1}\odot
  \mathbf 1_{\mathrm{active},t-1}.
  $$

  The exact block update is

  $$
  \Delta W=\frac{0.8}{3}\left[e_1q_{\mathrm{credit}}^{\mathsf T}
  +e_2p_1^{\mathsf T}+e_3p_2^{\mathsf T}\right].
  $$

  One structural projection and bounded additive install occurs at block end. Replay features use
  external drive zero, the exact continuous effective replay drive, forced NREM, and replay-present
  `1`; no action, cue, or target ID enters $\phi$. Symbolic target ID and decoder outputs never enter
  the update.
- Controls: predictor-only/no-write, transition-order shuffle, one-block delayed error, sign-flipped
  error, no replay, and target-assignment shuffle. M3 decoder threshold is `0.20`.

M3 is a supervised teacher-forced replay-state residual rule. The injected continuous replay target
is both a predictor input and a teacher signal even though no symbolic ID enters the update. Because the nonlinear runtime
predictor is not the Jacobian of recurrent weight, this outer product is a heuristic feedback rule,
not a proven gradient of runtime prediction loss.

## Decision gates

### Predictor gate

M3 held-out activation MSE must be no greater than `0.90` times persistence MSE on at least 80% of
circuits. Predictor parameters, training row count, freeze status, and absence of post-score refitting
must be audited. Passing this gate alone proves only bounded transition prediction.

Every M3 write arm must also reconstruct the exact three-term total above, apply a finite nonzero
delta only at block end, and match event, drive, mode, and replay schedules with its control.

### Binding gate

Clean accuracy at least `0.80`, corrupt accuracy at least `0.65`, deleted and unknown abstention at
least `0.95`, attractor cosine gain at least `0.05`, and clean advantage over every matched control
at least `0.20`, with finite weights, snapshot parity, dense/sparse parity, and zero stores.

### Factor-transfer gate

Held-out `(1,1)` accuracy at least `0.70` and advantage over every matched control at least `0.20`
on at least 80% of circuits. The held-out cue/value combination cannot enter training, weight
construction, threshold fitting, or decoder calibration.

T1, M2 binding, M2 transfer, M3 prediction, M3 binding, and M3 transfer receive independent verdicts.
A later failure cannot be hidden by an earlier success, and one route cannot inherit another route's
GO label.

For every factor-transfer route, the four disjoint factor codebooks and their hash are frozen before
learning; `(1,1)` absence is asserted over all replay rows, collectors, update matrices, threshold
fitting, and decoder calibration. Report decoder-only and chance baselines for the held-out item.

## Claim boundary

Passing T1 would show factorized transfer under the declared imbalanced schedule by the already confirmed fixed-clock learner in this
synthetic runtime. Passing M2 would show supervised contrastive acquisition. Passing M3 would show
the exact teacher-forced replay-residual mechanism at the gates it actually clears. None establishes
biological memory, hippocampal-cortical consolidation, a brain metric, selfhood, or consciousness.

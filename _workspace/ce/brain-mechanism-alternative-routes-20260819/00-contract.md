# Alternative brain-mechanism routes: research contract

Status: COMPLETE

PREDECESSORS:

- `_workspace/ce/brainruntime-native-all-loops-p1-20260819`
- `_workspace/ce/_archive/neural-riemannian-metric-multiroute-execution-20260818`
- `_workspace/ce/_archive/neural-riemannian-metric-validation-20260818`
- `_workspace/ce/_archive/agi-connectome-geometric-memory-20260816`

## Motivation and exact question

The predecessor established a sharp split. A bounded supervised recurrent projection can place
cue--target binding and factorized transfer in the native `BrainRuntime.weight`, but the current
causal-STDP route cannot learn either function. Separately, a released PFC pseudopopulation shows
a descriptive selectivity-covariance change, while the proposed causal chain

$$
\Delta W \longrightarrow \Delta g \longrightarrow \Delta x
$$

has not been identified. The existing next-state readout predicts a bounded internal summary but
does not use its prediction to control action, and SCC is only a graph-theoretic recurrence object.

This run asks which conclusions survive when the *method*, rather than only a hyperparameter, is
changed. It tests a finite, frozen universe of alternative mechanisms and reports each route
independently. No route may borrow a success label from another route.

## Frozen candidate universe

### M: recurrent memory acquisition

All memory routes use the predecessor's independent seed-fixed cue/value codebooks, temporal-source
manifest, physical temporal and hippocampal cutoff, cue plus zero-input rollout, and matched
no-write and target-shuffled controls.

1. **M0 -- low-rank supervised capacity ceiling.** Decompose the already successful Route B into
   rank-1, rank-2, rank-4, and full writes, with equal-norm random low-rank and cue-only controls.
   This can identify a structural capacity threshold only; it is never learning evidence.
2. **M1 -- delayed three-factor eligibility.** Accumulate causal local eligibility during a replay
   block and apply it once at block end using a scalar, target-identity-free modulation signal.
   Compare positive, zero, sign-flipped, time-shuffled, eligibility-reset, no-replay, and
   target-shuffled arms.
3. **M2 -- contrastive settling.** Use fixed positive and negative recurrent phases and a bounded
   difference of local state correlations. Compare positive-only, negative-only, identical-phase,
   phase-order-reversed, no-write, and target-shuffled arms. This is supervised phase learning,
   not STDP.
4. **M3 -- predictive-error plasticity.** Update recurrent weights only from pre-transition native
   state and next-native-state prediction error. Target IDs, codebook cosines, and hippocampal rows
   are prohibited update features. First test held-out transition prediction; only then test
   zero-store binding and factor transfer.

### G: weight, geometry, and dynamics

1. **G1 -- randomized edge intervention.** From identical snapshots, apply a preregistered signed
   edge-block intervention and sham, norm/sign/density-matched scrambled-edge, gain-only, and
   noise-only controls. Estimate a finite-horizon reachability/predictive covariance metric from
   calibration pulses only, then score held-out trajectory, endpoint, and first-passage behavior.
2. **G2 -- fixed-weight metric sufficiency.** Hold recurrent weight exactly fixed while varying a
   declared local drift/gain and noise factorially. Test whether the frozen metric adds held-out
   predictive value beyond a direct state-space model, raw covariance, Euclidean distance, and an
   unrestricted SPD fit of equal parameter budget.
3. **G3 -- learned mediation.** Run only for an M1--M3 route that passes its memory gate. Test the
   seed-level mediated chain from randomized learning contingency through applied weight change
   and frozen metric change to independent recall. A failed or unavailable G3 cannot be replaced
   by the supervised M0 ceiling.

### C: prediction used for control

**C1 -- metacognitive risk gate.** Freeze a pre-transition state/action predictor and a risk
calibration on development data. Before observing the next state, choose between a committed and
a safe action. Compare with same-coverage random gating, shuffled-risk gating, always-commit,
always-safe, and a post-state oracle used only as a ceiling. This tests a narrow prediction-guided
control mechanism, not consciousness or selfhood.

### S: recurrent graph intervention

**S1 -- SCC feedback lesion.** Freeze the effective-edge rule before outcomes are scored. From
identical runtime snapshots, compare an SCC feedback cut with edge-count, sign, Frobenius-mass,
degree-stratum, and spectral-radius-matched non-feedback cuts, degree-preserving rewires, and
shuffled SCC labels. The endpoint is rollout recovery under a standardized perturbation. This
tests the contribution of declared recurrent feedback, not `SCC = memory` or
`SCC = consciousness`.

## Common protocol and leakage gates

- Torch CPU, deterministic random generators, zero runtime noise unless noise is the declared
  intervention, and `axon_delay=False` for the first mechanism test.
- Development seeds are `97201..97216`. Confirmation seeds `99201..99232` remain unopened until
  code, hashes, thresholds, and primary endpoints are frozen.
- Each arm starts from the same serialized pre-intervention snapshot and uses matched input and RNG
  streams. Weight, dense/sparse parity, applied deltas, transient state, and store counts are logged.
- Confirmation fitting, threshold selection, candidate addition, and endpoint substitution are
  forbidden. A route may be revised only on development seeds and must retain its STOP history.
- A decoder sees only the final activation and the independently committed value codebook. No
  target, goal, old activation, memory context, hidden intervention label, or episodic row may enter
  an evaluation feature.
- Memory evaluation physically clears both temporal and hippocampal stores, disables new
  hippocampal writes, resets transient dynamics, and audits zero rows after every rollout.
- All statistical units are seed-level circuits. Neurons and time steps are not treated as
  independent replicates.
- Every result is finite, reproducible from its manifest, and accompanied by the strongest matched
  adverse control. A raw nonzero weight or metric change is never sufficient.

## Route-specific decision gates

### M gates

- Binding: clean accuracy at least `0.80`, corrupt-cue accuracy at least `0.65`, unknown and deleted
  abstention at least `0.95`, attractor cosine gain at least `0.05`, and advantage over every matched
  control at least `0.20`.
- Factor transfer: held-out `(1,1)` accuracy at least `0.70` and advantage over no-write and shuffled
  targets at least `0.20`.
- M1 additionally requires the intended applied post/pre block contrast in at least 80% of circuits
  and failure of zero, sign-flipped, time-shuffled, and eligibility-reset controls.
- M2 additionally requires positive-minus-negative phase dependence and preregistered rank/norm
  bounds. M3 first requires next-state MSE no greater than `0.90` times persistence; prediction
  success without binding is reported only as prediction learning.

#### Frozen M0 construction

- The SVD target is the Route-B-style raw desired recurrent write built from independently reset
  native cue and value states:

  $$
  W^*=\sum_k\left(v_kc_k^{\mathsf T}+0.65v_kv_k^{\mathsf T}\right).
  $$

  It is not an SVD of the pre-existing random recurrent matrix. Rank truncation occurs on $W^*$
  before subtracting the initial matrix and before the bounded native install.
- `dale_law=False`; M0 uses no structural projection. Every arm uses maximum installed Frobenius
  norm `5.0`. The random low-rank control has the same singular values as the corresponding
  truncated $W^*$, with independent seed-fixed orthonormal singular vectors. The cue-only control
  is rescaled to the same pre-install Frobenius norm.
- The minimum sufficient rank is the first member of `{1, 2, 4, full}` whose complete binding gate
  passes on at least 80% of development circuits. That rule and rank freeze before confirmation.

#### Frozen M1 modulation and applied contrast

- The primary modulation is exactly `m_b = +1.0` at the end of every replay block. Its source is a
  fixed block-end clock and it is identical for every episode, cue, target, and replay condition.
  It reads no runtime state, reward, target, decoder, memory value, or condition flag. The zero and
  sign lesions replace it by `0.0` and `-1.0`; all other arms receive `+1.0` with identical counts.
- The temporal-order control presents the same paired target before its cue while preserving every
  event and pulse. The target-shuffled control preserves order and changes only target assignment.
  The no-replay control executes the same number of zero-input NREM ticks and block-end pulses with
  an empty staged replay store. The eligibility-reset lesion clears traces between cue and target.
- After the first phase is observed, every M1 arm calls `reset_evaluation_state()` before the second
  phase while preserving recurrent weight. Forward-order arms preserve the explicitly staged replay
  row so the second phase can present the value. The time-reversed arm removes that row before its
  cue phase so hippocampal recall cannot re-inject the already presented target into the cue. Thus native
  activation, refractory, adaptation, STP, lifecycle, and delay state cannot carry the cue across
  phases; only the external eligibility object can do so. The reset lesion additionally clears that
  eligibility object. Time reversal applies the same reset between target and cue.
- With cue codeword $c_k$, value codeword $v_k$, and the post-projection applied matrix
  $\Delta W=W_{\mathrm{after}}-W_{\mathrm{before}}$, the frozen association contrast is

  $$
  B(\Delta W)=\frac{1}{K}\sum_k\left[
  v_k^{\mathsf T}\Delta Wc_k-
  \frac{1}{K-1}\sum_{\ell\ne k}v_\ell^{\mathsf T}\Delta Wc_k
  \right].
  $$

  The mechanistic predicate is `B > 1e-6` and a paired `B` margin above the target-shuffled arm
  greater than `1e-6` in at least 80% of confirmation circuits. Raw eligibility cannot satisfy it.
  M1's exact claim label is `fixed-clock delayed three-factor acquisition`, not autonomous reward
  learning.
- M1 alone uses decoder abstention threshold `0.20`, frozen after the second and final development
  revision. On development diagnostics the minimum known-target cosine was `0.529764` and the
  maximum deleted-cue cosine was `0.170074`; no confirmation seed was opened. M0 and predecessor
  routes retain threshold `0.15`. The M1 threshold cannot be revised again.

#### Frozen M3 fitting boundary

- Each circuit fits one deterministic ridge predictor on its first 64 native transitions using
  pre-transition activation, refractory, memory trace, adaptation, STP, lifecycle, declared action,
  and a bias. The output target is the actual next native activation produced by `BrainRuntime.step`,
  never a symbolic target or teacher-forced codeword. Ridge is `1e-4` and the parameter shape is
  recorded.
- The predictor freezes before all held-out transition, recurrent-write, recall, and intervention
  scoring. Online refitting and confirmation-wide pooling are forbidden. Held-out transitions use a
  reversed action schedule and unseen drive magnitudes. Persistence sees the same transition set.
- Passing prediction MSE authorizes the separate recurrent-write test; it does not itself authorize
  a binding or factor-transfer claim.

### G gates

- The declared intervention direction and metric-direction ranking are correct in at least 80% of
  confirmation circuits.
- The paired 95% lower confidence bound for improvement over the strongest matched control is above
  zero for the frozen primary held-out score, with a directionally correct first-passage contrast in
  at least 80% of circuits.
- Null-family false rejection is at most `0.05`, transform-law residuals pass their fixed tolerance,
  and a metric claim is rejected or narrowed if an equal-budget direct-dynamics model matches it.
- G3 requires a sign-correct cross-fitted seed-level mediation coefficient whose 95% interval
  excludes zero; otherwise the full chain remains unestablished.

### C gate

- At exactly matched commit/defer coverage, prediction-guided gating has lower paired composite loss
  than random and shuffled-risk gates with a 95% interval below zero, while task utility remains
  within the frozen non-inferiority margin relative to always-commit.
- Predictor/scaler/threshold hashes, equal budgets, same-snapshot forks, and pre-state-only decision
  features must all pass. Safer behavior explained only by more deferral is STOP.

### S gate

- The SCC feedback cut causes a larger frozen primary recovery loss than every matched structural
  lesion with a paired 95% interval above zero on confirmation seeds.
- Edge count, sign, removed Frobenius mass, degree strata, sparse/dense parity, and the preselected
  edge rule must pass. Giant-single-SCC fixtures without a meaningful matched comparator are not
  evaluable, and total-weight or spectral-radius confounding is STOP.

## Allowed conclusions

Passing M1, M2, or M3 establishes only the named learning mechanism in this synthetic native
runtime. Passing G1/G2 establishes only simulator-level causal or predictive geometry under the
declared estimator. Passing C1 establishes only prediction-guided risk control in this task.
Passing S1 establishes only a functional contribution of the declared feedback structure.

This run cannot establish a biological hippocampal--cortical consolidation mechanism, a unique
brain metric, real-connectome causal modules, general metacognition, subjective experience, or
consciousness. The identities `SCC = memory`, `SCC = self`, `recurrence = consciousness`, and
`Riemannian metric = physical cortical geometry` are explicitly outside the claim space.

## Deliverables

- `10-sources.md`: predecessor provenance and any newly required primary empirical inputs, or a
  justified skip because the run is simulator-only.
- `11-math.md`: update orientation, metric estimator, intervention identifiability, and statistical
  unit audit.
- `12-routes.md`: frozen route table, counterexamples, and implementation order.
- `20-audit.md`: formal status and P0/P1 gate decision before implementation.
- `30-implementation.md`: exact code seams and bounded changes.
- `31-validation.md`: development results, freeze hashes, confirmation results, and control audit.
- `40-final-report.md`: route-by-route verdict and biological claim boundary.

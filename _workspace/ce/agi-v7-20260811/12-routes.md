# Alternative routes: sparse causal bridge V7

Status: COMPLETE

## 1. Fixed target and status boundary

For the fixed four-chart synthetic SCM, the target is a **positive reduction in
independent-seed H20 free-rollout path RMSE** after observing only the prefix
`x[0:81]`.  Positive improvement is defined as

\[
\Delta_{b,s}=\operatorname{RMSE}_{b,s}^{H20}
              -\operatorname{RMSE}_{V7,s}^{H20}>0,
\]

where one fresh simulation seed is one independent unit.  H5 is only the exact
first five rows of the same H20 prediction.  The layer is an empirical
forecast-controller gate in a fully observed, known-family, matched-basis
world; it is not an AGI, open-world causal-discovery, brain, or physical CE
claim.

Formal status used below:

- The metric, prefix boundary, DGP, and candidate algorithms are definitions.
- V1--V5 numbers are historical empirical development results from `itself`.
- V6 is a target-aware preregistered proposal, not a result: it has no runner or
  validation artifact.
- Every V7 performance statement is a prediction to be killed by fresh locked
  validation.  None is a derivation or theorem.

## 2. `main` versus `itself`: dependency audit

Compared revisions:

- canonical base: `fcb754ee5b1f35324e9502d3b3f146387eb25823` (`main`);
- historical evidence: `33836b89855d86c73e2ddf271b2c5eee6e1191b3`
  (`itself`).

| Dependency | `itself` | current `main` | V7 consequence |
|---|---|---|---|
| registrations | `sparse_causal_bridge_v1.json` through `v6.json`, recursively extended | all absent | V7 cannot extend an on-branch V6 file; its registration should be self-contained |
| sparse DGP/paired-do learner | `sparse_causal_bridge.py`, 1,066 inserted lines | absent; no matching bridge learner found | must be minimally ported with provenance, not imported from a checked-out branch |
| rank-one residual context | `latent_causal_bridge.py`, 712 inserted lines; imports sparse base | absent | any V5-derived route depends on this port |
| free rollout and baselines | `free_rollout_bridge.py`, 928 inserted lines; imports both modules | absent | genuine H20, prefix-adaptive dense, persistence, and equal-probe controls all depend on this port |
| unit tests | 261 sparse + 177 latent + 192 rollout inserted lines | absent | leakage, orientation, intervention, and lock tests must be restored or rewritten |
| runners | 19 + 19 + 7 inserted lines | absent | a V7 runner is new canonical code |
| frozen parent/result ledger | passing V4 artifact, failing V5 artifact, and integrity hashes exist | absent | historical hashes can establish provenance, but are not a runnable `main` parent by themselves |
| V6 consensus implementation | registration names `consensus_rollout_bridge.py` | **absent even on `itself`** | V6 cannot be reproduced as code; implementing it would be a new V7 implementation from a target-aware specification |

The historical chain is therefore:

```text
V1/V2 sparse DGP + paired intervention
       |
       +--> V3/V4 rank-one residual filter
                  |
                  +--> V5 single-origin H20 free rollout
                             |
                             +--> V6 consensus specification only
                                  (no runner, tests, or outcomes)
```

The historical diff contains 2,706 inserted implementation lines, 630 test
lines, 45 runner lines, and 723 lines of V1--V6 registration.  This is the
shared **absolute** integration burden for a literal port.  A defensible V7
should instead extract the minimum DGP, frozen parent reconstruction, controls,
and H20 evaluator into current `main`, while recording source revision and
hashes.  Copying all AGI documents or making `itself` the implementation base
would violate the contract.

### 2.1 Historical failure localization

| Version | Historical status | What changed | Relevant result |
|---|---|---|---|
| V1 | validation FAIL, test unopened | initial sparse paired-do model | graph exact, but target confounding contaminated local prediction; causal global RMSE was 15.28% worse than local |
| V2 | validation/test PASS | removed direct hidden loading from bridge targets | narrowed bridge-identification result only |
| V3 | validation FAIL, test unopened | restored target confounding and fit rank-one context per OOD seed | short-prefix latent AR error and fixed-local/CI gates failed |
| V4 | validation/test PASS | pooled one shared train latent AR; prefix adapts direction/center only | sequential one-step gate passed; it still reread true state each step |
| V5 | validation FAIL, test unopened | genuine single-origin H5/H20 free rollout | stable and leakage-free, but four robustness/comparator checks failed |
| V6 | registered, unimplemented, unrun | target-aware three-expert prefix consensus | development pilot only; no evidence |

V5 localizes the open problem more sharply than the earlier graph results:

- H20 sparse causal-latent RMSE: `0.333083`;
- H20 persistence RMSE: `0.389129` (14.40% better in the mean, but paired-CI
  lower bound `-0.01481`);
- H20 strongest stable OOD-prefix dense RMSE: `0.312697` (sparse was 6.52%
  worse);
- H20 equal-probe dense-latent RMSE: `0.334016` (sparse was only 0.28%
  better);
- false V5 checks: H5 persistence win fraction, H5 persistence CI, H20
  persistence CI, and H20 improvement versus stable prefix-dense;
- the rollout was not diverging: maximum learned Jacobian radius `0.78142`, no
  nonfinite predictions, and H20/H5 RMSE ratio `1.61081`.

Thus exact sparse support was not the demonstrated bottleneck.  The principal
V7 question is whether prefix-only reliability information predicts which
forecast correction is safe for a seed.

## 3. Common lock, leakage, and fairness envelope

Every route below is admissible only inside the same envelope.

1. Write and SHA-lock a self-contained V7 registration, DGP, implementation,
   parent reconstruction, seed manifest, and environment manifest **before**
   inspecting any V7 validation outcome.
2. V7 validation/test seeds must be disjoint from V1--V6 and from any V7
   pilot/development seeds.  Test opens only after all registered validation
   gates pass.
3. The forecast API receives fitted objects, `x[0:81]` or the explicitly split
   prefix, and horizon only.  It cannot receive an `Episode`, score object,
   `x[81:101]`, or hidden state.
4. Poisoning `x[81:101]` must leave every non-oracle prediction bit-identical;
   poisoning all hidden states must do the same.  H5 must equal the first five
   rows of the one H20 rollout.
5. Report per-seed H20 path RMSE and Student-t paired intervals.  A candidate is
   killed if the lower 95% bound of mean `Delta` is not positive against
   persistence or the strongest stable dense prefix model.  A sparse-value
   claim is also killed if its ratio to the equal-information/equal-probe dense
   control exceeds the preregistered noninferiority margin (at most `1.02`).
6. Require finite rollouts, zero future reads, registered state/norm bounds,
   and an unchanged H20 trajectory when H5 diagnostics are enabled or removed.
7. Both sparse and dense-probe arms receive the same 512 paired-do batches
   (four probe seeds x four sources x 32 pairs), the same observational train
   data, prefix length, number of fitted experts, and forecast origins.  V7
   validation/test uses zero intervention probes.  Report CPU time, number of
   fits, number of component rollouts, peak memory, and probe pairs separately;
   wall-clock equality alone is not an information-fairness argument.

The numeric hurdle implied by V5 development is explicit: merely matching the
old strongest dense prefix model requires H20 mean RMSE below `0.312697`; a 5%
improvement would require below `0.297062`.  These are design diagnostics, not
V7 gates unless copied into the locked registration.

## 4. Route R1 -- frozen V6 three-expert consensus

### Path and one-system-change definition

Freeze every V5 component.  At pseudo-origin 60, fit and independently run for
20 prefix-only steps:

1. sparse causal-latent free rollout;
2. stable OOD-prefix dense rollout;
3. persistence.

For prefix backtest MSE `E_j`, set

\[
w_j=\frac{(E_j+10^{-12})^{-1/2}}
          {\sum_l(E_l+10^{-12})^{-1/2}}.
\]

Refit the two learned experts on `x[0:81]`, freeze the weights, generate their
complete H20 paths independently, and return the leadwise weighted Euclidean
barycenter.  The consensus is never fed back into a component.

**Exact single system change from V5:** replace the selected V5 trajectory by
the above frozen-weight three-trajectory output aggregator.  No DGP, mechanism,
filter, expert, probe, prefix, or metric change is allowed.

### Degrees of freedom and target awareness

- Fitted episode-level degrees of freedom: two continuous simplex coordinates;
  the third weight is fixed by sum-to-one.
- Adjustment choices frozen by the specification: expert membership,
  pseudo-origin 60, backtest horizon 20, all-chart squared loss, inverse-square-
  root exponent, epsilon, and nonrecursive barycenter (seven protocol choices).
- **TARGET-AWARE: YES, high.** V5 validation was used to reject a scalar Kalman
  candidate and select this rule, expert set, and constants.  The disclosed
  look-elsewhere lower bound is two algorithm families; the full number of
  tried weight/expert variants is unknown and must not be represented as one.

The target-aware V5 pilot moved H5 `0.206780 -> 0.204421` (1.14%) and H20
`0.333083 -> 0.319301` (4.14%).  It still sat 2.11% above the V5 stable
prefix-dense H20 value `0.312697`.  These numbers justify only a fresh
confirmation attempt, not a positive claim.

### Leakage, fairness, cross-prediction, and falsification

- Apply the common poisoning/API/H5-prefix tests.  Additionally, changing any
  `x[81:101]` value must not change `E_j` or `w_j`; weights must be finalized
  before outer-origin prediction.
- The equal-probe control replaces only the sparse expert with the all-12-edge
  paired-do dense expert and retains the other two experts **and the candidate's
  frozen weight vector**, as specified by historical V6.  This keeps the
  sparse-versus-dense component as the only control difference.  Both sides
  perform three inner and three outer component rollouts; the dense replacement
  is still generated and accounted for even though its own prefix error does
  not refit the shared weights.
- Cross-prediction: seeds with larger causal prefix error should receive smaller
  causal weight, and the preregistered slope between causal weight and its
  future relative error should be negative.  This is a secondary diagnostic,
  never a repair gate.
- Kill R1 on any common gate failure, on a nonpositive paired H20 improvement
  CI versus either persistence or stable prefix-dense, or if consensus weights
  fail to predict relative future error in the registered direction.

### Burden and rank

Absolute burden is very high because the full V5 stack is missing from `main`.
Incremental burden is medium-high: V6 names a nonexistent consensus module, so
the runner, hash lock, aggregation tests, target-poisoning tests, and equal-probe
consensus must be newly implemented.  R1 is ranked **second**, not first: it is
specified well, but has two fitted degrees of freedom and the strongest
target-aware history.

## 5. Route R2 -- one scalar reliability gain on the latent correction

### Path and one-system-change definition

Keep the frozen V5 sparse mechanism and rank-one residual filter.  Split the
observed prefix at transition 60.  Using only data through 60, fit the filter;
over prefix transitions 61--80, produce causal one-step residual corrections
prequentially (each correction is formed before reading its next outcome).  Fit
one scalar

\[
\hat\alpha=
\operatorname{clip}_{[0,1]}
\frac{\sum_t \widehat r_t^\top
      \{x_{t+1}-f(x_t)\}}
     {\sum_t \|\widehat r_t\|_2^2},
\]

with the registered fallback `alpha=0` when the denominator is zero.  Refit the
same filter on the full prefix and perform one H20 recursive rollout with

\[
\widehat x_{t+1}=f(\widehat x_t)+\hat\alpha\,\widehat r_t.
\]

**Exact single system change from V5:** multiply the existing rank-one residual
correction by one prefix-estimated scalar reliability gain.  The graph,
coefficients, AR, filter rank/direction, prefix length, recursion, and all
baselines remain unchanged.

### Degrees of freedom and target awareness

- Fitted episode-level degrees of freedom: one bounded continuous scalar.
- Fixed adjustment choices: split 60/80, all-chart one-step squared loss, and
  clipping interval `[0,1]` (three protocol choices).  There is no grid,
  threshold selection, H20 target-window fitting, or expert-subset search.
- **TARGET-AWARE: YES, moderate.** The route is motivated by V5 seed variance
  and inherits the 60/80 split from the target-aware V6 design.  The scalar
  formula itself has not been reported as piloted on V5 outcomes.  It must be
  labeled development-informed and confirmed only once on fresh V7 seeds.

No numerical performance estimate is available because the V5 artifact stores
errors, not the trajectories needed to reconstruct this gain.  Manufacturing a
pilot number from per-model RMSE alone would ignore error covariance.

### Leakage, fairness, cross-prediction, and falsification

- Each prefix correction used to fit `alpha` must be timestamp-audited: the
  corresponding `x[t+1]` may score that correction only after it was generated.
  Target-window values `x[81:101]` cannot enter the estimator.  Apply all common
  mutation and API tests.
- Give the all-12-edge equal-probe dense mechanism its own scalar gain fitted by
  the identical prefix procedure.  Sparse and dense then differ only in frozen
  paired-do support, not in context calibration, probes, split, or fitted-dof
  count.  The strongest observation-only dense prefix baseline remains
  separate and receives no interventions.
- Cross-prediction: `alpha` should be smaller on seeds where full latent
  correction loses to its no-latent ablation.  Register the sign of this
  association as a diagnostic and publish the full `alpha` distribution.
- Kill R2 if `alpha` collapses to a boundary on a preregistered excessive seed
  fraction, if it does not beat V5's unscaled parent and the strongest dense
  prefix model with positive paired H20 CIs, or on any common gate failure.

### Burden and rank

Absolute burden remains high because the V5 parent must be reconstructed on
`main`.  Incremental burden is the lowest of the performance-directed routes:
one scalar estimator, one multiplier in the residual path, and focused temporal
leakage/equal-probe tests.  R2 is ranked **first and recommended as the narrowest
defensible V7 route**.  It changes one uncertainty-control degree of freedom at
the locus implicated by V5, without adding an observational expert to hide a
failed sparse component.

## 6. Route R3 -- prefix hard selection among three frozen experts

### Path and one-system-change definition

At pseudo-origin 60, independently fit and H20-backtest the same three experts
as R1 using only `x[0:81]`: sparse causal-latent, stable prefix-dense, and
persistence.  Choose the single lowest-MSE expert with a preregistered
deterministic tie order.  Refit that expert using the full prefix and execute
only its one H20 rollout.

**Exact single system change from V5:** replace the fixed sparse trajectory by
one prefix-selected trajectory; do not average trajectories or change any
expert internals.

### Degrees of freedom and target awareness

- Fitted degrees of freedom: one categorical choice with three states and zero
  continuous weights (two decision boundaries).
- Fixed choices: three-expert set, pseudo-origin, H20 backtest, all-chart MSE,
  and tie order.
- **TARGET-AWARE: YES, high.** The expert set and pseudo-origin are inherited
  from V6 after V5 inspection.  Selection also incurs winner's-curse/look-
  elsewhere exposure over three experts per seed.

As an explicitly invalid future-aware ceiling, choosing the best expert from
the actual V5 target window would have produced H20 mean RMSE `0.277672`, with
6/11/3 seeds assigned to causal/dense/persistence.  This target oracle is 11.20%
below the fixed dense mean but is not implementable evidence.  It only shows
that useful seed heterogeneity exists if a prefix ranker can predict it.

### Leakage, fairness, cross-prediction, and falsification

- Selection must be complete before reading `x[81]`; future poisoning must not
  change the selected label.  Record inner-fit rows and chosen label for every
  seed, and apply the common tests.
- The equal-probe controller substitutes the same-probe dense causal expert for
  the sparse expert and otherwise has the identical three-way selection and
  fit/rollout budget.
- The decisive cross-prediction is registered selection accuracy: prefix error
  ranking must predict the target-window winner above the preregistered chance
  benchmark.  Report a 3x3 prefix-winner/target-winner table even if the main
  RMSE gate fails.
- Kill R3 if the prefix ranker does not beat the best fixed expert with a
  positive paired H20 CI, or if selection accuracy fails its registered lower
  bound.  Mean improvement without those conditions is compatible with
  selection noise.

### Burden and rank

Absolute burden is high; incremental code burden is low-medium because V5
already computes all experts, but duplicate inner/outer fitting, deterministic
selection, and fairness controls increase compute and tests.  R3 is ranked
**third**: it has no continuous mixture freedom, yet its discontinuity and
three-way target-aware selection can amplify 20-transition backtest noise.

## 7. Route R4 -- probe-only multiplicity-controlled support

### Path and one-system-change definition

Estimate all 12 directed off-diagonal paired-do effects using the same 512
probe batches.  Replace the geometry-proposal/threshold/top-two support rule by
a single Benjamini--Hochberg procedure at preregistered `q=0.05` over the 12
two-sided intervention tests, retaining split-half sign agreement.  Freeze the
selected coefficients before any OOD prefix or outcome is generated; keep the
V5 residual filter and rollout unchanged.

**Exact single system change from V5:** replace only the sparse edge-support
selector by the above all-edge, probe-only multiplicity-controlled selector.

### Degrees of freedom and target awareness

- Validation-fitted degrees of freedom: zero.
- Mechanism degrees of freedom: the random number `K` of rejected nulls,
  `0 <= K <= 12`, with one paired-do coefficient per retained edge; report K.
- Fixed adjustment choice: one conventional FDR level `q=0.05`; no edge budget,
  geometry affinity, validation threshold, or truth label is allowed.
- **TARGET-AWARE: YES, low-to-moderate.** It is proposed after seeing V1--V5,
  but the rule is not tuned to V5 rollout outcomes.  The route must still be
  treated as development-informed.  Trying other q values after V7 lock is
  prohibited.

### Leakage, fairness, cross-prediction, and falsification

- Support and coefficients are train/probe-only and hash-frozen before OOD
  episodes.  Changing all V7 validation/test seeds must leave the graph and
  coefficients bit-identical.  Apply the common future/hidden mutation tests.
- The dense control uses the exact same 12 estimates and retains all of them;
  observational data, probes, and residual calibration are identical.
- Cross-predictions: registered no-bridge and permuted-sign controls should
  select no edges; on the programmed SCM, the selected signs should be
  `A->C` positive and `C->D` negative without labels entering selection.
- Kill R4 for false edges in either negative control, failure to recover both
  programmed effects, or failure of the common H20 gates.

### Burden and rank

Absolute burden is high; incremental burden is medium because intervention
standard errors, p-values, BH ordering, negative controls, and graph-freeze
tests must be made explicit.  R4 is ranked **fourth**.  V5 sparse and equal-
probe dense H20 errors already differed by only 0.28%, so support selection is
not the evidenced performance bottleneck.  This route is scientifically clean
but unlikely to answer the current H20 robustness failure.

## 8. Ranking and recommendation

| Rank | Route | Structural locus | New fitted dof per seed | Target-aware risk | Incremental burden | Main reason |
|---:|---|---|---:|---|---|---|
| 1 | R2 scalar latent reliability | state/context correction | 1 continuous | moderate | lowest | directly attacks seed-varying over/under-correction without adding a new expert |
| 2 | R1 frozen V6 consensus | output aggregation | 2 continuous | high | medium-high | precise prior specification, but target-aware and pilot still trailed dense at H20 |
| 3 | R3 hard selector | controller/model selection | 1 categorical | high | low-medium | potentially large oracle ceiling, but noisy and discontinuous |
| 4 | R4 BH support | causal graph selection | random K coefficients, no validation fit | low-moderate | medium | clean fairness, but historical evidence says graph support is not the bottleneck |

Recommendation: preregister **R2 only**, with R1--R4 retained in this document
as disclosed development alternatives.  Do not run several V7 candidates and
promote the winner.  R2 is the smallest system change, has the fewest continuous
degrees of freedom among the performance-directed routes, preserves a direct
sparse-causal ablation, and admits a symmetric equal-probe dense control.

Before registration, one blocker must be resolved mechanically: reconstruct
the V5 parent on current `main` and verify it reproduces the frozen historical
V5 artifact under the recorded hashes and environment.  If that cannot be done
without changing the parent or inspecting V7 outcomes, the correct result is a
blocked V7 report, not a relaxed gate or a second candidate.

CE_RUN=_workspace/ce/agi-v7-20260811

# Mathematical and statistical audit

Status: COMPLETE

## Scope and bottom line

This lane independently audited the sparse-causal-bridge registrations, saved
artifacts, and implementations at historical commit `33836b8`.  V6 has a
locked registration but no consensus implementation, validation artifact, or
test artifact at that commit.  It is therefore a proposal, not evidence.

The strongest valid historical conclusion is narrow.  V4 passed fresh
validation and locked test for sequential one-step prediction, but it read the
true current state again at every step.  V5 was the first genuine single-origin
H20 free rollout and it failed its locked validation gate.  Its failure was not
numerical divergence: it was seed-level uncertainty and inferiority to the
strongest stable OOD-prefix dense model.  V6's prefix-only consensus is a
plausible engineering proposal, but its primary gates do not put paired
uncertainty on the claims that motivated it.  V7 must not inherit the V6 gate
unchanged.

No historical result supports AGI, open-world causal discovery, autonomous
agency, or a brain mechanism.  The admissible endpoint is conditional-mean
forecasting in the registered, fully observed, matched-basis four-chart
synthetic family, conditional on one frozen train/probe fit.

## Historical status

| Version | Result | What the result actually says |
|---|---|---|
| V1 | validation FAIL | The graph was recovered, but prediction failed when the latent cause loaded a bridge target; a negative-control defect was also exposed. |
| V2 | validation/test PASS | Passed only after removing direct hidden loading from bridge targets and narrowing the claim. |
| V3 | validation FAIL | Restored target confounding; per-seed estimation of the latent AR was unstable. |
| V4 | validation/test PASS | Pooling the train residual AR repaired the registered one-step sequential gate. It is not free rollout evidence. |
| V5 | validation FAIL, test unopened | First honest H5/H20 single-origin free rollout; stable, but not robust enough across seeds and worse than stable adaptive dense at H20. |
| V6 | preregistered, unimplemented | Prefix-backtested three-expert consensus; no empirical status can be assigned. |

The version sequence is adaptive development.  Earlier validation sets may be
used as disclosed development evidence, but results and p-values cannot be
pooled across versions.  Only a new locked test can be confirmatory for V7.

## V5 numerical audit

For seed `s`, the implementation first computes one path RMSE over all requested
leads and charts, then pairs models within the same seed.  For comparator `b`
and candidate `c`, it uses

\[
d_s=R_s(b)-R_s(c),\qquad
L=\bar d-t_{0.975,n-1}\,s_d/\sqrt n.
\]

This is the correct Student-t lower endpoint for the mean paired difference,
with the simulation seed as the independent unit.  The registered critical
value `t[19]=2.093024054` reproduces the artifact.

| Comparison | n | mean paired improvement | paired SD | 95% lower | seed wins |
|---|---:|---:|---:|---:|---:|
| H5 vs persistence | 20 | 0.024879 | 0.067669 | -0.006791 | 0.65 |
| H20 vs persistence | 20 | 0.056046 | 0.151404 | -0.014814 | 0.75 |
| H20 vs stable adaptive dense | 20 | -0.020386 | 0.089089 | -0.062080 | 0.35 |
| H20 vs same-probe dense+latent | 20 | 0.000933 | 0.002494 | -0.000234 | 0.50 |

The mean H20 RMSEs were `0.333083` for causal-latent, `0.389129` for
persistence, `0.312697` for stable adaptive dense, and `0.334016` for the
same-probe dense+latent control.  Thus V5 improved the persistence mean by
14.40%, but its seed differences ranged from `-0.346767` to `+0.345998`; it
was also 6.52% worse than stable adaptive dense.  The four reported failed
checks are reproduced exactly:

- H5 persistence seed-win fraction;
- H5 paired CI versus persistence;
- H20 paired CI versus persistence;
- H20 superiority versus stable adaptive dense.

The V5 implementation otherwise gets important protocol details right:

- each evaluation seed produces one episode and one forecast origin;
- all models are evaluated on the same seed, so pairing is correct;
- H5 is the exact first five rows of the one H20 rollout;
- the candidate API accepts fitted objects, `x[79]`, `x[80]`, and horizon, not
  an `Episode`, future outcome, or hidden state;
- V5 validation/test seeds are fresh, and the locked test was not opened;
- all candidate predictions were finite, maximum absolute prediction was
  `1.04922`, sampled mechanism Jacobian radius was `0.78142`, and the latent AR
  magnitude was `0.93693`.

Two qualifications matter.  First, `future_observation_reads_by_predictor` is
written as a constant zero; the poison test exercises the candidate function,
not every model and the complete gate pipeline.  The evaluator also constructs
the full episode before slicing the prefix, and a prefix `Episode` still
contains hidden values even though the fitting routine ignores them.  V7 needs
an enforced prefix-only interface and an instrumented read audit.  Second, the
seed CI is conditional on the one frozen train/probe fit.  It measures new
episode variability, not variability from retraining the whole learner.

The older V1--V4 helper used `1.96` instead of a finite-sample Student-t
critical value at n=20/30.  Recalculation with t critical values does not change
their pass/fail verdicts, but that helper must not be reused.  V5 corrected it.

## V6 design audit

### Defensible features

- V5 validation is explicitly disclosed as development data, V5 test remains
  unopened, and V6 registers fresh `67100...67119` validation and
  `68100...68129` test seeds.
- The pseudo-origin is 60.  Each expert fits on `x[0:61]`, predicts leads
  61--80, and scores only values already inside the final observed prefix
  `x[0:81]`.  Refit at origin 80 therefore need not read `x[81:101]`.
- Weights are nonnegative and sum to one, component H20 trajectories are
  generated independently, and the consensus is not fed back into components.
- One fresh simulation seed remains the inferential unit.  The 20 leads and
  four charts in a prefix backtest are training observations, not 80
  independent samples.
- Its claim boundary correctly describes a same-loading synthetic forecast
  controller, not AGI or unseen-environment generalization.

### Defects that block direct promotion to V7

1. **The motivating claims lack paired inference.**  V6 requires only mean
   reduction, terminal ratio, and a 0.55 seed-win fraction versus V5.  At n=20,
   0.55 is only 11 wins and is weak evidence.  Its 1.02 noninferiority ratio
   versus same-probe dense consensus and 1.05 ratio versus adaptive dense also
   have no paired CI and no justified noninferiority margin.  These are the
   claims in the hypothesis, so raw ratios are insufficient.

2. **H5 is still a large active gate.**  The current research contract makes
   H20 primary and H5 an exact-prefix diagnostic.  Inheriting V6 would violate
   that contract and reintroduce multiplicity without answering the primary
   question.

3. **The equal-probe consensus control is asymmetric.**  The dense consensus
   replaces the sparse component but is forced to reuse weights estimated from
   the sparse component's prefix error.  That is useful as a same-weight
   mechanism swap, but it is not the strongest equally treated dense
   controller.  A fair benchmark must also let the dense component obtain its
   weight by the identical rule, or include both the same-weight ablation and a
   symmetrically refit dense control.

4. **The weighting rule has no uncertainty-optimal derivation.**  Weighting by
   inverse individual RMSE ignores cross-expert error covariance.  One
   20-step, strongly autocorrelated pseudo-backtest can produce noisy episode
   weights, and `1e-12` allows near-degenerate domination.  Moreover, weights
   score inner models fit through time 60 but are applied to changed outer
   models refit through time 80, including a shifted persistence anchor.  This
   is not target leakage, but it is a calibration mismatch.

5. **The controller prevents a sparse-causal contribution claim.**  The
   candidate already contains the strongest adaptive dense expert.  A pass can
   support an ensemble/controller claim only.  Sparse necessity would require
   a preregistered no-sparse ablation and a symmetric dense replacement with
   paired uncertainty.

6. **Stability checks are incomplete.**  A sampled sparse-mechanism radius and
   the adaptive model's diagonal coefficient do not bound the full adaptive
   dense Jacobian, dense-probe component, or augmented mechanism-plus-AR state.
   A finite consensus can also hide large cancelling component paths.  Every
   gating component and control must be checked independently.

7. **The pilot record is insufficient for power analysis.**  It gives mean
   weights and mean H5/H20 RMSE but not seed-level paired differences, variance,
   all tried candidate rules, or a runnable consensus artifact.  V6 remains a
   preregistered hypothesis, not a reproducible result.

The many gates are not automatically an alpha-inflation problem if every gate
is preregistered and all are required for one strictly conjunctive claim; this
is an intersection-union decision.  But V6 mixes that engineering conjunction
with individually interpreted superiority/noninferiority statements, several
without uncertainty.  If any favorable secondary result may be highlighted,
familywise correction or a fixed hierarchy is required.

## Dimensionless audit

The generator uses `tanh(x_source)`, so `x`, hidden context, noise scales, and
intervention amplitude must be dimensionless normalized synthetic coordinates.
On that interpretation, self/bridge/loading/AR coefficients, spectral radii,
RMSE ratios, and convex weights are dimensionless and coherent.  The historical
registration does not say this explicitly.

V6's `E_j + 1e-12` is dimensionally valid only when MSE is dimensionless; raw
MSE aggregation also assumes the four chart coordinates are commensurate.
V7 should register fixed training-only chart scales `s_j` and either model
`z_j=x_j/s_j` or score normalized errors.  The epsilon must be stated as a
dimensionless normalized-MSE constant (or as `1e-12 * s^2`).  The absolute
prediction bound `5.0` must use the same normalized coordinate system.

The repository dimensionless script exits successfully.  The associated
pytest module could not be collected in this environment because the current
Python lacks `torch`; this does not constitute equation-specific machine
validation of the historical AGI files.

## Concrete V7 statistical contract

### Endpoint and evaluation

1. Use exactly one origin per fresh seed: observe `x[0:81]`, produce one H20
   path for `x[81:101]`, and never update from target-window observations.  H5
   is reported only as `prediction_H20[:5]`; it cannot affect pass/fail.
2. Define the seed endpoint before implementation:

   \[
   R_s(m)=\sqrt{\frac1{80}\sum_{k=1}^{20}\sum_{j=1}^{4}
   \left(\frac{x_{s,80+k,j}-\hat x^{(m)}_{s,80+k,j}}{s_j}\right)^2},
   \]

   where each `s_j` is frozen from training data only.  Report the arithmetic
   mean of `R_s`; do not pool leads as independent observations.
3. Treat validation as a locked screening run and the unopened test as the
   confirmatory result.  No gate, margin, model, seed, or normalization change
   is permitted after validation is seen.

### Seed count and uncertainty

V5's H20 persistence difference had paired SD `0.1514` and mean `0.0560`.
Using a two-sided-95 lower endpoint and 80% power gives about 58 seeds at that
effect; inflating the development SD by 20% gives about 83.  A defensible V7
floor is therefore **96 fresh validation seeds and 96 fresh locked-test
seeds**, with disjoint development, train/probe, validation, and test blocks.
If the selected V7 route's primary development comparison implies a larger
sample by

\[
n=\left\lceil
((z_{0.975}+z_{0.80})\,1.2s_d/\delta)^2
\right\rceil,
\]

use the larger number and freeze it before implementation.  Never reduce n
after viewing V7 outcomes.  Every V7 evaluation seed must also be disjoint from
all registered V1--V6 seeds, including unopened ones.

### Primary decision rule

Use H20 only and require a preregistered conjunction:

- superiority to persistence: the paired t 95% lower bound of
  `R_s(persistence)-R_s(candidate)` is greater than zero (or a positive,
  development-justified minimum effect);
- improvement over the frozen reproducible parent: the analogous paired lower
  bound is greater than zero if V7 is claimed as a repair;
- noninferiority to the strongest frozen stable dense-prefix model and the
  symmetric equal-probe dense control: for
  `l_s=log(R_s(candidate)/R_s(control))`, the paired 95% upper bound must be
  below `log(1+Delta_NI)`, where `Delta_NI` is justified and locked from
  development data, not copied post hoc from V6;
- finite/stable output and zero future reads for every seed.

If all clauses are necessary for the one stated claim, no multiplicity
adjustment is needed for that conjunction.  All other baseline, terminal,
downstream, seed-win, lead-wise, H5, and weight analyses are secondary.  Report
their paired intervals and apply Holm correction if they are used for separate
claims.  A seed-win fraction should be accompanied by an exact binomial or
Wilson interval and should not replace the paired magnitude analysis.

### Fair controls and resource accounting

Freeze before V7 validation:

- persistence;
- the strongest stable dense-prefix model selected only from historical or
  separate development evidence;
- the frozen V5 parent, and V6 only if its missing implementation can be
  reproduced exactly without V7 outcomes;
- an all-12-edge same-probe dense model receiving the identical train episodes,
  intervention pairs, prefix, optimization attempts, and controller rule;
- for any consensus, a no-sparse ablation and both a same-weight component swap
  and a symmetrically reweighted dense consensus.

The artifact must tabulate per model: train transitions, intervention pairs,
prefix transitions, fitted parameter count, pseudo-origins, optimization
attempts, CPU/wall time, and all data-dependent tuning.  A controller containing
a dense expert may claim controller forecasting only, not causal sparsity.

### Leakage and stability tests

- All non-oracle predictors and weight functions receive an immutable prefix
  array and frozen fitted objects only; they never receive an `Episode`, hidden
  array, score target, or future-sized buffer.
- Poison `x[81:101]` and all hidden values and assert bit-identical predictions
  and weights for every non-oracle model through the complete gate pipeline.
- Instrument or proxy state-array reads and assert maximum read index 80;
  replace the historical hard-coded read count.
- Assert H5 equals the exact prefix of the single H20 prediction and that
  changing validation/test seeds cannot change frozen global fits.
- Check finiteness, normalized maximum norm, and full component paths before
  consensus.  For each dynamic expert check the full Jacobian, including all
  dense cross terms; for latent experts also check the augmented AR state.
  Describe sampled pathwise stability as pathwise, not a global contraction
  proof.
- Hash-lock registration, parent artifacts, implementation, tests, and baseline
  definitions before validation.  Preserve every failure and leave test closed
  unless validation passes unchanged.

If V7 cannot supply a symmetric equal-probe control, a justified
noninferiority margin, and enough fresh seeds for the selected paired effect,
the mathematically correct outcome is `BLOCKED`, not another tuned gate.

# Math and eligibility lane — CloudCell actual recordings

Status: COMPLETE

## Finding first

The local Hallinen/CloudCell files support an **observational, output-relative
Fisher calculation** for GCaMP recordings if their same-recording timebase and
signal-class gates pass.  They do not supply a canonical source population
or target population.  Consequently, a row-index partition can test a
time-forward prediction diagnostic, but is not an estimator of an anatomical
brain router.

The existing GFP-only AML18 result is a complete counterexample to a stronger
interpretation of that previously tested lag signature.  The same local-memory
gate was positive in all 11 AML18 recordings, with an effect at least as large
as the AML32 GCaMP panel after the preregistered headroom comparison.  The
result and exact values are recorded in
`docs/7_AGI/25_Local_Temporal_Memory_Confirmation.md` and independently
recomputed by
`_workspace/ce/_archive/agi-clarus-field-20260812/artifacts/bio_gate_redesign_check.py`.
A lag score in that measurement family therefore cannot be used as evidence
for calcium memory, synaptic routing, or an intrinsic neural delay.  This
counterexample constrains lag-score interpretation; it is not itself a test of
the new output-Fisher tensor below.

## Q1: eligible operational metric

For one GCaMP recording j, let x^A be an explicitly selected source-group
fluorescence observation, and let z be a chart fitted on the training block
only.  The only eligible metric is the conditional output Fisher tensor

$$
G_{ab}^{o\leftarrow A}(z,c)
=
\mathbb E_{h\mid z,c}\mathbb E_{o\mid z,h,c}
\left[
\partial_a\log p_j(o\mid z,h,c)
\partial_b\log p_j(o\mid z,h,c)
\right].
$$

Here o is a future locomotion readout on the same `hasPointsTime` clock, c is
an observed locomotor regime rather than an experimental task context, and h
contains only information available no later than t.  In a Gaussian
mean-and-covariance model this is

$$
G_{ab}=
\mathbb E_{h\mid z,c}\left[
(\partial_a\mu)^T\Sigma^{-1}(\partial_b\mu)
+\tfrac12\operatorname{tr}
\left(\Sigma^{-1}(\partial_a\Sigma)
\Sigma^{-1}(\partial_b\Sigma)\right)
\right].
$$

This is not `Cov(z)^{-1}`.  It is positive semidefinite and becomes an SPD
summary only where the output distinguishes all chart directions, or under a
separately frozen calibration tensor.  A coordinate change z'=phi(z) gives

$$
G'=J_\phi^{-T}GJ_\phi^{-1},
\qquad dz'^T G' dz'=dz^T G dz.
$$

Thus AIRM comparisons are permitted only within the same recording/chart and
common state support.  Raw SPD matrices must not be averaged across worms.

### Minimal predeclared measurement choices

- Primary panel: the seven checksum-locked AML32 moving recordings.  AML18 is
  the matched GFP falsifier, not a GCaMP replicate; AML310 is not silently
  pooled because its BFP/AVA handling is a distinct apparatus condition.
- Clock: use strictly increasing `hasPointsTime` and only contiguous windows
  with finite raw observations.  Do not use interpolated or non-contiguous
  `I_smooth_interp_crop_noncontig` as a primary trace.
- State: a two-dimensional, training-only PCA chart of a predeclared source
  fluorescence group.  The low dimension matches the two-dimensional output
  below and makes full-rank Fisher status checkable rather than ridge-created.
- Output: o=(v, pc3) at t+H, where `pc3 := behavior.pc_3`.  Both fields are present in the same MATLAB
  `behavior` object as the fluorescence clock; pc3 avoids requiring a missing
  centerline reconstruction.  This is an output geometry of recorded
  fluorescence, not a metric of the animal's physical body or brain.
- Time: with the approximately 6-volume/s acquisition, fix H=ell=delta=6
  volumes (about one second).  Do not select among horizons after scores are
  seen.  Every history/future window must be fully observed.
- Context: define c from a pre-t signed-velocity history only, for example the
  sign of the median of v from t-6 through t.  It is a locomotor-regime
  stratification, not a randomized condition.  A future-derived ethogram
  label is not eligible as c.
- Nuisance history h: past target state, past v, pc1_2, pc3, x/y position,
  red/green whole-recording imaging summaries, and a disjoint global
  fluorescence proxy.  No future behavior or future fluorescence enters h.

Q1 is therefore **PASS_INPUT for a metric-only empirical preregistration**,
not a completed empirical result.  It becomes `BLOCKED_SIGNAL_CLASS` for any
claim beyond recorded-fluorescence/output geometry unless the AML18 control is
shown not to reproduce the claimed quantity under the same frozen pipeline.

## Q2: non-identifiability of anatomical routing

The local `heatDataMS.mat` schema contains fluorescence arrays, `XYZcoord`,
and behavior fields, but no per-unit canonical neuron-name/connectome join
key.  The local extracted AML32 recording inspected here has no `ID` field;
its locally available `pointStatsNew.mat` counterpart is absent.  Scanner
coordinates alone do not identify a neuron across animals or map it to an
edge in an anatomical connectome.

Therefore the target equation

$$
R_{j,c}^{A\to B}(\ell,\delta)
=\frac1{N_{\rm test}}\sum_{t\in\mathcal T_{\rm test}}
\left[\log p_1(x^B_{t+\delta}\mid H_t^B,z^A_{t-\ell:t},c)
-\log p_0(x^B_{t+\delta}\mid H_t^B,c)\right]
$$

has no anatomical A or B in the CloudCell files.  Calling a positive value a
synaptic, structural, or brain-router result is a P0 error.  Q2 is
`BLOCKED_SOURCE_TARGET_DEFINITION`.

An arbitrary disjoint row partition may be used only as a diagnostic:
fit both likelihoods on chronological training data, compare held-out target
group fluorescence, and report all fixed partitions rather than selecting a
winner.  The result is **held-out neuron-group predictive transfer in an
indicator-contaminated recording**.  It neither repairs the missing labels
nor provides a candidate route for the brain-algorithm ledger.

## Q3: causal boundary

No randomized, time-locked source intervention; sham; reverse-direction
control; and non-target control are available in this local moving-recording
input.  Hidden common drive and common imaging artifacts can yield R>0 with
no A-to-B edge.  Thus Q3 is `BLOCKED_INTERVENTION`.  Neither a negative-lag
control nor a row-shift null converts the observational diagnostic into a
causal result.

## Apparatus STOP rules

Stop the relevant recording or claim if any of the following holds:

1. archive checksum/provenance, finite signal arrays, common behavior clock,
   or strictly increasing timestamps after the frozen 12-volume leading guard
   cannot be verified; any window crossing a recorded acquisition gap is
   excluded before splitting;
2. source/target neuron names and an independently verified connectome join
   are absent (anatomical R STOP);
3. either locomotor regime lacks training and held-out support, the two PCA
   coordinates cannot be fitted from training data, or the output Fisher is
   rank-deficient (metric estimate STOP; do not rescue it with a chosen ridge);
4. windows cross a missing acquisition, split boundary, or a 12-volume
   embargo around a chronological 60/20/20 split;
5. smoothing/interpolation, outcome-selected clusters, a correlation ordering,
   or a post-score horizon/threshold change is introduced;
6. a claim interprets a lag score as neural rather than measurement-process
   evidence without defeating the AML18 GFP falsifier;
7. a causal claim is attempted without randomized perturbation plus sham,
   reverse, and non-target controls.

## Status ledger

| Item | Status | Claim ceiling |
|---|---|---|
| output-Fisher G on AML32 | PASS_INPUT | observational recorded-fluorescence to future-locomotion geometry |
| row-partition R | DIAGNOSTIC_ONLY | held-out group predictive transfer only |
| anatomical A-to-B R | BLOCKED_SOURCE_TARGET_DEFINITION | no claim |
| causal R | BLOCKED_INTERVENTION | no claim |
| calcium-memory explanation of lag | FALSIFIED by AML18 control | no claim |

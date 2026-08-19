# Neural Riemannian metric independent-test research contract

Status: COMPLETE

PREDECESSOR: _workspace/ce/neural-riemannian-metric-validation-20260818

## 1. Research question

Find and rank genuinely independent ways to test the hypothesis that learning,
context, and sleep alter a neural state-space geometry which in turn constrains
later neural trajectories. The target chain is

$$
\Delta W^s \longrightarrow \Delta g(z,c)
\longrightarrow p(x_{0:T},\tau_B\mid x_0,c).
\tag{1}
$$

The research must not treat mathematical definability, retrospective fit, or
agreement between quantities estimated from the same activity as evidence for
the causal chain in (1).

## 2. Typed objects and claim boundary

- $W^s$: measured structural connectivity for identified units.
- $W^e$: estimated effective connectivity; it is not interchangeable with
  $W^s$.
- $z$: a training-only, frozen state chart with an explicit transformation law.
- $h$: the physical cortical-surface/laminar metric. It is an anatomical
  baseline and nuisance candidate, not automatically the computational metric.
- $g_0(z)$: the pre-existing developmental and connectivity-conditioned
  state-space metric field before the experimental manipulation.
- $g(z,c)\succ0$: a symmetric Riemannian metric field. A directed drift or path
  action is a separate typed object.
- $v(z,c)$ and $Q(z,c)$: drift and process-noise covariance in an explicit
  stochastic dynamics model.
- $\tau_B$: a first-passage random variable, not a deterministic function of
  geodesic distance unless a coupling law is separately assumed and tested.

The broad theory is not that every brain computation is Riemannian. The testable
claim is that a preregistered map $\Phi$ predicts metric deformation from
measured connectivity or intervention, and that this deformation adds
out-of-sample information about later trajectories beyond direct dynamics,
state covariance, firing rate, Euclidean geometry, gain, and noise controls.

## 3. Required independent route families

Search primary sources and derive operational tests for at least these families:

1. same-unit structural-functional longitudinal measurement;
2. causal synaptic or circuit perturbation with activity readout;
3. perturbational system identification and local response ellipsoids;
4. spontaneous fluctuation versus evoked-response consistency;
5. transition-path, committor, first-passage, and large-deviation tests;
6. closed-loop optimal-control energy and controllability tests;
7. representational generalization, psychophysics, and behavioral transfer;
8. state-dependent field, holonomy, and curvature tests;
9. anatomical surface, pre-existing baseline geometry, induced deformation,
   and flat-pullback dissociation;
10. topology-versus-metric dissociation and shortcut formation;
11. sleep, replay, and post-sleep renormalization interventions;
12. cross-modal, cross-task, cross-animal, and cross-scale triangulation;
13. natural-experiment, dose-response, mediation, and invariance tests;
14. synthetic ground-truth and mechanistic simulation recovery;
15. public datasets and feasible new-data technologies.

Routes which are algebraic re-expressions of the same fitted $J,Q$, reuse their
outcome in both predictor and target, or lack a distinct falsification outcome
must be marked dependent or circular rather than counted as corroboration.

## 4. Evidence and source rules

- Use primary papers, official data repositories, and official technology or
  consortium documentation. Record access date 2026-08-18.
- Separate demonstrated measurement capability from evidence for this theory.
- For each proposed dataset or technology, record scale, unit identity support,
  structural signal, activity signal, perturbation, longitudinal support, and
  the strongest claim tier it can test.
- Do not imply that a connectome, calcium movie, decoder, covariance metric, or
  manifold visualization alone tests (1).

## 5. Mathematical requirements

For every route specify inputs, estimator, chart law, target, nulls, nuisance
variables, sample unit, uncertainty, success criterion, and kill criterion.
Every longitudinal route must estimate $g_0$ before intervention and distinguish
relative deformation from physical cortical surface geodesics, cortical depth,
layer, cell type, wiring length, and a nonlinear-coordinate pullback of a flat
metric. E17's sessionwise constant SPD candidates must not be described as
having measured either anatomical folding or a nonconstant $g_0(z)$.
At minimum, distinguish these hypotheses:

$$
H_W:\ \Delta W^s\to\Delta g,
\qquad
H_G:\ \Delta g\to\Delta p(x_{0:T}),
\qquad
H_C:\ \Delta W^s\to\Delta g\to\Delta p(x_{0:T}).
\tag{2}
$$

An explicit generative bridge is required before using $g$ causally, for
example

$$
dz_t=-g(z_t,c)^{-1}\nabla V(z_t,c)\,dt
      +B(z_t,c)\,dB_t,
\qquad Q=BB^\top,
\tag{3}
$$

or a separately declared control law. Competing bridges and an unconstrained
$v,Q$ model must receive equal or greater parameter budgets.

## 6. Deliverables

- `10-sources.md`: primary-source capability and dataset map.
- `11-math.md`: identifiability, independence, and counterexamples.
- `12-routes.md`: ranked route portfolio with exact decision rules.
- `20-audit.md`: formal status and overclaim audit.
- `30-implementation.md`: implementation disposition or a scoped prototype.
- `31-validation.md`: validation record.
- `40-final-report.md`: concise Korean synthesis and recommended experiment
  sequence.
- Supporting tables and scripts belong under `artifacts/`.

## 7. Completion criteria

1. The route map covers all required families and collapses dependent variants.
2. Every Tier A recommendation distinguishes the hypotheses in its declared
   claim scope. Any route claiming $H_C$ must distinguish $H_W$, $H_G$, and
   $H_C$ from gain/noise/direct-dynamics alternatives; an $H_G$-only Tier A
   route is explicitly not a full-chain experiment.
3. At least one complete documented counterexample is supplied for every
   invalid shortcut. Executable fixtures are labeled as spot checks unless they
   implement the whole documented counterexample set.
4. Candidate public data are assigned only the claim tier their measurements
   support.
5. The final report identifies the smallest decisive experiment, lower-cost
   precursors, and explicit results that would make the theory lose credibility.

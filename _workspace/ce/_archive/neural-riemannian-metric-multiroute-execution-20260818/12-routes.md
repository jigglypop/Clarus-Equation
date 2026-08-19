# Frozen execution matrix and fallback order

Status: COMPLETE

## Eligibility ladder

Every route must receive one contract status before any numerical outcome is
opened: `ELIGIBLE`, `PARTIAL_DESCRIPTIVE`, `UNTESTABLE_MISSING_INPUT`,
`ACCESS_BLOCKED`, `INELIGIBLE_DEPENDENT`, or `FAILED_EXECUTION`.  `ELIGIBLE`
requires raw inputs, sample identity, an independent target, license,
provenance, and the split fields in `11-math.md`.  An aggregated source table
may reproduce a published descriptive number, but it cannot become an
individual-level test by treating rows/windows/spines as independent subjects.

Every route also preserves the geometry types frozen in `00-contract.md`:
physical/anatomical $h$, training/pre-period baseline $g_0$, and induced or
post-state $g_t$. Existing folds and layers are controls, not learned metric
deformation. A longitudinal route reports generalized-eigenvalue or
affine-invariant relative deformation only after the $h$/depth/layer/physical
distance/wiring length/cell-type and nonlinear flat-pullback controls that its
schema supports.

## Route-by-route execution order

| Rank | Route | Minimum input gate | Frozen split and target-aware degrees of freedom | Outcome disposition / fallback |
|---:|---|---|---|---|
| 1 | R-SYNTH | generated data only | 6 generators x 20 seeds; candidate set, `d=3`, horizons, seed and thresholds fixed in `11-math.md`; no target tuning | Execute first. Failure blocks biological metric promotion. |
| 2 | R-ALLOPTICAL | trial-level stimulation identities, calibrated amplitudes, response trials, separate unperturbed paths, animal IDs | response geometry uses stimulation training directions; target uses disjoint directions and unperturbed trials; leave animal out | Highest-information public `H_G` route. Without separated paths: `PARTIAL_DESCRIPTIVE`. |
| 3 | R-DANDI-37 | raw longitudinal activity, ROI tracking, animal/day IDs, next-day behavior or activity | early-day chart/metric only; later-day target; predeclare one horizon and one score | Activity-only longitudinal `H_G`. Without identity audit: descriptive. |
| 4 | R-GRID-TORUS | spike times, state labels, animal/module IDs | random half of each state block for topology; other half for local metric/transition; swap halves as a single robustness check | Topology-metric dissociation. Aggregated module/state summaries are descriptive. |
| 5 | R-SLEEP-E19 | subject IDs, stage windows, later encoding/trajectory target | stage predictor windows strictly precede target trials; leave participant out | Sleep association only. No subject linkage: descriptive. |
| 6 | R-SLEEP-E15 | animal/session identity, treatment/time fields, replay and later endpoint | treatment window before endpoint; animal-level blocked split | Sleep association only. Aggregate session labels fall back to published-table reproduction. |
| 7 | R-MICRONS | queryable registered EM-function cells, cell types, positions, functional target | freeze `Phi`; hold out cell pairs by cells/area; no shared neurons across train/test pair sets | Static bridge only. If only graph or only functional table is available: missing input. |
| 8 | R-BCI | decoder perturbation, baseline neural data, trial learning curves, subject/session IDs | metric calibration before perturbation; hold out perturbation class or participant/session | Current official derived subject tables are `PARTIAL_DESCRIPTIVE`; raw Dryad archive acquisition is blocked. Behavioral accessibility only. |
| 9 | R-CELEGANS | full edge list and node coordinates/types | blocked edge pairs, degree-stratified negative sampling, synthetic perturbations separate from fit | Run-local `PARTIAL_DESCRIPTIVE` structural fixture because source-object redistribution license is unresolved. No claim on living neural change. |
| 10 | R-E17-F3 | released source arrays plus independent labels and animal IDs | source-supported pairing, leave animal out | If pairing/endpoint not independent, reproduce descriptively only. |
| 11 | R-E17-F4 | raw released arrays with temporal/session and unit identity fields | strictly earlier proxy to later drift, animal held out | Array order is not time. Missing chronology means descriptive only. |
| 12 | R-E17-F5 | raw transitions with earlier proxy and later independent errors | early-to-late block, animal held out | No temporal provenance means descriptive only. |
| 13 | R-E17-F2 | predecessor lock files | none | Hash/validator reference only; mark dependent for evidence. |

## Family, tie and kill protocol

The sole target-aware selection is the route order above and the primary
contrast specified in `11-math.md`; no candidate, metric, horizon, chart,
regularizer, or endpoint may be substituted after opening a route outcome.
Each eligible biological route contributes one primary score contrast to the
Holm family defined in `11-math.md`.  Secondary baselines diagnose why a route
fails, but do not create additional confirmatory wins.  Ties are wins for the
simpler or direct-dynamics model; a metric route only survives on a strictly
positive, Holm-surviving animal-level score difference with compatible effect
direction.

The following are automatic kill conditions for that route's maximum claim:

- the source/target share a fitted `J,Q`, covariance, or response trial;
- cell or state identity permutation gives the same result;
- gain-only, noise-only, direct `v,Q`, flat-pullback, or baseline producer ties;
- the result vanishes under sample-level resampling or reverses across
  independent animals without a predeclared moderator;
- the claimed arrow lacks its typed source (for example static `W` for a
  `Delta W` claim, or distance substituted for a first-passage model).

## Aggregation fallback

When only an aggregate table exists, retain the exact released rows and report
the published or source-table effect with its table-supported uncertainty.  Do
not impute animal identifiers, reconstruct individual trajectories, or apply
Holm to pseudo-replicated windows.  The route status is
`PARTIAL_DESCRIPTIVE`, unless predictor and target are the same aggregate,
which is `INELIGIBLE_DEPENDENT`.  Its output may establish provenance,
calculation feasibility, or a direction compatible with a claim; it never
supplies independent evidence for M0.

If raw data are officially present but inaccessible, record URL, date, license,
schema/manifest, requested object, HTTP/access response, and use
`ACCESS_BLOCKED`.  If a raw payload lacks only a required typed field, use
`UNTESTABLE_MISSING_INPUT`; do not substitute a nearby dataset that changes the
frozen route.

## What the portfolio can establish

No public route here can independently establish the full causal chain unless
the same-unit longitudinal structural and intervention information happens to
be present in one frozen target.  The strongest realistic public outputs are a
validated estimator (`R-SYNTH`), perturbational or longitudinal `H_G`
components (`R-ALLOPTICAL`, `R-DANDI-37`), topology-metric dissociation
(`R-GRID-TORUS`), sleep associations, and static bridge feasibility
(`R-MICRONS`).  A failed partial route must not be repaired by combining its
outcome with another species, modality, or data source into an apparent
observed `Delta W -> Delta g -> Delta x` chain.

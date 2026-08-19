# Neural operational metric validation routes

Status: COMPLETE

Routes are selected by verified data granularity before outcome analysis.  They are not interchangeable and may only be downgraded when a required field is absent.

## Tier A: direct connectivity plus causal perturbation

**Eligibility.** Same identified cells/synapses before and after learning/plasticity intervention; direct $W^s$ measurement, independent activity calibration for $A,J,Q$, future single-trial trajectories, behavior, intervention/sham labels, animal/session IDs, and a complete provenance manifest.

**Target-aware outcome.** This is the only route eligible for `NRM-H1A`, and it may test mediation only with a randomized/validly instrumented $W$ perturbation and the assumptions stated in `11-math.md`.

**Estimator and dof.** Outer hold out complete animals; fit matching/chart/$F,J,Q,H,\lambda$ only on train animals.  Unit is animal; sessions/trials/cells are nested repeats.  Confirmation dof are held-out animals minus the declared group-level comparison parameters.  If fewer than the preregistered number of animals survive quality filtering, report feasibility or `UNTESTABLE`.

**Multiplicity and kill tests.** The six-claim Holm family applies.  K1--K8 apply; additionally failure of intervention compliance, cell identity continuity, or direct connectivity reliability kills H1A/mediation.  A context/gain-only intervention that matches trajectory effects kills a $W$-specific mechanism.

## Tier B: longitudinal activity and calibrated effective dynamics

**Eligibility.** Same population has longitudinal activity, independent calibration data to estimate $J,Q$, future held-out trials and behavior, with animal/session/trial IDs.  No direct pre/post synaptic connectivity is claimed.

**Target-aware outcome.** `NRM-H1B`, H2, and where designed H3/H4/H5 can be assessed as predictive effective-dynamics claims.  It cannot upgrade to H1A or causal mediation.

**Estimator and dof.** Use nested animal/session split as specified in `11-math.md`, group-aggregated proper scores, and all direct-dynamics/parameter-matched SPD baselines.  The independent group count is the only confirmatory $n$; a single animal is technical feasibility only.

**Multiplicity and kill tests.** H1B/H2--H5 remain in the same six-claim family.  K2 (test-set tuning), K3 (baseline tie), K5 (shuffle/reversal match), K6 (gain/noise match), K7 (unlinked studies), or K8 (identity mismatch) each downgrades the affected claim.  No favorable H1B score licenses a causal/spatial conclusion.

## Tier C: condition means, representational geometry, or cross-session public data

**Eligibility.** Different populations across conditions, subject/session means, representation/behavior summaries, or source data without the synchronized fields required for A/B.

**Target-aware outcome.** Reproduce a named descriptive geometry or behavioral figure at its published unit.  It may assess data/schema feasibility and generate a future Tier A/B protocol.  It cannot test equation (1), H1A/H1B/H2 mediation, or a population causal metric claim.

**Estimator and dof.** Preserve the paper's independent unit and exclusions.  There are no newly created subject-level dof from averages, windows, or synthetic reconstruction.

**Multiplicity and kill tests.** One paper-matched reproduction endpoint only; all extensions are exploratory/`UNTESTABLE`.  Failure to reproduce with fixed preprocessing is a reproduction failure, not a license to choose a new chart/metric.

## Acquisition/split ledger required for every route

Before analysis record official release/version/license, local path and SHA-256, schema, animal/session/cell identifiers, intervention and condition labels, temporal ordering, exclusion count, outer-fold assignment, and frozen candidate family.  No cell matching, chart alignment, hyperparameter, target, or route decision may use held-out trajectories.  Combining $\Delta W$, $\Delta g$, and $\Delta x$ from different datasets is Tier C descriptive triangulation, never a test of one chain.

## Candidate-universe v2 routes

The frozen registry adds a route before biological claim adjudication: every formula first receives a mathematical and field-availability verdict.  A missing input is not replaced with activity from another figure or with an identity matrix unless the registry explicitly defines that identity case as a non-biological baseline.

### Route R1: E17 uncertainty geometry discovery

Use `S0`--`S5`, `S12/S13` and the direct full/diagonal/isotropic baselines on the Figure 2 branch trials.  Keep the affine mean dynamics common, choose tuple hyperparameters only from outer-train animal inner scores, and evaluate Gaussian NLPD on the held-out animal's final trial block.  Aggregate sessions inside animal before comparing animals.  This route ranks covariance shapes for technical prediction; it cannot establish a spatial mechanism or `NRM-H2` with $N=3$ opened animals.

### Route R2: deformation, observability and condition information

Use `S6/S7/S14/S15` only on their registered finite-time pair-separation targets.  Pair initial states without looking at endpoints, fit any scalar calibration in inner data, and evaluate endpoints once.  `S8/S9` use saline/DCZ condition labels, not a task or behavior label.  A decoder success is condition information and cannot be counted again as trajectory or connectivity evidence.

### Route R3: graph, directed action and distribution diagnostics

Build `G1/G2/G3a/G3b` only from the frozen effective-$J$ mappings and report disconnected or insufficient-pair failures.  Use held-out correlation/lag association as a technical diagnostic, with animal rather than node-pair degrees of freedom.  Evaluate `D1` with the full Gaussian NLPD against reversed and time-shuffled paths.  Compute one empirical `P1/P2` Wasserstein class with the frozen ground metrics and trial-block permutation.  None of these objects may win an SPD leaderboard because their types and endpoints differ.

### Route R4: locked continuation

After E17 discovery, select at most one candidate per typed primary endpoint using only the registered rule.  Freeze its complete tuple and code hash, then acquire a new cohort with enough independent animals.  For `NRM-H1A`, the cohort must add same-synapse direct $W^s$, intervention/sham, independently estimated $A,Q$, and later single-trial trajectories.  A failed lock starts a new preregistration generation; the failed cohort is never reopened to select another tuple.

### Immediate kill and resume conditions

Type mixing, test-driven tuple expansion, session-weighted pseudoreplication, reuse of the opened E17 test as a lock, or a flexible/direct baseline tie kills a winner claim.  Missing $B,R_u,Q_x$, a calibrated continuous-time SDE, an identifiable Finsler one-form, a nonquadratic potential, or a smooth state-dependent metric yields `UNTESTABLE_MISSING_INPUT`, not a numerical failure.  Those candidates resume only when the named fields are measured in the same units and time axis.

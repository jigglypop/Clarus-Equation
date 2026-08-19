# Viable public-data routes

Status: COMPLETE

These routes are alternatives chosen by data granularity, not opportunities to search for a favorable result.  Select the first route whose preregistered acquisition audit confirms the stated inputs.  A failed route is reported as failed; it does not authorize changing the claim.

## Route A: E15 event-level raw replay data

**Eligibility.** Officially released raw/processed spikes with animal, session, timestamp, unit-quality, SWR/replay-event information and condition labels; all objects have an acquisition receipt, license, schema and SHA-256 manifest.

**Target-aware claim.** RR-R1 may reproduce the paper-matched sleep-deprivation replay direction.  RR-H1 and RR-H2 may be tested only here, because this is the only route that can place replay and routing statistics in the same animal/session/window.

**Protocol.** Freeze event exclusions and paper replay endpoint first.  On training animals only, freeze quality threshold, fixed unit-set rule, non-overlapping window size/bin, branch estimator, entropy state map/smoothing, covariates, and a thinning/nonstationarity sensitivity grid.  Hold out complete animals (or, only if one animal per independent session is documented, sessions) and keep every session wholly on one side.  Score at group level; cluster uncertainty by animal and session.

**Confirmatory degrees of freedom.** Independent groups minus fitted group-level model parameters, reported explicitly.  Windows have zero additional confirmatory degrees of freedom.  If the group count cannot support the prespecified model, RR-H1/H2 are `UNTESTABLE`, not negative or positive.

**Multiplicity.** Six-test family from `11-math.md`; this route does not get to choose the better of $B$ and $H$ after seeing $Q$.  Bin/window/state alternatives are training-only tuning or exploratory diagnostics.

**Kill tests.** Fixed-unit and thinning sensitivity removes the sign; nonstationary-window rule removes the sign; held-out covariate-only model ties; group leakage is found; or replay endpoint cannot be reproduced.  Any one kills the corresponding confirmation claim.

## Route B: public spike/event data with sleep labels but no matched replay endpoint

**Eligibility.** Official raw event series with animal/session/time labels and sleep state, but no validated per-window replay score paired to the same recordings (for example a criticality replication source).

**Target-aware claim.** Estimate only the registered routing/criticality summary reproducibility at its published unit.  It can assess estimator feasibility, rate/subsampling sensitivity, and state-label association.  It cannot test RR-H1/H2 because $Q_{asw}$ is missing, and cannot stand in for E15 replay.

**Protocol and dof.** Use paper-matched session/animal endpoint; group split by animal/session when fitting any tuning choice.  The independent unit is the published independent group, never bins/events.  Use the same fixed-unit and stationarity checks as Route A.

**Multiplicity.** One paper-matched reproduction endpoint only.  Any added threshold/bin/state comparison is exploratory and explicitly counted.

**Kill tests.** Inability to reproduce the published endpoint after paper-matched preprocessing, missing group identities, or rate/unit-yield sensitivity large enough to reverse interpretation.  A pass here still leaves H1/H2 `UNTESTABLE`.

## Route C: participant-level geometry/sleep outcomes (E02 or E19)

**Eligibility.** Official participant/animal-level source data or raw activity with subject IDs, behavior/item/category outcomes, sleep measures, baseline and exclusions.  E02 needs task-relevant geometry plus behavior; E19 needs separate item/category outcomes plus REM, SWS, total sleep and baseline.

**Target-aware claim.** Test RR-H3 from E02 and/or RR-H4 from E19 only at participant/animal level.  This route does not test replay-routing H1/H2 and does not license the RR-H5 cross-study bridge.

**Protocol and dof.** Reproduce the published statistic before extension.  Fit all feature/tuning choices inside training subjects and hold out complete subjects.  For H4 use sleep composition plus total dose, not ratio alone.  Independent dof are held-out subjects minus declared model parameters; trials/items are repeated measures and require participant-clustered treatment.

**Multiplicity.** H3 is one family member.  H4 has two named outcomes (item and category) and remains in the six-test family.  E02/E19 route choice is determined by data availability, not effect direction.

**Kill tests.** Group-summary-only release; behavior/item/category data not linked by participant; no held-out gain over baseline/trial-count/activity-dimension controls; label/ratio permutation signal comparable to observed; or outcome-specific effect fails Holm adjustment.

## Route D: figure source-data or code-only release

**Eligibility.** Official source-data tables sufficient only to reproduce a published figure/statistic, or official code without the underlying records.

**Target-aware claim.** Reproduce only the stated figure/statistic and uncertainty at its published aggregation.  No individual-level association, predictive increment, group split, or causal claim is identifiable.

**Protocol and dof.** Preserve paper unit/exclusions exactly.  There is no new confirmatory dof beyond the supplied aggregate table.

**Multiplicity and kill test.** One deterministic reproduction target; all extensions are `UNTESTABLE`.  A mismatch after byte-level source-data/code reconciliation is a reproduction failure, not a reason to retune the definition.

## Route-selection record

Before downloading/analyzing outcomes, record official URL, release/version, license, file class, identifiers, expected independent groups and the selected route.  A route may move only downward with missing granularity (A to B/D, C to D); it may never be promoted by synthetic fixtures, article figures, or a favorable result from another dataset.

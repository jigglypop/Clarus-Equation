# Implementation record

Status: COMPLETE

## Frozen executions

Two eligible bounded routes were implemented, independently audited before
their first outcomes, hash-locked, executed once and then result-locked.

- `R-SYNTH` uses 120 independent paired train/test circuits, six generators,
  held-out e3 interventions and six blind candidate families. The raw trace
  contains paths, W, densities, seeds, directions, true meshes and
  first-passage outcomes. The validator recomputes linkage, scores, p-values,
  Holm decisions, recovery and numerical curvature rows. V1 and V2 remain
  explicitly invalid in `artifacts/synthetic-v1-invalidation.md` and
  `artifacts/synthetic-v2-invalidation.md`.
- `R-GRID-TORUS` uses the official Gardner Q/R-day2/S NPZ files. A seeded 20%
  of wake blocks fits the sole scaler/PCA chart and enters no endpoint.
  Primary and swapped topology/mobility roles are disjoint. The validator
  rebuilds the entire result from raw NPZs and official interval literals.
  V2 remains invalid in `artifacts/grid-v2-invalidation.md`.

The preexecution locks are
`artifacts/synthetic-v3-preexecution-lock.json` and
`artifacts/grid-v3-preexecution-lock.json`. The immutable output hashes and
scientific dispositions are in `artifacts/one-shot-result-lock.json`.

## One-shot outcomes

Synthetic v3 has integrity PASS but scientific-gate FAIL: G1 recovery was
15/20 rather than 18/20, G3 pullback-c recovery was 15/20 rather than 18/20,
and non-G1 any-Holm selections were 72/100 rather than at most 5/100. G1 did
beat all five controls after Holm, and the numerical flatness fixture had
0/20 false positives, but those components cannot override the failed suite.
The exact audit is `artifacts/synthetic-v3-postexecution-audit.md`.

Grid v3 has raw-recomputation integrity PASS and a partial descriptive result.
Only R3-REM met the frozen topology-below-noise plus mobility-above-noise rule
in both disjoint roles, for 1/12 REM/SWS module-state comparisons. A post-run
exact eigenvalue-log decomposition found that the median squared AIRM was
97.6% common scale rather than directional shape. The exact audit is
`artifacts/grid-v3-postexecution-audit.md`.

## Descriptive routes

`artifacts/run_descriptive_routes.py` and its validator materialize all 13
route dispositions without promoting aggregate rows to subjects. The current
ledger records E17 Figure 3's official distance transform and Spearman
summaries, E17 Figure 4/5 schemas, the dependent Figure 2 reference, the E19
and E15 predecessor references, a 7,379-edge C. elegans structural inventory,
and the BCI/MICRONS inventories. E15's nested object is explicitly 104 rows,
13 session labels, six NSD sessions and seven SD sessions. These are all
descriptive or input dispositions, not independent metric tests.

## Claim boundary

No implementation measures physical/anatomical fold geometry h, structural
change `Delta W`, a longitudinal local field `g0 -> gt`, and an independent
future trajectory in the same units. Grid supplies a pooled mobility
precision, not a C2 local metric field. Synthetic data validate only an
estimator when their full suite passes; this suite did not. Existing folds are
therefore neither ignored in the theory nor controlled in the executed data.
The HCP-YA, ABCD and OpenNeuro ds006072 anatomy-aware follow-ups are catalogued
in `10-sources.md` outside the frozen outcome universe.

## Executed commands

```powershell
python -B artifacts/verify_synthetic_suite.py --curvature-fixture
python -B artifacts/verify_synthetic_suite.py --candidate-sign-fixture
python -B artifacts/verify_synthetic_suite.py --force-fixture
python -B artifacts/run_synthetic_suite.py
python -B artifacts/verify_synthetic_suite.py

$env:PYTHONPATH = 'C:\Users\dongh\AppData\Local\Temp\clarus-nrm-deps-py311'
python -B artifacts/run_grid_torus.py --fixtures
python -B artifacts/run_grid_torus.py
python -B artifacts/verify_grid_torus.py
python -B artifacts/run_descriptive_routes.py --overwrite
python -B artifacts/verify_multiroute_results.py
```

The synthetic verifier's assertion failure is the intended machine expression
of a frozen scientific gate failure, not an execution crash. No scientific
code, seed, candidate, threshold or result was changed afterward.

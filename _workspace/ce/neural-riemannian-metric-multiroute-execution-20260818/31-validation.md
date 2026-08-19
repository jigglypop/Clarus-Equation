# Focused validation record

Status: COMPLETE

## Source and fixture checks

All synthetic and Grid scripts parse. Synthetic non-outcome fixtures returned:

```text
PASS_CURVATURE_FIXTURE: Euclidean 0; nonlinear pullback 2.738e-09;
known conformal field 3.60000002
PASS_SIGN_FIXTURE: exp(-beta h) identity exact
PASS_FORCE_FIXTURE: observed-y force pushforward exact
```

Grid's strict-bin, block-local-difference, disjoint calibration and symmetric
zero-distance fixture returned `PASS`. Both preexecution locks matched every
listed source and input hash before the first outcome.

## Synthetic v3

The first create-only run completed in 67 seconds and wrote the JSON result
and a 42,383,944-byte compressed trace. The result and trace hashes are
`5809c2df...17a9` and `3ab5657e...4bde`. The frozen verifier stopped at its
first scientific assertion because G1 recovery was 15/20. A separate read-only
status audit completed all remaining recomputations and found:

```text
integrity                         PASS
G1 beta recovery                 15/20  FAIL (need >=18)
G1 predictive Holm               5/5    component PASS
G2--G6 any-Holm                  72/100 FAIL (need <=5)
G3 c recovery                    15/20  FAIL (need >=18)
G3 numerical curvature FP        0/20   fixture PASS
```

The failed gate is preserved; the simulation was not rerun.

## Grid v3

The first raw run completed in 42.7 seconds. The independent validator then
spent 34.7 seconds rebuilding all six modules from the raw NPZs and returned:

```json
{"status":"PASS","schema":"nrm-grid-torus-v3","modules":6,"recomputed":"raw_npz"}
```

All Q1/Q2/R1/R2/R3/S1 modules were complete. The bounded heuristic was true
only for R3-REM: 1/6 REM, 0/6 SWS and 1/12 overall. This is an integrity and
reproducibility PASS, not a biological metric PASS.

## Descriptive and predecessor checks

The 13-route disposition validator returned:

```json
{"status":"PASS","routes":13}
```

The archived sleep real-data verifier separately returned
`OK realdata: 574 hashed files; E15/E19/E13 checks passed`. E17 Figure 3's
four source Spearman calculations are finite and labelled nested row-level
descriptions. E15's processed schema is 104 rows and 13 session labels.

## Result lock

`artifacts/one-shot-result-lock.json` was checked against all four locked
artifacts and returned `PASS`. No bare pytest, full benchmark, packaging build
or irreversible V5 stage was run.

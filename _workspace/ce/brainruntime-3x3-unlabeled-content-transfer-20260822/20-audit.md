# BA-TR26 audit

Status: `PASS`, after one receipt correction.

The stable read-only audits found no mathematical or implementation P0.  The
first implementation did execute the synthetic target probe but wrote
`endpoint_opened=false`.  This metadata defect was repaired to `true` at both
row and aggregate level, the focused tests gained a regression assertion, and
calibration/development artifacts were regenerated.  Confirmation remains
sealed.

The second correction narrows provenance language: each seed permutes one
fixed positive equal-norm content template.  The run therefore tests
coordinate/content permutation, not broad generalization over independently
varied content geometries.

The generic learner accepts only raw cue and observed content matrices.  The
compiler accepts only a cue, arrived coordinates, current weight columns, and
response coordinates.  Factor names, grid labels, targets, decoder values,
rewards, stores, and the held-out content are absent from those APIs.

`artifacts/source-freeze.json` binds the contract, calibration/development
inputs and outputs, experiment sources, focused test, runtime dependencies,
Python hook, interpreter, and dependency versions.  Independent SHA-256
recomputation returned zero mismatches.

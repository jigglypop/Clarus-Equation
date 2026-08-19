# C1 implementation record

Status: COMPLETE

## Implemented route

Only frozen route R1 was implemented.  The implementation lives in
`reality_stone/python/reality_stone/clarus/runtime_prediction_guided_metacontrol.py`
with a separate benchmark CLI and focused test file.  `RuntimeAgent` and the
predecessor Loop-10 implementation were not modified.

The harness constructs independent warm snapshots, fits a float64 ridge map
on the frozen fit split, audits it on the disjoint predictor split, and then
computes all three policy forecasts algebraically.  No policy candidate is
executed during selection.  Each arm restores the same episode snapshot and
executes exactly one selected drive through `BrainRuntime.step`.

## Intervention arms

- `intact` passes action-labelled forecasts to the planner.
- `edge_shuffle` cyclically deranges only the forecast-to-action mapping at
  the planner port.
- `readout_shuffle` preserves the planner input and changes only the emitted
  display copy.
- persistence, balanced-random, error-magnitude-only, and reactive-mean-effect
  controls use their frozen information boundaries.

Every episode records the pre-map forecast tensor and costs, planner mapping,
selected action, actual drive, transition count, loss, and integrity fields.
The full `BrainRuntimeSnapshot` hash includes transient tensors, config,
hippocampus, STDP/delay option state, mode occupancy, circadian fields, and
brainwave history.

## Stage and publication guards

Public single-seed and range APIs reject every official confirmation seed
before computation.  Confirmation can run only through `run_c1_stage` after a
manifest verifier independently checks the development artifact, exact seed
block, canonical finite result hash, source hashes, environment, regenerated
summary, and development `GO`.

## Focused validation

Command:

```text
.codex\hooks\python.cmd pytest tests\test_runtime_prediction_guided_metacontrol.py -q
```

Result: `6 passed` in `3.99 s` on system Python 3.11.9 / Torch 2.12.1 CPU.
The tests use a reduced, explicitly nonfrozen configuration for mechanics and
do not open development or confirmation outcomes.

No development seed has been executed at this point.  The implementation
requires a stable-snapshot audit before the frozen development stage.

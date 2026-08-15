# Implementation

Status: COMPLETE

## New isolated units

- `reality_stone/python/reality_stone/clarus/local_cloud_kernel.py`
  - exact typed local/private and shared observations;
  - synchronous local/cloud Jacobi transition;
  - bounded bilinear interaction;
  - live weighted-sup small-gain certificate;
  - exact finite composition and true transition lesions;
  - exactly twenty recurrent state features and no decision/readout bypass.
- `reality_stone/python/reality_stone/clarus/local_cloud_benchmark.py`
  - balanced local-identity by shared-context task;
  - full/local-only/cloud-only/no-memory arms with twenty state features each;
  - train-only frozen ridge readout;
  - intact-readout transition lesions;
  - no import-time execution and no registered seed run.
- `tests/test_local_cloud_kernel.py`
- `tests/test_local_cloud_benchmark.py`
- `reality_stone/python/reality_stone/clarus/__init__.py` exposes only the kernel types; the
  benchmark runner remains opt-in and import-safe.

## Boundary

This replaces the failed repeated-weak-shell benchmark route. The old negative artifact remains
preserved for provenance. No default agent runtime, V8 locked split, or confirmation artifact is
opened. The small discarded diagnostics are recorded only in `artifacts/` and are not evidence.

# Implementation

Status: COMPLETE

## Restored frozen parent

The sparse causal bridge lineage was absent from current `main`, so the exact
V1--V6 source/config/test lineage and the V4/V5 validation and integrity
artifacts were restored from historical branch `itself` at `33836b8`.  The
frozen source hashes match the historical attestations:

- `sparse_causal_bridge.py`: `0885d7244c3ea35367987ec59538d15c081d3ba6009897e5d0e5e42a24538ca7`
- `latent_causal_bridge.py`: `40306162c5d266a8ecb80d882202afb92fbc45aa5bc467a1004721706e57eef7`
- `free_rollout_bridge.py`: `13d38836f3fef8ef6cbad35bb5b79fc41e15b21ebf00eec5d61604e7b3832cd6`
- V4 validation artifact raw hash: `41c17778...` (historical CRLF lock)
- V5 validation artifact raw hash: `6dd4999e...` (historical CRLF lock)

`.gitattributes` now records the required LF source/config and CRLF historical
artifact boundaries.  The historical 29 focused tests passed, and the V5
validation failure replayed exactly with four false registered checks, H20
error `0.3330829265726758`, adaptive-dense H20 error
`0.3126973530814967`, and zero future reads.

## Registration-before-implementation

`experiments/preregistration/sparse_causal_bridge_v7.json` was created and
hash-recorded before the V7 implementation.  It fixes:

- one symmetric-consensus route and no replacement after validation;
- 96 validation seeds `77100..77195` and 96 locked-test seeds `78100..78195`;
- one H20 origin after the observed prefix through index 80;
- training-only normalization, seed-level paired inference, symmetric dense
  and no-sparse controls;
- a conjunctive sparse-contribution, parent-repair, persistence, dense-control,
  leakage, finiteness, and pathwise-stability gate;
- test opening only after an unchanged passing validation artifact.

Registration locks:

- raw: `134ddaa793170b898649b79e11407c10f35d1468ba95701544a06905d9448c3e`
- merged raw: `3cfa4ddc9bb6ab04bb7b37403780ef2fd4a894d26e7c45c1c84e062434fb4259`
- canonical: `2d1c06cb9259e52e435e28017b82d89924c4c305c0dc81b29beadf78ede13365`

## V7 implementation

Added:

- `reality_stone/python/reality_stone/clarus/reliability_rollout_bridge.py`
- `examples/agi/reliability_rollout_bridge_gate.py`
- `tests/test_reliability_rollout_bridge.py`

The implementation exposes only a prefix reader, instruments maximum state
index and future reads, fits every consensus controller by the same normalized
prefix rule, runs sparse/no-sparse/same-probe-dense controls, computes paired
seed confidence intervals, checks every dynamic component path, and refuses a
locked-test run without an unchanged all-clause validation artifact.

Implementation lock:

- module: `7abf17f260f0046cb6eace7ed57e1115657c2dd4d32bd1024bc7c1940e910310`
- test definition: `866e9e89274419b17e4b33a63df519c89565e6763480b7c8537b5f7b0ec88041`

`ruff format` was applied to the two new Python files and `ruff check` passed.
The three historical bridge suites plus the new suite passed 39 tests before
validation.

## P1 reproducibility repair

Four locked local-memory artifacts referenced by the active verifier were
missing from current `main`.  Their exact files were restored from the parent
of deletion commit `17a3d27`:

- `local_memory_aml32_preregistration.json`
- `local_memory_aml32_h1_confirmatory.json`
- `local_memory_aml32_h6_confirmatory.json`
- `local_memory_aml32_proof.json`

After restoration, the local-memory verifier and all four sparse bridge suites
passed 44 focused tests.

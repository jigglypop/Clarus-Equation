# Implementation

Status: COMPLETE

`BrainRuntime._step_torch` now writes the full source-qualified presynaptic
packet to the delay ring and reads it before overwriting the slot. The no-delay
branch is unchanged. `backend=auto` selects Torch when delay is enabled;
explicit Rust delay raises before cell execution, including after config
mutation. `tests/test_runtime_delay_events.py` adds the exact emission/arrival,
snapshot, and backend witnesses.

The development runner received one P2-only revision after the frozen
`ApparatusInvalid` exception aborted before writing a receipt. Revision 1 only
catches that pre-existing fail-closed condition and preserves partial rows; it
does not alter equations, routes, seeds, thresholds, or endpoints.

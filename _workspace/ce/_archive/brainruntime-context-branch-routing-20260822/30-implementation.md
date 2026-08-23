# Implementation

Status: COMPLETE

Revision 2 is implemented in:

- `reality_stone/python/reality_stone/clarus/runtime_context_branch_routing.py`
- `reality_stone/python/reality_stone/clarus/runtime_context_branch_routing_benchmark.py`
- `tests/test_runtime_context_branch_routing.py`

The implementation uses the actual delayed Torch `BrainRuntime`, its sealed
snapshot, heterogeneous neuronwise thresholds, event-time packet ring, sparse
recurrent matrix, and bounded recurrent write.  `ExactDelayEligibility`
accumulates row-post/column-pre local products separated by exactly two calls.
One projection/write occurs after all eight experienced trajectories; the
actual delta is masked to the four declared block maps and is exactly zero
outside them.

Recall creates a fresh runtime from one sealed zero-store snapshot for every
trial.  It injects two source payloads at call zero and zero external input
thereafter.  `_rollout` receives a cue and weight masks but no context.  The
pure mask compiler is the only state-construction boundary that receives the
context bit.  The decoder reads only the common $Y$ block.

Revision history is retained under `revisions/`.  Revision 0's extra relay
failed to emit; Revision 1 removed that relay; Revision 2 added explicit
pre-endpoint receipts without changing any scored endpoint.

Frozen Revision 2 source hashes are recorded in
`artifacts/source-freeze-r2.json` (SHA-256
`9b68715a724c06cf51e78364cf5a4cd83462e1b18ffdbcaf5a7bfd5e184ff302`).


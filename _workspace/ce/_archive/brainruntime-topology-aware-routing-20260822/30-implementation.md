# Implementation

Status: COMPLETE

The implementation is isolated in
`reality_stone/python/reality_stone/clarus/runtime_topology_routing.py` with
focused checks in `tests/test_runtime_topology_routing.py`.  Existing M1, T1,
and core runtime files were not changed for this route.

`construct_route_mask(weight, cue, blocks, seed, route, budget)` is the sole
mask-construction boundary.  Its signature excludes targets, labels, decoder,
post-rollout state, and endpoints.  It implements `FULL`, `WEIGHT`, `CLUSTER`,
`PATH_ONLY`, `TOPOLOGY`, `RETURN_SHUFFLED`, `RANDOM_MATCHED`, and
`WRONG_CONTEXT`, deterministic ties, exact sparse budgets, diagonal removal,
and fail-closed degenerate support.

Training uses the unchanged delayed signed eligibility and block-end M1 write
on a Torch `BrainRuntime`.  The experiment runtime enables delay length 2 and
the frozen heterogeneous threshold vectors.  STP, refractory state,
lifecycle selection, and the delay ring remain live.  Every route is applied
only to a clone of one sealed learned snapshot.

Main-agent review corrected four issues before development:

1. downstream clusters now exclude every cue-active source block;
2. exposed edges are counted from the pre-step lifecycle source mask;
3. exposed-edge and switch-cost denominators use the original learned graph;
4. the topology unit fixture now forces a different `PATH_ONLY` mask.

The source also provides shared-snapshot circuit execution and a pairwise M1
binding baseline.  No confirmation runner was added or executed.

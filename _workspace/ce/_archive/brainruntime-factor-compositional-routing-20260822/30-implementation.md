# Implementation

Status: COMPLETE

BA-TR5 is implemented as a new isolated module. Neither the frozen BA-TR3 branch apparatus nor BA-TR4 gate module was modified.

The implementation uses two independent BA-TR3 20-dimensional snapshots and composes their masks by a block-diagonal direct sum. Each factor gate stores a $2\times2$ local accumulator and a two-entry exposure count. Its frozen matrix is exactly the columnwise mean $\Theta=C/n$. The compiler accepts only `(gate_snapshot, factor_cue, weight, blocks)` and selects one 12-edge factor mask. Pair recall therefore uses 24 edges; `FULL_32` alone uses 32.

Physical gate experience contains twelve rows: four payload repetitions for each of `00`, `01`, and `10`. No output pulse, decoder read, endpoint read, or `11` row occurs. A four-action joint lookup is constructed only as a preflight receipt; its `11` column stays zero and tied, so its endpoint is never opened.

The paired endpoint is an exact Cartesian conjunction of twelve ordered-distinct payload trials from factor A and twelve from factor B. Per-trial outcomes are retained as deterministic hashes while aggregate counts remain explicit, avoiding a multi-megabyte redundant Cartesian JSON expansion without changing any score or formula.

Files:

- `reality_stone/python/reality_stone/clarus/runtime_factor_compositional_routing.py`
- `reality_stone/python/reality_stone/clarus/runtime_factor_compositional_routing_benchmark.py`
- `tests/test_runtime_factor_compositional_routing.py`
- `artifacts/source-freeze.json`
- `artifacts/development-results.json`

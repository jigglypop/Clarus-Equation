# Implementation

Status: COMPLETE

## Scope

BA-TR3 remains unchanged. The new implementation is isolated in:

- `reality_stone/python/reality_stone/clarus/runtime_context_learned_gate.py`
- `reality_stone/python/reality_stone/clarus/runtime_context_learned_gate_benchmark.py`
- `tests/test_runtime_context_learned_gate.py`

The module reuses the frozen TR3 recurrent learner, branch masks, decoder, and zero-store rollout. `LocalContextGate` owns a separate $2\times4$ matrix and updates it with `outer(branch_use, context_cue)`. Gate experience contains only $S_b$ and $H_b$ pulses. It contains no $Y$ pulse, decoder access, endpoint access, reward, target matrix, or mask label.

`compile_learned_mask` has the exact signature `(gate_snapshot, context_cue, weight, blocks)`. It validates the frozen matrix, computes `theta @ cue`, rejects a nonfinite result or tie, selects one actuator, and opens that branch plus the common output trunk. It has no closure and receives no seed, task mapping, schedule, payload, target, decoder, or endpoint.

## Anti-oracle implementation

Before endpoint scoring, the implementation checks an independently recomputed frozen-$\Theta,q$ action, cue-swap equivariance, a finite row-swapped-$\Theta$ counterfactual, seed/$\sigma$/schedule-metadata invariance, function signatures and AST identifiers, exact 12-edge masks, shared trunk, branch-use separation, source snapshot immutability, and gate digest immutability.

Revision 1 added these receipts before the development run. The first focused test exposed only an `inspect.getsource` indentation error for a class method; `textwrap.dedent` repaired that harness. No equation, threshold, decoder, delay, horizon, mask budget, seed, or endpoint changed.

## Frozen source

`artifacts/source-freeze.json` records the interpreter and exact hashes. The final frozen source hashes are:

- module: `51cf4502cad2dcb39f6b469a6247001b47089b39556be131e95651376c5e63c9`
- benchmark: `5b41b8542f54992b3f4864f810b6b9193d284db93158b2066eae2d3e5544dc98`
- focused test: `82a34f905d3256197b16266f6e17706e857a4ac7901dbf27835539debdf6fde6`
- source-freeze receipt: `540086950be848437511cf1eed94547af6219bac89c296a4e637a4ebdb271116`

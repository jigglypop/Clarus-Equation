# Routes

Status: COMPLETE

## R1 — discrete Torch branch (selected)

- independent float64 NumPy implementation of equations H3--H6a with immutable
  state copies for every perturbation;
- exact one-tick comparison with frozen `BrainRuntime._step_torch`;
- all-block interior Jacobian versus central finite difference;
- one-sided directional tests at runtime-reachable clip faces, with unreachable
  upper faces kept only as scalar primitive diagnostics;
- bit/TopK/lifecycle crossings reported as discrete events;
- ring arrival and previous-lifecycle one-tick lag;
- neuron-permutation equivariance outside ties, separated into a float64 mirror
  gate and a float32 Torch gate.

Claim ceiling: `DISCRETE_HYBRID_SPEC_PASS / EMPIRICAL_UNTESTED`.

## R2 — backend boundary (selected adverse route)

- no-delay, no-tie Torch/Rust parity must pass;
- delay-on parity must reproduce the preregistered mismatch because Rust has no ring
  buffer/index input;
- result status is `RUNTIME_DELAY_PARITY_BLOCKED`, not a repaired PASS.

## R3 — Rust delay repair (deferred implementation route)

Requires an explicit user-authorized runtime code change: add ring buffer/index to the Rust
API, snapshot/restore and kernel update order, then rerun the exact R2 fixture. It must not
be approximated by setting `axon_delay=False` while retaining a delayed claim.

## R4 — full runtime learning/memory hybrid (deferred)

Adds retrieval/encoding, STDP weight state, auto-mode, F1 budget, sleep variables and seeded
noise to the augmented state. R1 success does not validate this larger map.

## R5 — empirical/anatomical bridge (blocked)

Requires actual identity, delay, strength, threshold calibration, intervention and anatomy
receipts. Synthetic branch derivatives are not substitutes.

## Stop/revision order

1. preflight already killed continuous saltation; do not restore it.
2. run R1/R2 once with frozen source/config/tolerances.
3. formula mismatch permits one equation revision with all tests unchanged.
4. Torch source mismatch permits only a source/config apparatus correction.
5. delay backend mismatch remains an honest blocker until R3 is separately authorized.

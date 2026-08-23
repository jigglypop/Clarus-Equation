# Validation

Status: COMPLETE

Result: `NEURONWISE_THRESHOLD_RUNTIME_PASS / SCALAR_BACKWARD_COMPATIBLE / RUST_VECTOR_BIT_UNSUPPORTED_FAIL_CLOSED / EMPIRICAL_UNTESTED`.

Command:

```powershell
.codex\hooks\python.cmd pytest tests/test_runtime_neuronwise_thresholds.py tests/test_runtime_contracts.py::test_brain_runtime_rust_backend_matches_torch_cell_step -q
```

Result: `8 passed`. Validation covered vector canonicalization and malformed
inputs, live scalar mutation, exact scalar-broadcast equivalence, the fixed
three-neuron bit/eligibility witness, nonzero delay snapshot continuation,
vector-bit Torch fallback/Rust rejection, active-vector no-delay Rust parity,
and the existing scalar Rust parity node.

Current source SHA-256
`d6722238d7a09660125c69e5df42a2aec101b62eb9cfa6520a4146e00dbf4dc2`;
focused test
`67ba2bae4b598b2f3538c27a76c31b52571e86114cdbf9ce03f1bab41c8a4043`;
scalar contract test
`b365f061fb74353a724988d6fc266c286e38296107007e1a7746dca46bfffefd`;
JUnit
`7ac218c81de5d4b48b51657c811d69714d55f12d7760509e987c75f3f4591085`.

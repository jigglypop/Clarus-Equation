# Validation

Status: COMPLETE

Gate: PASS

## 1. Focused command

```powershell
.codex\hooks\python.cmd python `
  _workspace/ce/brainruntime-discrete-hybrid-property-loop-20260821/artifacts/a7_discrete_hybrid_witness.py
```

The final invocation exited `0` and wrote strict JSON with
`status=PROPERTY_PASS`. A second run to an independent temporary output had
the same SHA-256 as the canonical result:

```text
cc6954b7f8120fb231494a4cc5cc0498895270ff1a93dfa834027f3fb22c9e6a
```

## 2. Formula and runtime results

| gate | frozen requirement | observed | result |
|---|---:|---:|---|
| H-A Torch/mirror continuous max error | <= 2e-6 | 4.8676e-8 | PASS |
| H-B 24x24 Jacobian normalized error | <= 2e-6 | 2.6270e-12 | PASS |
| H-B minimum raw clip margin | >= 1e-4 | 6.9844e-2 | PASS |
| H-C worst final reachable-face error | <= 3e-6 | 6.5854e-8 | PASS |
| H-E calls t=0,1 recurrent norm | <= 1e-7 | 0, 0 | PASS |
| H-E call t=2 recurrent norm | >= 1e-4 | 3.5435e-3 | PASS |
| H-E next-tick lifecycle effect | >= 1e-4 | 2.3413e-2 | PASS |
| H-F float64 permutation residual | <= 1e-10 | 0 | PASS |
| H-F float32 runtime residual | <= 2e-6 | 0 | PASS |
| H-G no-delay backend state error | <= 1e-5 | 7.4506e-9 | PASS |
| H-G delay-on activation mismatch | >= 1e-4 | 3.3332e-2 | EXPECTED FAIL |

All bit/mask checks were exact. The ring counter was `1,2,3` after calls
`t=0,1,2`; the read slots were `0,1,0`. The delay-on adverse arm also recorded
Torch counter `1`, Rust counter `0`, Torch slot overwrite true, Rust buffer
unchanged true, and no delay-buffer/index identifiers in the frozen Rust
kernel.

## 3. Discrete and boundary receipts

- bit upper and lower perturbations crossed in the prescribed directions;
- eligibility and kth-boundary masks changed exactly;
- exact TopK tie remained `TIE_POLICY_UNSPECIFIED` and was excluded from
  derivative/backend parity;
- lifecycle below/above `.08` produced DORMANT/IDLE;
- every discrete crossing was labeled `UNDEFINED_DISCRETE_EVENT`, never a
  saltation or straight-through matrix;
- the unreachable STP-u/adaptation upper faces were not counted as runtime
  evidence.

## 4. Independent audits

The independent mathematics audit returned PASS with no P0/P1 defect and
independently reproduced the result. It verified the old-u STP derivative,
recurrent product rule, ring overwrite derivative, reachable-face split,
permutation boundary, and backend counterexample.

The independent status/provenance audit first returned REVISE only for the
missing package-version receipt. After the preserved P2-only correction, it
returned `Gate: PASS`: package version `0.2.10`, version source
`reality_stone.__version__`, source/contract/constant hashes current, all H-A
through H-G true, and no empirical asset opened.

## 5. Exact status

```text
DISCRETE_HYBRID_SPEC_PASS / RUNTIME_DELAY_PARITY_BLOCKED /
HETEROGENEOUS_THRESHOLD_RUNTIME_UNIMPLEMENTED / EMPIRICAL_UNTESTED
```

This is a mathematical/runtime property result. It is not evidence for AGI,
learning, cortical anatomy, physical manifold deformation, or neuron-specific
threshold implementation.

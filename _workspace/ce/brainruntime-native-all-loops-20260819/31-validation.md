# Validation

Status: IN_PROGRESS

## Code checks

- `git diff --check`: PASS.
- Focused native/STDP tests: `21 passed`.
- Adjacent runtime/temporal/replay STANDARD tests: `39 passed`.
- Two PyTorch sparse-CSR beta/invariant warnings were emitted; no test failed.

## Development results

Artifacts:

- `artifacts/development-results.json`: Route A first implementation, embedded result hash
  `e2ae1d81192243bcc236d6bd84e7891111b70492b8950850615f5758e6afa6ed`.
- `artifacts/development-results-v2.json`: final Route A development, embedded result hash
  `084f50b8c04b235eda9421a2903bec8703c27983980ad9ddfbd210050ed80d41`.
- `artifacts/route-b-development-results.json`: Route B development, embedded result hash
  `980faebb5c1b9934d39ed3fd4c83402031b96e1ab1b1c8c3435d0d3b735f0780`.

Final Route A development over seeds 97101--97108:

| Loop | GO seeds | Result |
|---|---:|---|
| 6 | 8/8 | GO |
| 7 | 8/8 | GO |
| 8 | 0/8 | STOP; clean recall 0 despite mean weight drift 5.7584 |
| 9 | 0/8 | STOP; held-out intervention accuracy 0 |
| 10 | 8/8 | GO; mean persistence-relative improvement 0.2763 |

Route B development:

| Loop | GO seeds | Result |
|---|---:|---|
| 8B | 0/8 | STOP; mean clean 0.0625, corrupt 0.03125, cosine gain +0.0949 |
| 9B | 0/8 | STOP; held-out accuracy and control advantage 0 |

The native and Route B recurrent matrices changed and remained finite, both stores stayed empty
through rollout, and fixed-codebook trajectory similarity sometimes improved. Neither mechanism
produced task-level binding or held-out intervention transfer.

## Pending locked stage

The preregistered confirmation seeds 98101--98132 have not been opened. Execution was rejected by
the approval boundary because a 32-seed scientific confirmation stage needs explicit user
authorization. No workaround was attempted.

The final stable-snapshot audit also found that applied causal weight direction, Loop 6-to-8
latest-valid wiring, and Loop 7 supplied-context precedence are not yet demonstrated in this
harness. Consequently, confirmation authorization is necessary but not by itself sufficient for
formal closure.

Authorized command after approval:

```powershell
C:\Users\dongh\AppData\Local\Programs\Python\Python311\python.exe -B -m reality_stone.clarus.runtime_native_loops_benchmark --confirmation --output _workspace\ce\brainruntime-native-all-loops-20260819\artifacts\confirmation-results.json
```

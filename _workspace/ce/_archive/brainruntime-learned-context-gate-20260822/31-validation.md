# Validation

Status: COMPLETE

## Focused test

Command:

```powershell
.codex\hooks\python.cmd pytest -q -p no:cacheprovider tests\test_runtime_context_learned_gate.py --junitxml=_workspace\ce\brainruntime-learned-context-gate-20260822\artifacts\focused-pytest.xml
```

Result: `3 passed` in `5.66 s`. JUnit SHA-256: `42088cc1e0873735da39f5c749f5916e1fb798c02b040d9b0ecb969daddd530c`. The only warnings were PyTorch's existing sparse-CSR beta/invariant warnings.

## Frozen development run

Command:

```powershell
.codex\hooks\python.cmd python -m reality_stone.clarus.runtime_context_learned_gate_benchmark --stage development --output _workspace\ce\brainruntime-learned-context-gate-20260822\artifacts\development-results.json
```

Result artifact SHA-256: `afc9d0aba4606f9dcc7a0370894c5b5682ceabfd9128b083fed5f46522d5f064`.

All 16 development seeds passed. Every one of the 23 pre-endpoint gates passed 16/16, including the independent $\Theta q$ reference, seed/$\sigma$/schedule invariance, cue swap, row-swapped-$\Theta$ counterfactual, local branch-use separation, exact budget, shared trunk, source immutability, and no target/decoder/endpoint reads. The smallest learned logit margin was `0.643174171447754`, well above the frozen `1e-6` tie boundary. All gate and recurrent digests remained unchanged after evaluation.

| Arm | Mean accuracy | Opposite payload | Mean runtime-energy proxy | Mean active fraction |
|---|---:|---:|---:|---:|
| `LEARNED` | 1.000000 | 0.000000 | 0.200115 | 0.096429 |
| `ORACLE` | 1.000000 | 0.000000 | 0.200115 | 0.096429 |
| `CONTEXT_SHUFFLE_TRAIN` | 0.000000 | 1.000000 | 0.200115 | 0.096429 |
| `WRONG_CUE` | 0.000000 | 1.000000 | 0.200115 | 0.096429 |
| `POST_CUE_SWAP` | 0.000000 | 1.000000 | 0.200115 | 0.096429 |
| `GATE_LESION_STATIC_0` | 0.500000 | 0.500000 | 0.200115 | 0.096429 |
| `STATIC_1` | 0.500000 | 0.500000 | 0.200115 | 0.096429 |
| `CANONICAL_CUE_MAP` | 0.500000 | 0.500000 | 0.200115 | 0.096429 |
| `RANDOM_MATCHED` | 0.283854 | 0.283854 | 0.227694 | 0.111607 |
| `FULL` | 0.000000 | 0.000000 | 0.253479 | 0.125000 |

The seed-specific task bijection was balanced at eight identity and eight swapped mappings. Consequently the fixed canonical cue mapping averaged exactly `0.5`; the learned selector could not pass by using cue index or seed parity. Correct, wrong, static, and oracle branch masks had identical recurrent-edge budget, delay, threshold, STP, runtime-energy proxy, and activity fraction. Wrong controls transmitted the other live payload rather than silencing the network.

Confirmation seeds `99601..99632` were not opened.

## Validation boundary

This is a deterministic synthetic comparison. The runtime-energy value is a dimensionless proxy, not joules. The result validates an experience-learned binary association over fixed gate actuators; it does not validate support discovery, held-out context composition, graph morphology, cortical biology, curvature-as-memory, disease intervention, or AGI.

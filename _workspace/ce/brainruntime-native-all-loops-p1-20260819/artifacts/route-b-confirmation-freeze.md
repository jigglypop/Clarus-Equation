# Frozen Route B confirmation candidate

Status: EXECUTED_AFTER_USER_APPROVAL

Frozen on 2026-08-19 after development seeds 97101--97108 and before any confirmation seed was
opened.

## Fixed protocol

- Confirmation seeds: 98101--98132.
- Runtime: Torch CPU, noise 0, axon delay off, `BrainRuntime.step()` trajectory only.
- Codebooks: independently seeded orthonormal cue and target blocks; no cue/value association is
  present in codebook construction.
- Write: `B_bounded_supervised_recurrent_projection`, gain 1.0, delta Frobenius bound 5.0.
- Evaluation: cue drive gain 5.0 followed by six zero-input WAKE steps.
- Cutoff: temporal rows 0, hippocampal rows 0 before and after every probe.
- Loop 8 gates: clean >= 0.8, corrupt >= 0.65, unknown abstention >= 0.95,
  attractor gain >= 0.05, control advantage >= 0.2, finite nonzero installed write, zero-store audit.
- Loop 9 gates: held-out `(1,1)` accuracy >= 0.7, control advantage >= 0.2, finite nonzero installed
  write, zero-store audit.
- Target-shuffled controls preserve the cue schedule and write procedure and change target
  assignment only.

## Frozen source identity

- Base commit: `fa73f9aecb4cdc2caf3e335f6d1a1725fd77fded` plus the recorded working-tree patch.
- `runtime_native_loops.py` SHA-256:
  `7a2611c2b776768cd290a7d9caae7bedf1470e0944b1d8fb4bf8595cd1aad4ed`.
- `runtime_native_loops_benchmark.py` SHA-256:
  `68492665aca903c2a6bee08a924071ebf2f819f2e6618ba19c83b36e82dd2410`.
- `tests/test_runtime_native_loops.py` SHA-256:
  `836e2132ab7778dc775cee72e4735c792b37c057e12a8c7a8ebcf9ae4daf8577`.
- Selected development artifact SHA-256:
  `f0825e5404d8c27bd1401aa3499d5025ca758572eb31514f18f0672e3e0831b3`.
- Selected embedded result SHA-256:
  `3329c2a333d6db8c6a314c72e3b25b2090ce6163c92ec4b0f8758031933fe8a8`.

## Executed locked command

This exact command was executed only after explicit user authorization:

```powershell
$env:PYTHONPATH='reality_stone/python'
C:\Users\dongh\AppData\Local\Programs\Python\Python311\python.exe -B -m reality_stone.clarus.runtime_native_loops_benchmark --route-b --confirmation --output _workspace/ce/brainruntime-native-all-loops-p1-20260819/artifacts/route-b-confirmation-results.json
```

The resulting embedded SHA-256 is
`496c614d18b03a8898fd93156704c630d99b33da7d290c94d25b9f6d88a1b111`; the artifact-file SHA-256
is `a4bc4821b19a735f7fd5451816934e8ad788f7a38ff79a34e28cdc0641919dc9`.

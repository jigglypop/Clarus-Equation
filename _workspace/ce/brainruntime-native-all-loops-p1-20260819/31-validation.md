# Validation

Status: COMPLETE

## Focused validation

- `git diff --check`: PASS.
- Frozen-candidate focused and adjacent tests: **28 passed** in 3.96 s.
- CE research core gates: `OK build`, `OK final`.
- Coverage includes applied `delta[post, pre] > 0`, legacy-default preservation, independently
  orthogonal codebooks, Route B task gates, physical cutoff, query-order invariance, latest-valid
  temporal selection, deletion exclusion, and Loop 7 context precedence.
- PyTorch emitted two known sparse-CSR beta/invariant warnings; no test failed.

## Selected development evidence

Route A artifact: `artifacts/development-results-p5-orthocode.json`, embedded hash
`348dda0af756da3cd6c419867e09eed9a05e4948ef0f441b05083dfe7252b20c`.

| Route A loop | GO seeds | Result |
|---|---:|---|
| 6 | 8/8 | GO |
| 7 | 8/8 | GO |
| 8 | 0/8 | STOP; causal STDP did not bind cue to target |
| 9 | 0/8 | STOP; no held-out transfer |
| 10 | 8/8 | GO; mean persistence-relative improvement 0.276272 |

Selected Route B artifact: `artifacts/route-b-development-results-p8-orthocode.json`, embedded hash
`3329c2a333d6db8c6a314c72e3b25b2090ce6163c92ec4b0f8758031933fe8a8`.

| Route B loop | GO seeds | Main metric | Control |
|---|---:|---|---|
| 8B | 8/8 | clean 1.0; corrupt mean 0.9375; gain mean 0.291224 | advantage 1.0 |
| 9B | 8/8 | held-out accuracy 1.0 | shuffled accuracy 0; advantage 1.0 |

Every selected Route B probe had zero temporal and hippocampal rows after cutoff and rollout.
Installed write norms were finite and bounded: 3.6470--3.7338 for Loop 8B and 2.6315--2.7308 for
Loop 9B.

## Confirmation evidence

Both frozen commands were executed after explicit user authorization. Each artifact contains exactly
32 unique seeds spanning 98101--98132.

Route A artifact: `artifacts/confirmation-results.json`, embedded hash
`c3ddbe2bffc80f27690aa1fbddfb2bc713862fe790002184ab1b7971ff5df45c`, artifact-file hash
`2fd40c7e32f2ed8b143701bc517393b7df279d36a293483a6846279863726633`.

| Route A loop | GO seeds | Confirmation result |
|---|---:|---|
| 6 | 32/32 | GO |
| 7 | 32/32 | GO; context precedence and zero-read audit passed |
| 8 | 0/32 | STOP; clean accuracy 0, mean gain 0.013666 |
| 9 | 0/32 | STOP |
| 10 | 32/32 | GO; mean improvement 0.282425, minimum 0.217379 |

Route B artifact: `artifacts/route-b-confirmation-results.json`, embedded hash
`496c614d18b03a8898fd93156704c630d99b33da7d290c94d25b9f6d88a1b111`, artifact-file hash
`a4bc4821b19a735f7fd5451816934e8ad788f7a38ff79a34e28cdc0641919dc9`.

| Route B loop | GO seeds | Confirmation result | Control |
|---|---:|---|---|
| 8B | 32/32 | clean 1.0; corrupt mean 0.976562 (min 0.75); gain mean 0.299547 | advantage min 1.0 |
| 9B | 32/32 | held-out accuracy 1.0 | shuffled 0; advantage min 1.0 |

All native, no-write/no-replay, and shuffled probes retained zero temporal and hippocampal rows
through rollout. All weights and scores were finite, all Route A snapshot parity/source-selection
audits passed, and installed Route B write norms stayed below the frozen bound 5.0.

## Interpretation

The combined actual-runtime development result now covers every loop: Route A supplies Loops 6,
7, and 10; supervised bounded Route B supplies Loops 8 and 9. Route A's failed STDP result remains
part of the evidence and is not replaced by Route B.

The combined actual-runtime confirmation result covers every loop: Route A supplies Loops 6, 7,
and 10; supervised bounded Route B supplies Loops 8 and 9. Route A's failed STDP result remains part
of the evidence and is not replaced by Route B.

# V10 local/cloud registered development contract

Status: COMPLETE

PREDECESSOR: `_workspace/ce/agi-v10-local-cloud-kernel-20260812`

## Question

On fresh synthetic episodes, does the frozen local/private plus shared-state transition kernel
outperform both local-only and cloud-only factorial controls, and do registered transition
lesions remove that increment when the intact train-only readout is reused?

## Frozen scope

- Candidate: `full` local/cloud kernel.
- Factorial controls: `local_only`, `cloud_only`, `no_memory`.
- Lesions: full-kernel `cross_cut` on all ticks; `local_reset` and `cloud_reset` on the decision
  tick only. Lesions reuse the intact full readout and are never refit.
- Every arm exposes 20 state features and uses the same fixed ridge rule.
- Development: 64 exact seeds, 256 train and 256 evaluation episodes per seed.
- Confirmation seed roles are reserved but remain unopened in this run.

## Primary gates

All must pass:

1. full mean accuracy at least `0.60`;
2. paired full-minus-strongest-factorial 95% bootstrap LCB at least `0.05`;
3. paired interaction `full-local_only-cloud_only+no_memory` LCB at least `0.05`;
4. cross-cut, decision-local-reset, and decision-cloud-reset paired loss LCB each positive;
5. no duplicate/overlapping/burned seed, nonfinite result, label-state bypass, or hash mismatch.

This is a narrow mechanism development test. GO does not establish AGI, whole-brain biology,
or SCC necessity.

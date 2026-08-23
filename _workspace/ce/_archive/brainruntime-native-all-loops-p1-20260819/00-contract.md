# BrainRuntime-native Loops 6--10 P1 correction contract

Status: COMPLETE

PREDECESSOR: `_workspace/ce/brainruntime-native-all-loops-20260819`

## Scope

This light continuation corrects exactly the three stable-snapshot audit gaps that prevent the
predecessor's integrated harness from satisfying its own contract:

1. prove the applied causal weight invariant $\Delta W_{\mathrm{post},\mathrm{pre}}>0$ under a
   positive gate while preserving legacy-default behavior;
2. make Loop 8 replay episodes come from actual latest-valid `TemporalAuditedMemory` selection,
   including stale update and deletion cases, rather than iterating an independent target array;
3. include supplied-context precedence in Loop 7's executable metrics and gate, with zero memory
   reads for that branch.

No development or confirmation thresholds change. Route A and Route B development artifacts in
the predecessor remain immutable evidence. The 98101--98132 confirmation seeds remain unopened
until explicit user authorization.

## Acceptance

- A focused asymmetric applied-update test passes for causal orientation and shows the legacy
  default remains unchanged.
- Loop 8's report includes a replay-source audit proving selected evidence IDs/values match
  latest-valid temporal recall and deleted keys do not enter replay.
- Reversing arrival order cannot change the selected replay episodes; replacing valid-time
  selection with arrival order must fail the audit fixture.
- Loop 7 reports context precedence accuracy 1.0 and zero context-branch memory reads.
- Existing focused and adjacent runtime tests remain green.
- Development seeds may be rerun; confirmation seeds may not.

## Claim boundary

This correction improves executable integration evidence only. It does not change the predecessor
development verdicts: native Loop 8 and Loop 9 remain STOP unless a separately preregistered
mechanism succeeds, and Loop 10 remains bounded self-prediction rather than consciousness.

# P1 implementation

Status: COMPLETE

## Applied changes

- Added an asymmetric live-runtime invariant proving that an ordered pre-then-post event with a
  positive external gate produces finite `BrainRuntime.weight[post, pre] > 0` in causal mode.
  The public default remains `stdp_orientation="legacy"`.
- Extended Loop 7 with a conflicting supplied-context probe. Its executable GO gate now requires
  context precedence accuracy 1.0, zero temporal-memory reads on that branch, and the existing
  disabled-memory invariants.
- Replaced Loop 8's independent target iteration with a replay-source manifest resolved through
  `TemporalAuditedMemory.recall`. The fixture contains a stale update and a later deletion; only
  the latest valid non-deleted records become replay episodes.
- Added reverse-arrival equality and arrival-last negative-control fields, and made both mandatory
  for Loop 8 GO. The no-replay control receives the same manifest but marks every item unreplayed.

## Scope boundary

No threshold was relaxed. No confirmation seed was opened. The implementation changes auditability
and source wiring; it does not turn the failed native association mechanism into a successful one.

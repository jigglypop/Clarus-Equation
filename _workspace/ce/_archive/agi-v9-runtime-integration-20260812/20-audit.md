# Formal status audit

Status: COMPLETE

Gate: PASS

## Binding decisions

| Item | Status | Decision |
|---|---|---|
| Cosine action-evidence encoder | derived finite map | implement with finite and zero-norm guards |
| Tower observation and token issuance | inherited unit mechanism | reuse public API only |
| V9 policy as action source | implementation invariant | required; external output bypass forbidden |
| Default RuntimeAgent behavior | compatibility invariant | must remain unchanged when disabled |
| Belief control plus V9 | unresolved composition | reject simultaneous enablement |
| Dormant depth tolerance/hysteresis | unused DOF | remove |
| Parameter-count metadata | mislabeled metadata | rename; forbid capacity/MAC interpretation |
| V9 efficacy / AGI / biology | empirical claim | blocked and untested |

## Allowed implementation

- Modify `agent.py` and package exports for an opt-in V9 branch.
- Modify the V9 spec/manifest only for the two P2 cleanup items.
- Add or update focused tests and one deterministic non-evidence demo.
- Update CodeMap and the V9 canonical document after validation.

## P0/P1 status

No open P0 or P1 blocks this isolated implementation. Any discovered output bypass,
non-atomic failure, broken legacy behavior, invalid dimension, or stale-token acceptance is P1
and must be repaired before the build gate closes.

## Evidence lock

This PASS authorizes code and unit/property validation only. It does not authorize the V9
256-seed development run, V8 locked data access, confirmation, biological inference, or an AGI
claim.

# Revision 2 — explicit parity receipts only

Status: COMPLETE

The stable-snapshot audit found no P0 and reproduced the Revision 1 numerical
result, but marked a P1 receipt gap: correct/wrong depth, delay, threshold,
STP, decoder, and context-to-state exclusions were consequences of the source
rather than explicit pre-endpoint machine gates.

Revision 2 changes no learning equation, mask, route score, decoder threshold,
timing, configuration, or seed.  It adds those explicit gates to `_preflight`
and removes the `context` argument from `_rollout`; hidden-branch norms are now
returned for both branches and labeled only after the state computation.

# Pre-implementation audit

Status: COMPLETE

Gate: PASS

The predecessor failure is carried forward without reinterpretation.  The
Revision 1 mechanism changes the causal seam rather than retuning the failed TR2
endpoint: two payloads are simultaneous, context is absent from the state and
decoder, and only the entry branch changes while the trunk is shared.

Revision 0 failed because an unnecessary third hop did not emit a delayed
packet.  Its support and hidden-activation receipts rule out decoder leakage;
the change to two hops is a recorded formula revision, while all thresholds,
decoder gates, payloads, seeds, and delay semantics remain frozen.

A stable-snapshot audit then found no formula or outcome defect but required
explicit pre-endpoint receipts for the already-shared delay, threshold, STP,
decoder, and correct/wrong profiles.  Revision 2 adds only those gates and
removes the context argument from the state-rollout function; route scores,
learning, masks, thresholds, decoder, timing, and seed lists are unchanged.

The equations are dimensionally closed in normalized runtime units.  Exact
support, rank, bypass, cutoff, budget, and AST receipts run before endpoint
scoring.  The claim ceiling is synthetic context-branch selection only.

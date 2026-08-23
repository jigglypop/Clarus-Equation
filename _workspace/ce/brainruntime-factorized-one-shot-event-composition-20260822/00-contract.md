# BA-TR20 one-shot source-event composition

Status before Revision 2 execution: `CALIBRATION_ONLY / DEVELOPMENT_SEALED`.

BA-TR19 established source-factorized composition under a repeated source
packet stream. BA-TR20 freezes the learned weights, delay, horizon, decoder
floor, and competition equation, and changes only the source event apparatus.

For the native presynaptic packet (p_t), declared source projector (P_S),
and a source cue at tick zero,

\[
\widetilde p_t=(I-P_S)p_t+\mathbf1[t=1]P_Sp_t.
\]

Revision 0 attempted to realize this by clamping source activation after tick
1. Calibration seed `107001` falsified that apparatus: retained refractory
state generated a negative rebound and a new source packet at tick 5. The
failed result is retained in `calibration-results.json`; no development input
was opened.

Revision 1 implements the equation directly without changing BrainRuntime.
After every step it edits only the source coordinates of the ring slot just
written: tick 1 is retained and all other source-coordinate writes are zero.
It never clears the already emitted tick-1 packet and never clamps neuronal
activation, refractory, adaptation, or memory state. With `L=2`, the exact
pair receipt must be `[0,0,0,2,0,0,0]`; the packet reaches H at tick 3, H emits
normally at tick 4, and Y is read at tick 6.

Revision 1 calibration seed `107002` then exposed a runtime formula mismatch:
`delayed_pre` was a view of the ring slot, so the subsequent ring overwrite
changed the packet seen by factorized competition from the arrived packet to
the newly emitted packet.  The failed result is retained in
`calibration-r1-results.json`.  The runtime repair clones `delayed_pre` before
overwrite; it changes no equation or threshold.

Fresh Revision 2 calibration seed: `107003`. Fresh development seeds, opened only after
calibration passes: `107101..107116`.

Every development seed must satisfy: atomic 4/4; factorized pair 4/4 with two
first-arrival H positives; legacy WTA 0/4; one-tick-misaligned provenance 0/4;
independent one-shot atomic union 4/4; suppressed event 0/4; unmodified stream
control 4/4; exact atomic/pair packet receipts; zero stores. Any failure is a
frozen STOP—no gain, horizon, threshold, weight, or clamp-time adjustment.

Claim ceiling on GO: deterministic synthetic two-source composition under an
explicit externally imposed one-shot delayed source-event clamp. This is a
sufficiency test, not evidence that native BrainRuntime or a biological neuron
naturally emits one-shot spikes.

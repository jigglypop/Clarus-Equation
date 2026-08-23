# BA-TR20 implementation

BA-TR20 keeps the BA-TR19 weights and source-factorized competition law, but
turns the source cue into an explicit one-shot axonal event. If (P_S) projects
onto the declared source coordinates and (p_t) is the native presynaptic
packet, the experimental packet is

\[
\widetilde p_t=(I-P_S)p_t+\mathbf1[t=1]P_Sp_t.
\]

The harness implements this after each runtime step by retaining source
coordinates in the ring slot written at tick 1 and zeroing only those
coordinates in every other newly written slot. Neuronal activation,
refractory, adaptation, STP, memory trace, lifecycle, downstream packets, and
the already emitted source packet are not clamped.

Revision history before development:

- R0 clamped source activation after tick 1. Refractory rebound generated a
  second signed source packet at tick 5, so the apparatus failed.
- R1 gated ring writes directly. Its suppressed-event control exposed a
  runtime alias: `delayed_pre` was a view of the ring slot and changed when the
  slot was overwritten later in the same step.
- R2 clones `delayed_pre` before overwrite. This restores the intended
  read-before-write equation. No weight, gain, horizon, decoder threshold, or
  competition parameter changed.

R0 and R1 calibration artifacts remain in the run directory. Development was
opened only after fresh R2 calibration seed `107003` passed.


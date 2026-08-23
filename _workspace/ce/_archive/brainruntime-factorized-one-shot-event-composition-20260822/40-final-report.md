# BA-TR20 final

Status: COMPLETE

Verdict: `SYNTHETIC_FACTORIZED_ONE_SHOT_EVENT_COMPOSITION /
DEVELOPMENT_GO / CONFIRMATION_SEALED`.

The strengthened test passed on all 16 fresh development seeds. A single
pair-valued source packet was written once, arrived once after the exact delay,
and produced both learned target coordinates in all 64 pair probes. Removing
the packet or shifting its declared provenance by one tick produced 0/64;
global WTA also produced 0/64.

The operative equation is therefore not an adaptive top-(K) on the mixed
field. It preserves source provenance until after local route selection:

\[
c_h(t)=\sum_{s:p_s(t)\ne0}
\left[
[W_{hs}p_s(t)]_+-\max_{k\ne h}[W_{ks}p_s(t)]_+
\right]_+ .
\]

The R1 suppressed-event control also found and closed a real implementation
error: a ring-slot view caused the factorized rule to inspect the newly written
packet rather than the packet delivered that tick. BA-TR18/19 artifacts were
created before that repair and must not be cited as correct delayed-provenance
evidence; BA-TR20's fresh post-repair seeds supersede them.

What is established is narrow but concrete: within this synthetic learned
source-hidden-target circuit, provenance-preserving selection is sufficient
for exact two-item composition from one delayed source event. The event latch
is externally imposed by the harness. This does not show that native
BrainRuntime generates spikes, that biological inhibition uses this rule, or
that curvature stores memory.

Next falsifier: replace the externally declared source identity/projector with
a locally carried synaptic packet tag or an endogenous event representation,
then test whether the same composition survives without an experiment-owned
source mask.


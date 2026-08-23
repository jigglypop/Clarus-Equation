# BA-TR14 experience-only delayed binding contract

Date: 2026-08-22

Input is a frozen BA-TR10 learned `S -> H` matrix. The new `Y <- H` block starts
at zero. For each of four micro-experiences with axon delay `L=2`, source is
presented at call 0, hidden first responds at call 3, and the target sensory
experience is externally presented only at call 6.

At the true first hidden arrival, each presynaptic neuron writes its own binary
event trace. The trace is retained locally until the target experience:

\[
z_h=\mathbf 1[a_h(3)>10^{-6}],
\qquad
E_{yh}\leftarrow E_{yh}+[a_y(6)]_+z_h,
\qquad
\Delta W_{yh}=\eta E_{yh},\quad \eta=1.
\]

This Revision 1 is an apparatus correction made after the focused calibration
seed showed that using only the residual call-6 axon packet attenuated the
write twice and produced a maximum recalled target of order `1e-7`. No
development row was opened. The event trace is dimensionless, coordinate-local,
and derived only from the actual hidden arrival; it is neither a target label
nor a route selector.

Here the initial `W_YH=0`, so the first block is the local coincidence sum.
There is no structural projection, threshold pruning, row normalization,
top-k write, per-pair write, target label, decoder, reward, or endpoint read in
the learner. One bounded install occurs only after all four experiences, and
raw/installed deltas must match at `1e-7`.

Before evaluation, eligibility and transient state are discarded, temporal and
hippocampal stores are physically empty, and a sealed snapshot is made. Each
probe presents only a source at call 0 and zeros at calls 1..6. Target decoding
occurs offline after call 6.

Required controls: eta zero, cyclic target-experience shuffle, target-before-
source reversal, no target experience, uniform/reset `S -> H` code, and random
same-norm `Y <- H`. Learned accuracy must be 1.0, strongest original-target
control at most .5, advantage at least .5, shuffled training must reproduce its
shuffled association, and all support/timing/store/snapshot gates must pass.

Claim ceiling: synthetic experience-supervised local delayed association on
declared S/H/Y blocks. No semantic, biological, curvature, morphology, or AGI
claim.

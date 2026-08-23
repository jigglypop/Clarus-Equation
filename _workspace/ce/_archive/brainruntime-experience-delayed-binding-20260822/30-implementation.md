# BA-TR14 implementation

The experiment consumes each frozen BA-TR10 learned `4 x 4` source-to-hidden
matrix and creates a fresh 20-neuron Torch runtime. The `Y <- H` block starts
at zero.

For each source experience, the source is presented at call 0. With the frozen
two-tick axon ring, the first hidden response occurs at call 3. A local binary
presynaptic eligibility trace is set there,

\[
z_h=\mathbf 1[a_h(3)>10^{-6}],
\]

and is retained only until the sensory target pulse at call 6. The block
accumulator is

\[
E_{yh}\leftarrow E_{yh}+[a_y(6)]_+z_h.
\]

After all four experiences, exactly one additive install applies
`Delta W_YH = E_YH`. No structural projection, threshold pruning, row
normalization, top-k write, per-pair mutation, decoder, reward, or endpoint is
used by the learner.

The initial calibration formula used the residual hidden axon packet at call 6
instead of the retained first-arrival event. Its write norm was about `8e-4`
and cue-only target activation was only `1e-8` to `1e-7`. Revision 1 replaced
that attenuated quantity before the 16-row development artifact was opened.

Before every probe, temporal and hippocampal stores are physically empty and a
sealed snapshot is cloned. A probe presents only the source at call 0 and zero
external input at calls 1 through 6.


# BA-TR21 implementation

BA-TR21 removes the experiment-declared source projector from BA-TR20. The
factorized competition receives every coordinate in the 20-dimensional
runtime chart and acts on whichever delayed packet entries are actually
nonzero:

\[
c_h(t)=\sum_{j:\lvert p_j(t)\rvert>\epsilon}
\left[
[W_{hj}p_j(t)]_+-\max_{k\ne h}[W_{kj}p_j(t)]_+
\right]_+ .
\]

No new runtime dynamics were added. The all-input snapshot sets
`competition_input_indices=(0,...,19)` while retaining the repaired
read-before-write delay, one-shot packet gate, learned weights, and target
threshold. A matched upper reference uses the old four-coordinate source
projector. An adverse control cyclically permutes only the learned source
columns of (W), preserving column norms while breaking packet-column
identity.


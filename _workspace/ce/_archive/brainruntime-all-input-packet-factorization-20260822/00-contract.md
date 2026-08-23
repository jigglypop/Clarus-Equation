# BA-TR21 all-input packet factorization

Status before execution: `CALIBRATION_ONLY / DEVELOPMENT_SEALED`.

BA-TR20 required an experiment-declared source projector (P_S). BA-TR21
removes that projector from the competition input set. For every nonzero
coordinate of the actual arriving delayed packet,

\[
c_h(t)=\sum_{j:\lvert p_j(t)\rvert>\epsilon}
\left[
[W_{hj}p_j(t)]_+-\max_{k\ne h}[W_{kj}p_j(t)]_+
\right]_+ .
\]

The compiler receives the full coordinate set `0..dim-1`; it does not receive
the source block. Packet column identity is the ordinary presynaptic neuron
index already present in the weight matrix. The BA-TR20 one-shot ring gate,
weights, delay, horizon, and `1e-5` target-set threshold remain frozen.

Fresh calibration seed: `108001`. Fresh development seeds: `108101..108116`,
opened only after calibration passes.

Per seed gates: all-input atomic 4/4; all-input and source-projected pair
outputs bit-exact and 4/4; legacy WTA 0/4; cyclic source-column shuffle 0/4;
suppressed event 0/4; independent union 4/4; exact one-shot packet receipt;
full coordinate input-set receipt; zero stores. Any failure is STOP without
changing support, threshold, horizon, weights, or seed list.

Claim ceiling: the explicit source projector is unnecessary in this sparse
synthetic circuit. Learned nonzero weight support still identifies which
packet columns can affect H; this does not establish support discovery or
robustness to broad nonzero distractor connectivity.


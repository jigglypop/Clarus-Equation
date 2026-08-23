# BA-TR22 implementation

The runtime source-factorized sum was first generalized from exactly two to
every active delayed packet coordinate, matching the BA-TR21 equation. The
test then adds one fixed distractor coordinate outside the learned source
block. For each pair, its H-input column is an exact copy of a learned source
column absent from that pair. This preserves local weight scale and route
quality without supplying target or decoder information to the runtime.

All three events receive the same external amplitude, one-shot ring gate, and
delay. No context, task goal, reward, target label, or relevance signal is
provided. The expected task readout remains the original pair; the third route
is recorded separately as the adverse identity witness.


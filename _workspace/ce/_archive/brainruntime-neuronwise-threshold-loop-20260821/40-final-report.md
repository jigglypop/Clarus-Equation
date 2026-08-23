# Neuronwise runtime thresholds

Status: COMPLETE

The runtime now represents heterogeneous active-selection and bit-hysteresis
thresholds directly instead of absorbing them into a scalar or into $W$.
Legacy scalar configs remain the exact broadcast path, and mutable config
changes are resolved on the next use. Snapshot restore preserves the vector
configuration without an extra cached tensor state.

This closes an implementation gap only. The chosen threshold vectors in
synthetic experiments are fixtures, not measured neuronal thresholds. No
biological threshold distribution, cortical folding, memory geometry,
learning advantage, disease mechanism, or AGI claim follows. Rust still lacks
vector-bit and axonal-delay ABI support and must fail closed in those domains.

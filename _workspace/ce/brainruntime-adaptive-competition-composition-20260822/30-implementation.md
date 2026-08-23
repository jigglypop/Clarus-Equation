# BA-TR17 implementation

The runtime gains an opt-in, Torch-only competition budget derived from the
explicit source coordinates in the axon packet being delivered now. Default
configuration executes the prior branch unchanged. Adaptive configuration is
structural, snapshot-restored, requires true delay and lateral gain exactly 1,
and remains Rust-inadmissible under the existing local-competition guard.

For one source packet the old peer-maximum code path is used byte-for-byte. For
two packets, the third-largest attenuated H value is subtracted from all H
values; a 2/3 boundary tie returns all zeros. More than two packets is outside
the admitted capacity and also returns zero.

The composition runner trains only atomic cyclic associations with BA-TR15,
then compares aligned adaptive count, legacy count-blind WTA, misaligned count,
and independent-union probes on sealed zero-store snapshots.


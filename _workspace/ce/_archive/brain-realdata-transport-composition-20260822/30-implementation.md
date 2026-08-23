# Implementation

Status: COMPLETE

The isolated implementation is
`reality_stone/python/reality_stone/clarus/realdata_transport_composition.py`.
It loads only `cont_data`, keeps session/condition identities separate,
extracts the frozen phase windows, constructs a train-only common latent chart,
fits the three affine maps, and records every held-out control SSE. The CLI
writes finite JSON and hashes every source MAT file. No BrainRuntime or
synthetic routing code is modified.
